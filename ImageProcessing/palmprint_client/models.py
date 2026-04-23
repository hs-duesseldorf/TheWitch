from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .constants import IMAGENET_MEAN, IMAGENET_STD, MODEL_INPUT_SIZE, MODEL_RESIZE_SIZE
from .embedding import load_embedding_checkpoint
from .preprocessing import RoiToneSettings, prepare_cnn_input_roi
from .utils import l2_normalize


def create_hand_landmarker(model_path: Path) -> mp_vision.HandLandmarker:
    if not model_path.exists() or model_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing MediaPipe hand landmarker task file: {model_path}")
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


class TextureCNN:
    def __init__(self, device: torch.device, weights_path: Path, roi_tone_settings: RoiToneSettings):
        self.device = device
        self.weights_path = weights_path
        self.roi_tone_settings = roi_tone_settings
        self.model, self.preprocess, self.source, self.input_mode = self._build_model()

    def _default_preprocess(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(MODEL_RESIZE_SIZE),
                transforms.CenterCrop(MODEL_INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _palm_preprocess(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(MODEL_RESIZE_SIZE),
                transforms.CenterCrop(MODEL_INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _load_imagenet_model(self) -> tuple[nn.Module, transforms.Compose, str]:
        try:
            weights = models.ResNet18_Weights.DEFAULT
            return models.resnet18(weights=weights), weights.transforms(), "imagenet_default"
        except Exception:
            try:
                return models.resnet18(pretrained=True), self._default_preprocess(), "imagenet_legacy"
            except Exception:
                return models.resnet18(weights=None), self._default_preprocess(), "random_init"

    def _build_model(self) -> tuple[nn.Module, transforms.Compose, str, str]:
        preprocess = self._default_preprocess()
        source = "random_init"
        input_mode = "rgb_imagenet"

        if self.weights_path.exists():
            try:
                model, checkpoint = load_embedding_checkpoint(self.weights_path, device=self.device)
                source = f"local:{self.weights_path.name}"
                preprocess = self._palm_preprocess()
                input_mode = checkpoint.get("input_mode", "palm_gray")
            except Exception:
                model = models.resnet18(weights=None)
                try:
                    state = torch.load(str(self.weights_path), map_location="cpu")
                    if "fc.weight" in state:
                        model.fc = nn.Linear(model.fc.in_features, state["fc.weight"].shape[0])
                    model.load_state_dict(state, strict=True)
                    source = f"local:{self.weights_path.name}"
                    preprocess = self._palm_preprocess()
                    input_mode = "palm_gray"
                    model = nn.Sequential(*list(model.children())[:-1])
                except Exception:
                    model, preprocess, source = self._load_imagenet_model()
                    source = f"local_failed->{source}"
                    model = nn.Sequential(*list(model.children())[:-1])
        else:
            model, preprocess, source = self._load_imagenet_model()
            model = nn.Sequential(*list(model.children())[:-1])

        model = model.eval().to(self.device)
        if self.device.type == "cuda":
            model = model.half()
        return model, preprocess, source, input_mode

    def _forward_batch(self, images: list[Image.Image]) -> np.ndarray:
        x = torch.stack([self.preprocess(image) for image in images], dim=0).to(self.device)
        if self.device.type == "cuda":
            x = x.half()
        with torch.inference_mode():
            feat = self.model(x).flatten(1)
        matrix = feat.detach().cpu().numpy().astype(np.float32)
        return matrix

    def _embed_palm_roi(self, roi_bgr: np.ndarray) -> np.ndarray:
        gray = prepare_cnn_input_roi(roi_bgr, self.roi_tone_settings)
        vec = self._forward_batch([Image.fromarray(gray, mode="L")])[0]
        return l2_normalize(vec)

    def embed(self, roi_bgr: np.ndarray) -> np.ndarray:
        if self.input_mode == "palm_gray":
            return self._embed_palm_roi(roi_bgr)

        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        vec = self._forward_batch([pil])[0]
        return l2_normalize(vec)
