from __future__ import annotations

from pathlib import Path

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

WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "assets" / "weights"
DEFAULT_EMBEDDING_MODEL = "arcface"
EMBEDDING_MODELS: dict[str, tuple[str, Path]] = {
    "arcface": ("Tongji ArcFace", WEIGHTS_DIR / "tongji_resnet18_arcface_256d.pt"),
    "contrastive": ("Tongji Contrastive", WEIGHTS_DIR / "tongji_resnet18_contrastive_person_256d.pt"),
}


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
    def __init__(
        self,
        *,
        device: torch.device,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        roi_tone_settings: RoiToneSettings,
    ):
        self.device = device
        self.model_id = embedding_model
        self.weights_path = self._model_path(embedding_model)
        self.display_name = EMBEDDING_MODELS[self.model_id][0]
        self.roi_tone_settings = roi_tone_settings
        self.model, self.preprocess, self.source, self.input_mode, self.embedding_dim = self._build_model()

    def _model_path(self, model_id: str) -> Path:
        if model_id not in EMBEDDING_MODELS:
            raise ValueError(f"Unknown embedding model: {model_id}")
        return EMBEDDING_MODELS[model_id][1]

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

    def _load_legacy_resnet18(self, weights_path: Path) -> nn.Module:
        state = torch.load(str(weights_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError(f"{weights_path} does not contain a state dict")

        state = {str(key).removeprefix("module."): value for key, value in state.items()}
        model = models.resnet18(weights=None)
        if "fc.weight" in state:
            model.fc = nn.Linear(model.fc.in_features, state["fc.weight"].shape[0])
        model.load_state_dict(state, strict=True)
        return nn.Sequential(*list(model.children())[:-1])

    def _build_model(self) -> tuple[nn.Module, transforms.Compose, str, str, int]:
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Missing embedding model weights: {self.weights_path}")

        try:
            model, checkpoint = load_embedding_checkpoint(self.weights_path, device=self.device)
            input_mode = str(checkpoint.get("input_mode", "palm_gray"))
            embedding_dim = int(checkpoint.get("embedding_dim", 256))
            return self._prepare_model(model), self._palm_preprocess(), f"local:{self.weights_path.name}", input_mode, embedding_dim
        except Exception:
            model = self._load_legacy_resnet18(self.weights_path)
            return self._prepare_model(model), self._palm_preprocess(), f"local:{self.weights_path.name}", "palm_gray", 512

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        model = model.eval().to(self.device)
        if self.device.type == "cuda":
            model = model.half()
        return model

    def _forward_batch(self, images: list[Image.Image]) -> np.ndarray:
        x = torch.stack([self.preprocess(image) for image in images], dim=0).to(self.device)
        if self.device.type == "cuda":
            x = x.half()
        with torch.inference_mode():
            feat = self.model(x).flatten(1)
        return feat.detach().cpu().numpy().astype(np.float32)

    def embed(self, roi_bgr: np.ndarray) -> np.ndarray:
        gray = prepare_cnn_input_roi(roi_bgr, self.roi_tone_settings)
        return self.embed_preprocessed(gray)

    def embed_preprocessed(self, roi_gray: np.ndarray) -> np.ndarray:
        gray = roi_gray.astype(np.uint8, copy=False)
        vec = self._forward_batch([Image.fromarray(gray, mode="L")])[0]
        return l2_normalize(vec)

    def status_payload(self) -> dict[str, object]:
        return {
            "id": self.model_id,
            "display_name": self.display_name,
            "source": self.source,
            "input_mode": self.input_mode,
            "embedding_dim": self.embedding_dim,
        }
