from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


MODEL_PATH = Path("hand_landmarker.task")
WEIGHTS_PATH = Path("tongji_resnet18_arcface_256d.pt")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_INPUT_SIZE = 224
MODEL_RESIZE_SIZE = 256

# code from imageprocessing / vision - marcel

def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _normalize_vector(vector: np.ndarray, eps: float = 1e-6) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return None
    return (vector / norm).astype(np.float32)


def _perpendicular(vector: np.ndarray) -> np.ndarray:
    return np.array([-vector[1], vector[0]], dtype=np.float32)

def _finger_length(points: np.ndarray, ids: tuple[int, int, int, int]) -> float:
    a, b, c, d = ids
    return _distance(points[a], points[b]) + _distance(points[b], points[c]) + _distance(points[c], points[d])


def palm_facing_score(points3d: np.ndarray, handedness_str: str) -> float | None:
    points = np.asarray(points3d, dtype=np.float32)
    wrist = points[WRIST, :3]
    index_mcp = points[INDEX_MCP, :3]
    pinky_mcp = points[PINKY_MCP, :3]
    normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
    normal_unit = _normalize_vector(normal)
    if normal_unit is None:
        return None

    camera_direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    facing_score = float(np.dot(normal_unit, camera_direction))

    if handedness_str == "right":
        return facing_score
    if handedness_str == "left":
        return -facing_score
    return None


def is_palm_frontal(points3d: np.ndarray, handedness_str: str, *, min_confidence: float = 0.82) -> bool:
    score = palm_facing_score(points3d, handedness_str)
    if score is None:
        return True
    return abs(score) >= min_confidence


def extract_geometry_features(points3d: np.ndarray) -> dict[str, float]:
    palm_width = _distance(points3d[INDEX_MCP], points3d[PINKY_MCP])
    palm_height = _distance(points3d[WRIST], points3d[MIDDLE_MCP])

    return {
        "palm_width": palm_width,
        "palm_height": palm_height,
        "thumb_length": _finger_length(points3d, (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP)),
        "index_length": _finger_length(points3d, (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)),
        "middle_length": _finger_length(points3d, (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP)),
        "ring_length": _finger_length(points3d, (RING_MCP, RING_PIP, RING_DIP, RING_TIP)),
        "pinky_length": _finger_length(points3d, (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP))
    }


class RoiPose:
    def __init__(self, center: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, palm_width: float,
                 palm_height: float):
        self.center = center
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.palm_width = palm_width
        self.palm_height = palm_height


def estimate_roi_pose(landmarks_px: np.ndarray) -> RoiPose | None:
    points_xy = np.asarray(landmarks_px[:, :2], dtype=np.float32)
    wrist = points_xy[WRIST]
    thumb_mcp = points_xy[THUMB_MCP]
    index_mcp = points_xy[INDEX_MCP]
    middle_mcp = points_xy[MIDDLE_MCP]
    ring_mcp = points_xy[RING_MCP]
    pinky_mcp = points_xy[PINKY_MCP]

    index_side = (index_mcp + middle_mcp) / 2.0
    pinky_side = (ring_mcp + pinky_mcp) / 2.0
    mcp_band_center = (index_mcp + middle_mcp + ring_mcp + pinky_mcp) / 4.0

    y_seed = mcp_band_center - wrist
    y_axis = _normalize_vector(y_seed)
    if y_axis is None: return None

    x_seed = index_side - pinky_side
    x_seed = x_seed - float(np.dot(x_seed, y_axis)) * y_axis
    x_axis = _normalize_vector(x_seed)
    if x_axis is None:
        x_axis = _normalize_vector(_perpendicular(y_axis))
        if x_axis is None: return None
    if float(np.dot(thumb_mcp - wrist, x_axis)) < 0.0:
        x_axis = -x_axis

    y_axis = _perpendicular(x_axis)
    if float(np.dot(y_axis, y_seed)) < 0.0:
        y_axis = -y_axis

    palm_width = 0.55 * float(np.linalg.norm(index_mcp - pinky_mcp)) + 0.45 * float(
        np.linalg.norm(index_side - pinky_side))
    palm_height = 0.65 * float(np.linalg.norm(mcp_band_center - wrist)) + 0.35 * float(
        np.linalg.norm(middle_mcp - wrist))
    if palm_width < 3.0 or palm_height < 3.0: return None

    return RoiPose(wrist + y_axis * (0.46 * palm_height), x_axis, y_axis, palm_width, palm_height)


def roi_quad_from_pose(pose: RoiPose) -> np.ndarray:
    top_center = pose.center + pose.y_axis * (0.50 * pose.palm_height)
    bottom_center = pose.center - pose.y_axis * (0.31 * pose.palm_height) + pose.x_axis * (0.065 * pose.palm_width)
    half_w_top, half_w_bottom, thumb_bias = 0.51 * pose.palm_width, 0.58 * pose.palm_width, 0.07 * pose.palm_width
    tl = top_center - pose.x_axis * (half_w_top - 0.02 * pose.palm_width)
    tr = top_center + pose.x_axis * (half_w_top + thumb_bias)
    br = bottom_center + pose.x_axis * (half_w_bottom + thumb_bias)
    bl = bottom_center - pose.x_axis * (half_w_bottom - 0.03 * pose.palm_width)
    return np.float32([tl, tr, br, bl])


def warp_roi_from_quad(frame_bgr: np.ndarray, quad: np.ndarray, roi_size: int) -> np.ndarray:
    dst = np.float32([[0, 0], [roi_size - 1, 0], [roi_size - 1, roi_size - 1], [0, roi_size - 1]])
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(frame_bgr, matrix, (roi_size, roi_size), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)

class ResNet18EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embedding = nn.Sequential(
            nn.Linear(in_features, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embedding = self.embedding(features)
        return F.normalize(embedding, dim=1)


def load_network_model(weights_path: Path, device: torch.device) -> nn.Module:
    model = ResNet18EmbeddingNet(embedding_dim=256)
    checkpoint = torch.load(str(weights_path), map_location="cpu")
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return model

# processing batch
def process_mohi_dataset_with_task(dataset_dir: str, output_json_path: str):
    if not MODEL_PATH.exists():
        print(f"{MODEL_PATH} is not exist")
        return

    if not WEIGHTS_PATH.exists():
        print(f"{WEIGHTS_PATH} is not exist")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = load_network_model(WEIGHTS_PATH, device)

    transform_pipeline = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(MODEL_RESIZE_SIZE),
        transforms.CenterCrop(MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # task ai setting
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.6
    )

    final_dataset_json = {}

    # init MediaPipe
    with HandLandmarker.create_from_options(options) as landmarker:
        if not os.path.exists(dataset_dir):
            print(f"no dataset in {dataset_dir}")
            return

        files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"dataset len total : {len(files)}")

        for idx, filename in enumerate(files):
            image_full_path = os.path.join(dataset_dir, filename)

            # check file name validity
            parts = filename.split('-')
            if len(parts) <= 1 or not ''.join(filter(str.isdigit, parts[1])):
                continue

            # convert image with open cv so that mediapipe can read.
            raw_frame = cv2.imread(image_full_path)
            if raw_frame is None:
                continue

            frame_height, frame_width = raw_frame.shape[:2]

            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # find landmark of hand
            result = landmarker.detect(mp_image)

            # no result
            if not result.hand_landmarks:
                print(f"{filename} -> no result")
                continue

            # left / right hand
            handedness_str = result.handedness[0][0].category_name.strip().lower()

            # convert to 3D
            world_landmarks = result.hand_world_landmarks[0]
            world_pts = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks], dtype=np.float32)

            # check frontal side
            if not is_palm_frontal(world_pts, handedness_str, min_confidence=0.82):
                print(f"{filename} -> tilted")
                continue

            # extract geometry
            geometry = extract_geometry_features(world_pts)
            if not geometry:
                continue

            normal_landmarks = result.hand_landmarks[0]
            pixel_pts = np.array([[lm.x * frame_width, lm.y * frame_height] for lm in normal_landmarks],
                                 dtype=np.float32)

            pose = estimate_roi_pose(pixel_pts)
            if pose is None: continue
            quad = roi_quad_from_pose(pose)

            warped_roi = warp_roi_from_quad(raw_frame, quad, roi_size=MODEL_RESIZE_SIZE)
            roi_gray = cv2.cvtColor(warped_roi, cv2.COLOR_BGR2GRAY)

            input_tensor = transform_pipeline(roi_gray).unsqueeze(0).to(device)
            with torch.inference_mode():
                embedding_vector = cnn_model(input_tensor).squeeze(0).cpu().numpy()

            person_id = filename.split('-')[1]
            input_key = f"{person_id}_input_{idx}"

            main_json_key = filename.rsplit('.', 1)[0]

            final_dataset_json[main_json_key] = {
                "request_id": f"{person_id}-{handedness_str}",
                "type": "hand",
                "trigger": "hand_detected",
                "hand": handedness_str,
                "lengths": {
                    key: round(float(value), 6)
                    for key, value in geometry.items()
                },
                "vector": [round(float(v), 6) for v in embedding_vector.tolist()],
                "source_file": filename
            }

        # dump json
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(final_dataset_json, f, indent=4, ensure_ascii=False)

        print(f"data len : {len(final_dataset_json)} : {output_json_path}")


if __name__ == "__main__":
    MOHI_FOLDER = "./hand_images"

    process_mohi_dataset_with_task(MOHI_FOLDER, "../analysis/hand_informs.json")
