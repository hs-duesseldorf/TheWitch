import cv2
import numpy as np
import mediapipe as mp
import json
import os
from pathlib import Path


MODEL_PATH = Path("hand_landmarker.task")

# MediaPipe 최신 Tasks API 핵심 모듈 정의
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

#code from imageprocessing / vision - marcel

def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def _normalize_vector(vector: np.ndarray, eps: float = 1e-6) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return None
    return (vector / norm).astype(np.float32)

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
        "pinky_length": _finger_length(points3d, (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP)),
    }

#processing batch

def process_mohi_dataset_with_task(dataset_dir: str, output_json_path: str):
    if not MODEL_PATH.exists():
        print(f"{MODEL_PATH} is not exist")
        return

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

            # 1. 파일 이름 유효성 선검사 (결측치 스킵)
            parts = filename.split('-')
            if len(parts) <= 1 or not ''.join(filter(str.isdigit, parts[1])):
                continue

            # convert image with open cv so that mediapipe can read.
            raw_frame = cv2.imread(image_full_path)
            if raw_frame is None:
                continue
            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # find landmark of hand
            result = landmarker.detect(mp_image)

            # no result
            if not result.hand_landmarks:
                print(f"[탈락] {filename} -> no result")
                continue

            # left / right hand
            handedness_str = result.handedness[0][0].category_name.strip().lower()

            # convert to 3D
            world_landmarks = result.hand_world_landmarks[0]
            world_pts = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks], dtype=np.float32)

            # check frontal side
            if not is_palm_frontal(world_pts, handedness_str, min_confidence=0.82):
                print(f"[탈락] {filename} -> tilted")
                continue

            # extract geometry
            geometry = extract_geometry_features(world_pts)
            if not geometry:
                continue

            person_id = filename.split('-')[1]
            input_key = f"{person_id}_input_{idx}"

            final_dataset_json[input_key] = {
                "request_id": f"{person_id}-{handedness_str}",
                "session_id": "session-mohi-task-verified",
                "handedness": handedness_str,
                "lengths": {
                    key: round(float(value), 6)
                    for key, value in geometry.items()
                }
            }

        # dump json
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(final_dataset_json, f, indent=4, ensure_ascii=False)

        print(f"data len : {len(final_dataset_json)} : {output_json_path}")


if __name__ == "__main__":
    MOHI_FOLDER = "./hand_images"

    process_mohi_dataset_with_task(MOHI_FOLDER, "hand_informs.json")

# # [참고] ISO 7250-1 규격 기반 성인 남녀 통합 표준 손 비율 데이터
# input_average = {
#     "palm_aspect_ratio": 0.81,  # 손바닥 가로 너비 / 손바닥 세로 길이 (보통 0.8 대 1 비율)
#     "finger_length_ratio": 0.77,  # 중지 전체 길이 / 손바닥 세로 길이
#     "index_to_ring_ratio": 0.98,  # 검지 전체 길이 / 약지 전체 길이 (거의 1대 1에 수렴)
#     "finger_profile": {
#         "index": 0.67,  # 검지 길이 / 손바닥 세로 길이
#         "middle": 0.77,  # 중지 길이 / 손바닥 세로 길이
#         "ring": 0.68,  # 약지 길이 / 손바닥 세로 길이
#         "little": 0.53  # 새끼 길이 / 손바닥 세로 길이
#     }
# }

# "palm_aspect_ratio": 0.48,
# "finger_length_ratio": 0.77,
# "index_to_ring_ratio": 0.49,
# "finger_profile": {
#     "index": 0.67,
#     "middle": 0.77,
#     "ring": 0.68,
#     "little": 0.53