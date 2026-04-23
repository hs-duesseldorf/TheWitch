from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .constants import (
    HAND_CONNECTIONS,
    INDEX_DIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_DIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_DIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_MCP,
    RING_DIP,
    RING_PIP,
    RING_TIP,
    THUMB_CMC,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
)
from .preprocessing import RoiToneSettings, prepare_cnn_input_roi


@dataclass(frozen=True)
class RoiPose:
    center: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    palm_width: float
    palm_height: float


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


def extract_geometry_features(points3d: np.ndarray) -> dict[str, float]:
    palm_width = _distance(points3d[INDEX_MCP], points3d[PINKY_MCP])
    palm_height = _distance(points3d[WRIST], points3d[MIDDLE_MCP])
    palm_scale = max(float(np.hypot(palm_width, palm_height)), 1e-9)

    thumb_len_raw = _finger_length(points3d, (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP))
    index_len_raw = _finger_length(points3d, (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP))
    middle_len_raw = _finger_length(points3d, (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP))
    ring_len_raw = _finger_length(points3d, (RING_MCP, RING_PIP, RING_DIP, RING_TIP))
    pinky_len_raw = _finger_length(points3d, (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP))

    palm_width_norm = palm_width / palm_scale
    palm_height_norm = palm_height / palm_scale
    thumb_len = thumb_len_raw / palm_scale
    index_len = index_len_raw / palm_scale
    middle_len = middle_len_raw / palm_scale
    ring_len = ring_len_raw / palm_scale
    pinky_len = pinky_len_raw / palm_scale

    return {
        "palm_width": palm_width_norm,
        "palm_height": palm_height_norm,
        "thumb_length": thumb_len,
        "index_length": index_len,
        "middle_length": middle_len,
        "ring_length": ring_len,
        "pinky_length": pinky_len,
        "thumb_to_palm_width": thumb_len_raw / max(palm_width, 1e-9),
        "thumb_to_palm_height": thumb_len_raw / max(palm_height, 1e-9),
        "index_to_palm_width": index_len_raw / max(palm_width, 1e-9),
        "index_to_palm_height": index_len_raw / max(palm_height, 1e-9),
        "middle_to_palm_width": middle_len_raw / max(palm_width, 1e-9),
        "middle_to_palm_height": middle_len_raw / max(palm_height, 1e-9),
        "ring_to_palm_width": ring_len_raw / max(palm_width, 1e-9),
        "ring_to_palm_height": ring_len_raw / max(palm_height, 1e-9),
        "pinky_to_palm_width": pinky_len_raw / max(palm_width, 1e-9),
        "pinky_to_palm_height": pinky_len_raw / max(palm_height, 1e-9),
    }


def estimate_roi_pose(landmarks_px: np.ndarray) -> RoiPose | None:
    points_xy = np.asarray(landmarks_px[:, :2], dtype=np.float32)

    wrist = points_xy[WRIST]
    index_mcp = points_xy[INDEX_MCP]
    middle_mcp = points_xy[MIDDLE_MCP]
    ring_mcp = points_xy[RING_MCP]
    pinky_mcp = points_xy[PINKY_MCP]

    index_side = (index_mcp + middle_mcp) / 2.0
    pinky_side = (ring_mcp + pinky_mcp) / 2.0
    mcp_band_center = (index_mcp + middle_mcp + ring_mcp + pinky_mcp) / 4.0

    y_seed = mcp_band_center - wrist
    y_axis = _normalize_vector(y_seed)
    if y_axis is None:
        return None

    x_seed = index_side - pinky_side
    x_seed = x_seed - float(np.dot(x_seed, y_axis)) * y_axis
    x_axis = _normalize_vector(x_seed)
    if x_axis is None:
        x_axis = _normalize_vector(_perpendicular(y_axis))
        if x_axis is None:
            return None
    if x_axis[0] < 0.0:
        x_axis = -x_axis

    y_axis = _perpendicular(x_axis)
    if float(np.dot(y_axis, y_seed)) < 0.0:
        y_axis = -y_axis

    mcp_span = float(np.linalg.norm(index_mcp - pinky_mcp))
    inner_span = float(np.linalg.norm(index_side - pinky_side))
    palm_width = 0.55 * mcp_span + 0.45 * inner_span

    wrist_to_band = float(np.linalg.norm(mcp_band_center - wrist))
    wrist_to_middle = float(np.linalg.norm(middle_mcp - wrist))
    palm_height = 0.65 * wrist_to_band + 0.35 * wrist_to_middle
    if palm_width < 3.0 or palm_height < 3.0:
        return None

    center = wrist + y_axis * (0.46 * palm_height)
    return RoiPose(
        center=center.astype(np.float32),
        x_axis=x_axis.astype(np.float32),
        y_axis=y_axis.astype(np.float32),
        palm_width=float(palm_width),
        palm_height=float(palm_height),
    )


def roi_quad_from_pose(pose: RoiPose) -> np.ndarray:
    top_center = pose.center + pose.y_axis * (0.34 * pose.palm_height)
    bottom_center = pose.center - pose.y_axis * (0.40 * pose.palm_height)
    half_w_top = 0.56 * pose.palm_width
    half_w_bottom = 0.66 * pose.palm_width

    tl = top_center - pose.x_axis * half_w_top
    tr = top_center + pose.x_axis * half_w_top
    br = bottom_center + pose.x_axis * half_w_bottom
    bl = bottom_center - pose.x_axis * half_w_bottom
    return np.float32([tl, tr, br, bl])


def warp_roi_from_quad(frame_bgr: np.ndarray, quad: np.ndarray, roi_size: int) -> np.ndarray | None:
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        return None
    if abs(float(cv2.contourArea(quad.astype(np.float32)))) < 16.0:
        return None

    dst = np.float32(
        [
            [0, 0],
            [roi_size - 1, 0],
            [roi_size - 1, roi_size - 1],
            [0, roi_size - 1],
        ]
    )

    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    roi = cv2.warpPerspective(
        frame_bgr,
        matrix,
        (roi_size, roi_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return roi


def mirror_quad_horizontally(quad: np.ndarray | None, frame_width: int) -> np.ndarray | None:
    if quad is None:
        return None
    mirrored = np.array(quad, dtype=np.float32, copy=True)
    mirrored[:, 0] = (frame_width - 1) - mirrored[:, 0]
    return mirrored


def draw_hand_overlay(frame_bgr: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    out = frame_bgr.copy()
    for a, b in HAND_CONNECTIONS:
        pa = tuple(points_xy[a].astype(int).tolist())
        pb = tuple(points_xy[b].astype(int).tolist())
        cv2.line(out, pa, pb, (80, 220, 80), 2, cv2.LINE_AA)
    for point in points_xy:
        pt = tuple(point.astype(int).tolist())
        cv2.circle(out, pt, 3, (0, 180, 255), -1, cv2.LINE_AA)
    return out


def draw_roi_quad(frame_bgr: np.ndarray, quad: np.ndarray | None) -> np.ndarray:
    out = frame_bgr.copy()
    if quad is None:
        return out
    pts = quad.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [pts], isClosed=True, color=(255, 180, 0), thickness=2, lineType=cv2.LINE_AA)
    return out


def encode_frame_data_url(frame_bgr: np.ndarray, *, max_width: int = 720, quality: int = 72) -> str:
    frame = frame_bgr
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / float(width)
        frame = cv2.resize(frame, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)

    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode debug frame.")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def render_placeholder_frame(width: int, height: int, title: str, subtitle: str) -> np.ndarray:
    frame = np.full((height, width, 3), (230, 236, 232), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (190, 204, 199), 3)
    if title:
        cv2.putText(frame, title, (20, max(40, height // 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (39, 66, 72), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(frame, subtitle, (20, max(72, height // 3 + 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (86, 106, 112), 1, cv2.LINE_AA)
    return frame


def render_roi_feature_preview(roi_bgr: np.ndarray, roi_tone_settings: RoiToneSettings) -> np.ndarray:
    gray = prepare_cnn_input_roi(roi_bgr, roi_tone_settings)
    height, width = gray.shape[:2]
    crop_size = max(1, int(round(min(height, width) * (224.0 / 256.0))))
    offset_y = max((height - crop_size) // 2, 0)
    offset_x = max((width - crop_size) // 2, 0)
    cropped = gray[offset_y : offset_y + crop_size, offset_x : offset_x + crop_size]
    preview = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)


def select_primary_hand_from_tasks(result: Any) -> int | None:
    if not result.hand_landmarks:
        return None
    best_idx = None
    best_area = -1.0
    for index, hand in enumerate(result.hand_landmarks):
        pts = np.array([[lm.x, lm.y] for lm in hand], dtype=np.float32)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        area = float(max(0.0, x_max - x_min) * max(0.0, y_max - y_min))
        if area > best_area:
            best_area = area
            best_idx = index
    return best_idx
