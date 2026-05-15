from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from shared.events import Hand

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


def palm_facing_score(points3d: np.ndarray, hand: Hand | None) -> float | None:
    points = np.asarray(points3d, dtype=np.float32)
    if points.shape[0] <= max(WRIST, INDEX_MCP, PINKY_MCP):
        return None

    wrist = points[WRIST, :3]
    index_mcp = points[INDEX_MCP, :3]
    pinky_mcp = points[PINKY_MCP, :3]
    normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
    normal_unit = _normalize_vector(normal)
    if normal_unit is None:
        return None

    # MediaPipe depth grows away from the camera, so the viewing direction is -Z.
    camera_direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    facing_score = float(np.dot(normal_unit, camera_direction))

    if hand is Hand.RIGHT:
        return facing_score
    if hand is Hand.LEFT:
        return -facing_score
    return None


def is_palm_side_visible(points3d: np.ndarray, hand: Hand | None, *, min_confidence: float = 0.2) -> bool:
    score = palm_facing_score(points3d, hand)
    if score is None:
        return True
    return score >= min_confidence


def is_palm_frontal(points3d: np.ndarray, hand: Hand | None, *, min_confidence: float = 0.82) -> bool:
    score = palm_facing_score(points3d, hand)
    if score is None:
        return True
    return abs(score) >= min_confidence


def are_normalized_hand_landmarks_fully_visible(
    landmarks: list[Any],
    frame_width: int,
    frame_height: int,
    *,
    margin_px: float = 0.0,
) -> bool:
    min_x = margin_px / float(frame_width)
    min_y = margin_px / float(frame_height)
    max_x = (float(frame_width - 1) - margin_px) / float(frame_width)
    max_y = (float(frame_height - 1) - margin_px) / float(frame_height)

    lo_x = float("inf")
    lo_y = float("inf")
    hi_x = float("-inf")
    hi_y = float("-inf")
    for landmark in landmarks:
        x = landmark.x
        y = landmark.y
        if x < lo_x:
            lo_x = x
        if x > hi_x:
            hi_x = x
        if y < lo_y:
            lo_y = y
        if y > hi_y:
            hi_y = y

    return lo_x >= min_x and hi_x <= max_x and lo_y >= min_y and hi_y <= max_y


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
    if y_axis is None:
        return None

    x_seed = index_side - pinky_side
    x_seed = x_seed - float(np.dot(x_seed, y_axis)) * y_axis
    x_axis = _normalize_vector(x_seed)
    if x_axis is None:
        x_axis = _normalize_vector(_perpendicular(y_axis))
        if x_axis is None:
            return None
    thumb_projection = float(np.dot(thumb_mcp - wrist, x_axis))
    if thumb_projection < 0.0:
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

    return RoiPose(
        center=(wrist + y_axis * (0.46 * palm_height)).astype(np.float32),
        x_axis=x_axis.astype(np.float32),
        y_axis=y_axis.astype(np.float32),
        palm_width=float(palm_width),
        palm_height=float(palm_height),
    )


def roi_quad_from_pose(pose: RoiPose) -> np.ndarray:
    top_center = pose.center + pose.y_axis * (0.50 * pose.palm_height)
    bottom_center = pose.center - pose.y_axis * (0.31 * pose.palm_height) + pose.x_axis * (0.065 * pose.palm_width)
    half_w_top = 0.51 * pose.palm_width
    half_w_bottom = 0.58 * pose.palm_width
    thumb_bias = 0.07 * pose.palm_width

    left_top = half_w_top - 0.02 * pose.palm_width
    right_top = half_w_top + thumb_bias
    left_bottom = half_w_bottom - 0.03 * pose.palm_width
    right_bottom = half_w_bottom + thumb_bias

    tl = top_center - pose.x_axis * left_top
    tr = top_center + pose.x_axis * right_top
    br = bottom_center + pose.x_axis * right_bottom
    bl = bottom_center - pose.x_axis * left_bottom
    return np.float32([tl, tr, br, bl])


def warp_roi_from_quad(frame_bgr: np.ndarray, quad: np.ndarray, roi_size: int) -> np.ndarray:
    dst = np.float32(
        [
            [0, 0],
            [roi_size - 1, 0],
            [roi_size - 1, roi_size - 1],
            [0, roi_size - 1],
        ]
    )

    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(
        frame_bgr,
        matrix,
        (roi_size, roi_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def mirror_quad_horizontally(quad: np.ndarray, frame_width: int) -> np.ndarray:
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


def draw_roi_quad(frame_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    out = frame_bgr.copy()
    pts = quad.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [pts], isClosed=True, color=(255, 180, 0), thickness=2, lineType=cv2.LINE_AA)
    return out


def encode_frame_jpeg(frame_bgr: np.ndarray, *, max_width: int = 720, quality: int = 72) -> bytes:
    frame = frame_bgr
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / float(width)
        frame = cv2.resize(frame, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)

    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode video frame.")
    return encoded.tobytes()


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
