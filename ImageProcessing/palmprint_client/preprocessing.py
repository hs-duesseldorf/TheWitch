from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np


@dataclass(frozen=True)
class RoiToneSettings:
    brightness: float = -8.0
    contrast: float = 1.2
    gamma: float = 1.1
    clahe_clip_limit: float = 1.8
    clahe_tile_size: int = 8


@lru_cache(maxsize=8)
def _clahe(clip_limit: float, tile_size: int) -> cv2.CLAHE:
    return cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_size), int(tile_size)),
    )


def enhance_palm_roi(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat_small = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, kernel_small)
    blackhat_large = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, kernel_large)
    line_map = cv2.addWeighted(blackhat_small, 0.65, blackhat_large, 0.35, 0.0)
    line_map = cv2.normalize(line_map, None, 0, 255, cv2.NORM_MINMAX)

    sharpened = cv2.addWeighted(clahe, 1.25, cv2.GaussianBlur(clahe, (0, 0), 2.2), -0.25, 0.0)
    enhanced = cv2.addWeighted(sharpened, 0.82, line_map, 0.38, 0.0)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def prepare_cnn_input_roi(roi_bgr: np.ndarray, settings: RoiToneSettings) -> np.ndarray:
    if roi_bgr.ndim == 2:
        gray = roi_bgr.astype(np.uint8, copy=False)
    else:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    processed = gray
    if settings.clahe_clip_limit > 0:
        tile = max(1, int(settings.clahe_tile_size))
        processed = _clahe(float(settings.clahe_clip_limit), tile).apply(processed)

    adjusted = processed.astype(np.float32) * float(settings.contrast) + float(settings.brightness)
    adjusted = np.clip(adjusted, 0.0, 255.0) / 255.0
    adjusted = np.power(adjusted, float(settings.gamma))
    return np.clip(adjusted * 255.0, 0.0, 255.0).astype(np.uint8)
