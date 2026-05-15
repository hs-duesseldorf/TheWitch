from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

ROI_BRIGHTNESS: float = -8.0
ROI_CONTRAST: float = 2.0
ROI_GAMMA: float = 1.1
ROI_CLAHE_CLIP_LIMIT: float = 0.0
ROI_CLAHE_TILE_SIZE: int = 8


@lru_cache(maxsize=8)
def _clahe(clip_limit: float, tile_size: int) -> cv2.CLAHE:
    return cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_size), int(tile_size)),
    )


@lru_cache(maxsize=16)
def _tone_lut(contrast: float, brightness: float, gamma: float) -> np.ndarray:
    values = np.arange(256, dtype=np.float32)
    values = np.clip(values * float(contrast) + float(brightness), 0.0, 255.0) / 255.0
    values = np.power(values, float(gamma))
    return np.clip(values * 255.0, 0.0, 255.0).astype(np.uint8)


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


def prepare_cnn_input_roi(roi_bgr: np.ndarray) -> np.ndarray:
    if roi_bgr.ndim == 2:
        gray = roi_bgr.astype(np.uint8, copy=False)
    else:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    processed = gray
    if ROI_CLAHE_CLIP_LIMIT > 0:
        tile = max(1, int(ROI_CLAHE_TILE_SIZE))
        processed = _clahe(float(ROI_CLAHE_CLIP_LIMIT), tile).apply(processed)

    lut = _tone_lut(float(ROI_CONTRAST), float(ROI_BRIGHTNESS), float(ROI_GAMMA))
    return cv2.LUT(processed, lut)
