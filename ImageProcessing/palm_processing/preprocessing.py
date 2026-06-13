from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

ROI_CONTRAST: float = 1.4
ROI_GAMMA: float = 1.1
ROI_TARGET_MEAN: float = 128.0
ROI_BRIGHTNESS_CLAMP: float = 192.0
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


def _auto_exposure_brightness(gray: np.ndarray, contrast: float) -> float:
    mean = float(np.mean(gray))
    brightness = ROI_TARGET_MEAN - mean * contrast
    return max(-ROI_BRIGHTNESS_CLAMP, min(ROI_BRIGHTNESS_CLAMP, brightness))


def prepare_cnn_input_roi(roi_bgr: np.ndarray) -> np.ndarray:
    if roi_bgr.ndim == 2:
        gray = roi_bgr.astype(np.uint8, copy=False)
    else:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    processed = gray
    if ROI_CLAHE_CLIP_LIMIT > 0:
        tile = max(1, int(ROI_CLAHE_TILE_SIZE))
        processed = _clahe(float(ROI_CLAHE_CLIP_LIMIT), tile).apply(processed)

    brightness = _auto_exposure_brightness(processed, float(ROI_CONTRAST))
    lut = _tone_lut(float(ROI_CONTRAST), brightness, float(ROI_GAMMA))
    return cv2.LUT(processed, lut)


def enhance_palm_roi(roi_bgr: np.ndarray) -> np.ndarray:
    return prepare_cnn_input_roi(roi_bgr)
