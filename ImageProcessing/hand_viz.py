from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from shared.events import HandTrigger

HAND_VIZ_INTERVAL_S = 1.0 / 15.0
MASK_SIZE = 128
MASK_JPEG_QUALITY = 80
MASK_BLUR_KERNEL = 9
MASK_MORPH_KERNEL = 3
BACKGROUND_HISTORY = 300
BACKGROUND_VAR_THRESHOLD = 32.0
BACKGROUND_LEARNING_RATE = 0.01
BACKGROUND_RELEARN_RATE = 0.2
FOREGROUND_FREEZE_RATIO = 0.02
UNEXPLAINED_FOREGROUND_MAX_S = 2.0


class HandVizPublisher:
    """Sends hand landmarks and a blurred silhouette mask (no real image) for the Unreal hand stage.

    Text message: {"trigger": str, "points": [[x, y], ...]} with x/y in 0..1 of the mirrored centre square.
    Binary message: JPEG (MASK_SIZE x MASK_SIZE), white = something in front of the background.
    """

    def __init__(self, client: Any):
        self.client = client
        self.last_publish_at = 0.0
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=BACKGROUND_HISTORY,
            varThreshold=BACKGROUND_VAR_THRESHOLD,
            detectShadows=False,
        )
        self.morph_kernel = np.ones((MASK_MORPH_KERNEL, MASK_MORPH_KERNEL), np.uint8)
        self.unexplained_foreground_since: float | None = None

    def publish(self, frame_bgr: np.ndarray, trigger: HandTrigger, landmarks: list[Any] | None) -> None:
        if (time.monotonic() - self.last_publish_at) < HAND_VIZ_INTERVAL_S:
            return
        self.last_publish_at = time.monotonic()

        square = self._mirrored_centre_square(frame_bgr)
        mask = self._silhouette_mask(square, hand_present=landmarks is not None)
        points = self._square_points(frame_bgr, landmarks) if landmarks is not None else []

        self.client.send_message({"trigger": trigger.value, "points": points})
        self.client.send_message(self._encode_mask(mask))

    @staticmethod
    def _mirrored_centre_square(frame_bgr: np.ndarray) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        square = frame_bgr[top:top + side, left:left + side]
        return cv2.flip(cv2.resize(square, (MASK_SIZE, MASK_SIZE), interpolation=cv2.INTER_AREA), 1)

    def _silhouette_mask(self, square_bgr: np.ndarray, *, hand_present: bool) -> np.ndarray:
        # Peek first: freeze the background while a hand (or anything larger) is in view,
        # otherwise a resting hand slowly becomes background.
        foreground = self.subtractor.apply(square_bgr, learningRate=0.0)
        foreground_ratio = float(np.count_nonzero(foreground)) / foreground.size
        learning_rate = self._learning_rate(foreground_ratio, hand_present)
        if learning_rate > 0.0:
            self.subtractor.apply(square_bgr, learningRate=learning_rate)

        cleaned = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, self.morph_kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.morph_kernel)
        return cv2.GaussianBlur(cleaned, (MASK_BLUR_KERNEL, MASK_BLUR_KERNEL), 0)

    def _learning_rate(self, foreground_ratio: float, hand_present: bool) -> float:
        if hand_present:
            self.unexplained_foreground_since = None
            return 0.0
        if foreground_ratio < FOREGROUND_FREEZE_RATIO:
            self.unexplained_foreground_since = None
            return BACKGROUND_LEARNING_RATE
        # Foreground without a hand: wait briefly (hand may be entering), then relearn
        # (camera exposure change, lighting, startup).
        now = time.monotonic()
        if self.unexplained_foreground_since is None:
            self.unexplained_foreground_since = now
        if now - self.unexplained_foreground_since > UNEXPLAINED_FOREGROUND_MAX_S:
            return BACKGROUND_RELEARN_RATE
        return 0.0

    @staticmethod
    def _square_points(frame_bgr: np.ndarray, landmarks: list[Any]) -> list[list[float]]:
        height, width = frame_bgr.shape[:2]
        side = min(width, height)
        left, top = (width - side) / 2.0, (height - side) / 2.0
        points = []
        for landmark in landmarks:
            x = (landmark.x * width - left) / side
            y = (landmark.y * height - top) / side
            points.append([round(1.0 - x, 4), round(y, 4)])
        return points

    @staticmethod
    def _encode_mask(mask: np.ndarray) -> bytes:
        # 3 channels: Unreal's Import Buffer As Texture 2D handles RGB JPEGs reliably
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), MASK_JPEG_QUALITY])
        if not ok:
            raise RuntimeError(f"Hand viz mask JPEG encoding failed for mask shape {mask.shape}")
        return encoded.tobytes()
