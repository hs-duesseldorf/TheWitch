from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np

from shared.events import HandTrigger

logger = logging.getLogger("image-processing")

HAND_VIZ_INTERVAL_S = 1.0 / 15.0
MASK_SIZE = 256
MASK_JPEG_QUALITY = 80
MASK_BLUR_KERNEL = 9
MASK_MORPH_KERNEL = 3
# Larger close only: fills holes in the mask (each hole would draw its own fog outline); open stays small so fingers survive
MASK_CLOSE_KERNEL = 7
BACKGROUND_HISTORY = 300
BACKGROUND_VAR_THRESHOLD = 32.0
# No skin colour: fixed colour ranges failed for bright light and do not cover every skin tone.
# With landmarks the fog is the foreground inside a hand-shaped stencil (palm + one thick line per bone).
# A convex hull let finger gaps through whenever the background counts as foreground (exposure change,
# hand close to the camera), so the fingers merged into one block.
HAND_STENCIL_WIDTH_RATIO = 0.3  # bone line thickness relative to the wrist-to-middle-knuckle length (real finger ~0.2)
WRIST_LANDMARK = 0
MIDDLE_KNUCKLE_LANDMARK = 9
PALM_LANDMARKS = (0, 1, 5, 9, 13, 17)
HAND_BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),  # palm edges, so the palm border gets the same margin
)
# Keeps the background frozen briefly after landmarks drop out (hand partly out of view while aligning).
HAND_RECENT_S = 3.0
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
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_CLOSE_KERNEL, MASK_CLOSE_KERNEL))
        self.unexplained_foreground_since: float | None = None
        self.last_hand_seen_at: float | None = None

    def publish(self, frame_bgr: np.ndarray, trigger: HandTrigger, landmarks: list[Any] | None) -> None:
        if (time.monotonic() - self.last_publish_at) < HAND_VIZ_INTERVAL_S:
            return
        self.last_publish_at = time.monotonic()

        square = self._mirrored_centre_square(frame_bgr)
        points = self._square_points(frame_bgr, landmarks) if landmarks is not None else []
        mask = self._silhouette_mask(square, points)

        self.client.send_message({"trigger": trigger.value, "points": points})
        self.client.send_message(self._encode_mask(mask))

    @staticmethod
    def _mirrored_centre_square(frame_bgr: np.ndarray) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        square = frame_bgr[top:top + side, left:left + side]
        return cv2.flip(cv2.resize(square, (MASK_SIZE, MASK_SIZE), interpolation=cv2.INTER_AREA), 1)

    def _silhouette_mask(self, square_bgr: np.ndarray, points: list[list[float]]) -> np.ndarray:
        # Peek first: freeze the background while a hand (or anything larger) is in view,
        # otherwise a resting hand slowly becomes background.
        foreground = self.subtractor.apply(square_bgr, learningRate=0.0)
        foreground_ratio = float(np.count_nonzero(foreground)) / foreground.size
        if points:
            self.last_hand_seen_at = time.monotonic()
        # A partly visible hand (landmarks just lost) must not fade into the background while aligning.
        hand_recent = self.last_hand_seen_at is not None and time.monotonic() - self.last_hand_seen_at < HAND_RECENT_S
        learning_rate = self._learning_rate(foreground_ratio, hand_recent)
        if learning_rate > 0.0:
            self.subtractor.apply(square_bgr, learningRate=learning_rate)
        # Without landmarks (hand still entering) all foreground shows, as an alignment aid.
        hand_like = cv2.bitwise_and(foreground, self._hand_region(points)) if points else foreground

        cleaned = cv2.morphologyEx(hand_like, cv2.MORPH_OPEN, self.morph_kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.close_kernel)
        return cv2.GaussianBlur(cleaned, (MASK_BLUR_KERNEL, MASK_BLUR_KERNEL), 0)

    @staticmethod
    def _hand_region(points: list[list[float]]) -> np.ndarray:
        pixels = (np.asarray(points, dtype=np.float32) * MASK_SIZE).astype(np.int32)
        region = np.zeros((MASK_SIZE, MASK_SIZE), np.uint8)
        cv2.fillConvexPoly(region, cv2.convexHull(pixels[list(PALM_LANDMARKS)]), 255)
        # Thick lines have round caps, so the stencil also covers the fingertip beyond the tip landmark.
        width = max(2, int(HAND_STENCIL_WIDTH_RATIO * np.linalg.norm(pixels[MIDDLE_KNUCKLE_LANDMARK] - pixels[WRIST_LANDMARK])))
        for start, end in HAND_BONES:
            cv2.line(region, tuple(int(v) for v in pixels[start]), tuple(int(v) for v in pixels[end]), 255, width)
        return region

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
