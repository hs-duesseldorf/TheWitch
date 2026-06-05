from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import torch

from shared.events import Hand, HandEvent, HandTrigger

from .models import TextureCNN, create_hand_landmarker
from .preprocessing import prepare_cnn_input_roi
from .transport import WebSocketClient
from .utils import l2_normalize, parse_camera_source
from .vision import (
    are_normalized_hand_landmarks_fully_visible,
    draw_hand_overlay,
    draw_roi_quad,
    encode_frame_jpeg,
    estimate_roi_pose,
    extract_geometry_features,
    is_palm_frontal,
    is_palm_side_visible,
    mirror_quad_horizontally,
    roi_quad_from_pose,
    select_primary_hand_from_tasks,
    warp_roi_from_quad,
)

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
DEFAULT_HAND_MODEL_PATH = ASSETS_DIR / "models" / "hand_landmarker.task"
CAMERA_FPS = 15.0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
VIDEO_STREAM_FPS = 15.0
VIDEO_STREAM_WIDTH = 640
VIDEO_STREAM_INTERVAL_S = 1.0 / VIDEO_STREAM_FPS
CAMERA_SOURCE = os.environ["WITCH_CAMERA_SOURCE"]
FRAME_INTERVAL_MS = int(round(1000.0 / CAMERA_FPS))
HISTORY_SIZE = 45
ROI_SIZE = 256
EMBED_EVERY = 3
EMBEDDING_MODEL = "arcface"

HAND_LANDMARK_VISIBILITY_MARGIN_PX = 2.0
NOT_FULLY_IN_VIEW_DEBOUNCE_S = 2.5
MIN_HAND_EVENT_INTERVAL_S = 0.15
GEOMETRY_FEATURE_KEYS = (
    "palm_width",
    "palm_height",
    "thumb_length",
    "index_length",
    "middle_length",
    "ring_length",
    "pinky_length",
)


def resolve_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class HeadlessPalmClient:
    def __init__(
        self,
        *,
        event_client: WebSocketClient,
        video_client: WebSocketClient,
        roi_client: WebSocketClient,
    ):
        self.event_client = event_client
        self.video_client = video_client
        self.roi_client = roi_client

        self.camera_fps = CAMERA_FPS
        self.width = CAMERA_WIDTH
        self.height = CAMERA_HEIGHT
        self.interval_ms = FRAME_INTERVAL_MS
        self.roi_size = ROI_SIZE
        self.embed_every = EMBED_EVERY

        self.camera_source = parse_camera_source(CAMERA_SOURCE)
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.started_at = time.monotonic()
        self.frame_counter = 0
        self._not_fully_in_view_since = 0.0
        self.last_hand_observation_signature: tuple[Any, ...] | None = None
        self.last_video_frame_publish_at = 0.0
        self.last_roi_frame_publish_at = 0.0
        self._last_hand_event_at = 0.0

        self.cap = cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened():
            logger.warning(
                "Could not open camera source %r. Continuing without camera. "
                "Video processing will be disabled.",
                CAMERA_SOURCE,
            )
            self.cap = None
        else:
            self._configure_capture_resolution()
            with suppress(cv2.error):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.landmarker = create_hand_landmarker(DEFAULT_HAND_MODEL_PATH)
        self.device = resolve_torch_device()
        logger.info("Palm embedding torch device: %s", self.device)
        self.cnn = TextureCNN(
            device=self.device,
            embedding_model=EMBEDDING_MODEL,
        )

        self.geom_history: deque[dict[str, float]] = deque(maxlen=HISTORY_SIZE)
        self.embedding_history: deque[np.ndarray] = deque(maxlen=HISTORY_SIZE)
        self.feature_cache = self._empty_feature_cache()
        self._publish_hand_event(
            HandEvent(
                trigger=HandTrigger.ABSENT,
            )
        )

    def _configure_capture_resolution(self) -> None:
        self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def _normalize_capture_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        if width == self.width and height == self.height:
            return frame_bgr
        return cv2.resize(
            frame_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA
        )

    def _empty_feature_cache(self) -> dict[str, Any]:
        return {
            "hand_lengths": {},
            "embedding_vector": [],
        }

    def start(self) -> None:
        if self.worker is not None:
            return
        self.stop_event.clear()
        self.worker = threading.Thread(
            target=self._run_loop, name="palmprint-client-runtime", daemon=True
        )
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        if (
            self.worker is not None
            and self.worker.is_alive()
            and self.worker is not threading.current_thread()
        ):
            self.worker.join(timeout=3.0)
        self.worker = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def _should_publish_video_frame(self) -> bool:
        return (
            time.monotonic() - self.last_video_frame_publish_at
        ) >= VIDEO_STREAM_INTERVAL_S

    def _should_publish_roi_frame(self) -> bool:
        return (
            time.monotonic() - self.last_roi_frame_publish_at
        ) >= VIDEO_STREAM_INTERVAL_S

    def _publish_video_frame(self, camera_frame_bgr: np.ndarray) -> None:
        self.video_client.send_message(
            encode_frame_jpeg(
                camera_frame_bgr, max_width=VIDEO_STREAM_WIDTH, quality=72
            )
        )
        self.last_video_frame_publish_at = time.monotonic()

    def _publish_roi_frame(self, roi_frame_bgr: np.ndarray) -> None:
        self.roi_client.send_message(
            encode_frame_jpeg(roi_frame_bgr, max_width=ROI_SIZE, quality=72)
        )
        self.last_roi_frame_publish_at = time.monotonic()

    def _publish_hand_status(
        self,
        *,
        trigger: HandTrigger,
        hand: Hand | None = None,
        camera_frame_bgr: np.ndarray | None = None,
    ) -> None:
        if camera_frame_bgr is not None and self._should_publish_video_frame():
            self._publish_video_frame(camera_frame_bgr)
        if not self._should_publish_hand_signature(trigger, hand, False):
            return
        self._publish_hand_event(
            HandEvent(
                trigger=trigger,
                hand=hand,
            )
        )

    def _publish_feature_message(
        self,
        hand: Hand | None,
    ) -> None:
        vector = self.feature_cache["embedding_vector"]
        if not self._should_publish_hand_signature(
            HandTrigger.DETECTED, hand, bool(vector)
        ):
            return
        self._publish_hand_event(
            HandEvent(
                trigger=HandTrigger.DETECTED,
                hand=hand,
                lengths=dict(self.feature_cache["hand_lengths"]),
                vector=list(vector),
            )
        )

    def _should_publish_hand_signature(
        self,
        trigger: HandTrigger,
        hand: Hand | None,
        has_vector: bool,
    ) -> bool:
        return (trigger, hand, has_vector) != self.last_hand_observation_signature

    def _publish_hand_event(self, event: HandEvent) -> None:
        now = time.monotonic()
        if now - self._last_hand_event_at < MIN_HAND_EVENT_INTERVAL_S:
            return
        signature = (
            event.trigger,
            event.hand,
            bool(event.vector),
        )
        if signature == self.last_hand_observation_signature:
            return
        self.last_hand_observation_signature = signature
        self._last_hand_event_at = now
        self.event_client.send_message(event)

    def _update_feature_cache(self) -> None:
        geom_matrix = np.array(
            [
                [sample[key] for key in GEOMETRY_FEATURE_KEYS]
                for sample in self.geom_history
            ],
            dtype=np.float32,
        )
        geom_median = np.median(geom_matrix, axis=0)

        emb_matrix = np.stack(list(self.embedding_history), axis=0)
        emb_median = l2_normalize(np.median(emb_matrix, axis=0).astype(np.float32))
        self.feature_cache = {
            "hand_lengths": {
                key: round(float(value), 6)
                for key, value in zip(GEOMETRY_FEATURE_KEYS, geom_median.tolist())
            },
            "embedding_vector": [
                round(float(value), 6) for value in emb_median.tolist()
            ],
        }

    def _publish_roi_failure(
        self, hand: Hand | None, frame_bgr: np.ndarray | None
    ) -> None:
        self._publish_hand_status(
            trigger=HandTrigger.TILTED,
            hand=hand,
            camera_frame_bgr=frame_bgr,
        )

    def _publish_palm_side_required(
        self, hand: Hand | None, frame_bgr: np.ndarray | None
    ) -> None:
        self._publish_hand_status(
            trigger=HandTrigger.WRONG_SIDE,
            hand=hand,
            camera_frame_bgr=frame_bgr,
        )

    def _publish_hand_not_fully_in_view(
        self, hand: Hand | None, frame_bgr: np.ndarray | None
    ) -> None:
        self._publish_hand_status(
            trigger=HandTrigger.NOT_FULLY_IN_VIEW,
            hand=hand,
            camera_frame_bgr=frame_bgr,
        )

    def _publish_pose_quality(
        self, hand: Hand | None, frame_bgr: np.ndarray | None
    ) -> None:
        self._publish_hand_status(
            trigger=HandTrigger.TILTED,
            hand=hand,
            camera_frame_bgr=frame_bgr,
        )

    def _publish_error(self, message: str) -> None:
        logger.warning("Palmprint runtime error: %s", message)
        self._publish_hand_status(
            trigger=HandTrigger.ABSENT,
        )

    def _run_loop(self) -> None:
        target_period = max(self.interval_ms / 1000.0, 0.0)
        while not self.stop_event.is_set():
            loop_started = time.monotonic()
            if self.cap is None:
                self.stop_event.wait(target_period)
                continue
            try:
                ok, raw_frame = self.cap.read()
                if not ok:
                    self._publish_error("Webcam frame read failed.")
                    if self.stop_event.wait(target_period):
                        break
                    continue

                raw_frame = self._normalize_capture_frame(raw_frame)
                self.frame_counter += 1

                rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((time.monotonic() - self.started_at) * 1000.0)
                result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
                index = select_primary_hand_from_tasks(result)

                if index is None:
                    self._not_fully_in_view_since = 0.0
                    if self._should_publish_video_frame():
                        self._publish_video_frame(cv2.flip(raw_frame, 1))
                    self._publish_hand_status(
                        trigger=HandTrigger.ABSENT,
                    )
                else:
                    hand = result.hand_landmarks[index]
                    display_hand = Hand(
                        result.handedness[index][0].category_name.strip().lower()
                    )

                    height, width = raw_frame.shape[:2]
                    should_publish_video = self._should_publish_video_frame()
                    if not are_normalized_hand_landmarks_fully_visible(
                        hand,
                        width,
                        height,
                        margin_px=HAND_LANDMARK_VISIBILITY_MARGIN_PX,
                    ):
                        overlay_hand_frame = None
                        if should_publish_video:
                            display_frame = cv2.flip(raw_frame, 1)
                            display_pts = np.array(
                                [
                                    [(1.0 - lm.x) * width, lm.y * height, lm.z]
                                    for lm in hand
                                ],
                                dtype=np.float32,
                            )
                            overlay_hand_frame = draw_hand_overlay(
                                display_frame, display_pts[:, :2]
                            )
                        if self._not_fully_in_view_since == 0.0:
                            self._not_fully_in_view_since = time.monotonic()
                        elif (
                            time.monotonic() - self._not_fully_in_view_since
                            >= NOT_FULLY_IN_VIEW_DEBOUNCE_S
                        ):
                            self._publish_hand_not_fully_in_view(
                                display_hand, overlay_hand_frame
                            )
                        continue

                    self._not_fully_in_view_since = 0.0
                    raw_pts = np.array(
                        [[lm.x * width, lm.y * height, lm.z] for lm in hand],
                        dtype=np.float32,
                    )

                    overlay_hand_frame = None
                    if should_publish_video:
                        display_frame = cv2.flip(raw_frame, 1)
                        display_pts = raw_pts.copy()
                        display_pts[:, 0] = (width - 1) - display_pts[:, 0]
                        overlay_hand_frame = draw_hand_overlay(
                            display_frame, display_pts[:, :2]
                        )

                    world = result.hand_world_landmarks[index]
                    world_pts = np.array(
                        [[lm.x, lm.y, lm.z] for lm in world], dtype=np.float32
                    )

                    if not is_palm_side_visible(
                        world_pts, display_hand, min_confidence=0.3
                    ):
                        self._publish_palm_side_required(
                            display_hand, overlay_hand_frame
                        )
                        continue

                    if not is_palm_frontal(world_pts, display_hand):
                        self._publish_pose_quality(display_hand, overlay_hand_frame)
                        continue

                    geometry = extract_geometry_features(world_pts)
                    current_roi_pose = estimate_roi_pose(raw_pts[:, :2])
                    if current_roi_pose is None:
                        self._publish_roi_failure(display_hand, overlay_hand_frame)
                        continue

                    raw_roi_quad = roi_quad_from_pose(current_roi_pose)
                    overlay_frame = None
                    if overlay_hand_frame is not None:
                        display_roi_quad = mirror_quad_horizontally(raw_roi_quad, width)
                        overlay_frame = draw_roi_quad(
                            overlay_hand_frame, display_roi_quad
                        )
                    roi = warp_roi_from_quad(raw_frame, raw_roi_quad, self.roi_size)

                    should_publish_roi = self._should_publish_roi_frame()
                    roi_enhanced = prepare_cnn_input_roi(roi)
                    run_embed = (
                        self.frame_counter % self.embed_every == 0
                    ) or not self.embedding_history
                    if run_embed:
                        embedding = self.cnn.embed_preprocessed(roi_enhanced)
                        self.embedding_history.append(embedding)
                    if should_publish_roi:
                        self._publish_roi_frame(roi_enhanced)

                    self.geom_history.append(geometry)
                    self._update_feature_cache()
                    if overlay_frame is not None:
                        self._publish_video_frame(overlay_frame)
                    self._publish_feature_message(display_hand)
            except Exception as exc:
                logger.exception("Palmprint processing loop failed")
                self._publish_error(str(exc))

            elapsed = time.monotonic() - loop_started
            remaining = target_period - elapsed
            if remaining > 0 and self.stop_event.wait(remaining):
                break
