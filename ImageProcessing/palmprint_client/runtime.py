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

from .models import TextureCNN, create_hand_landmarker
from .preprocessing import RoiToneSettings, prepare_cnn_input_roi
from .transport import WebSocketClient
from .utils import l2_normalize, parse_camera_source
from shared.events import Hand, HandEvent, HandTrigger
from .vision import (
    are_hand_landmarks_fully_visible,
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
VIDEO_STREAM_INTERVAL_S = 1.0 / 30.0
WEBCAM_BRIDGE_URL = "http://host.docker.internal:8090/video"
CAMERA_SOURCE = os.getenv("WITCH_CAMERA_SOURCE", "0")
CAMERA_FPS = 0.0
CAMERA_WIDTH = 0
CAMERA_HEIGHT = 0
FRAME_INTERVAL_MS = 0
HISTORY_SIZE = 45
ROI_SIZE = 256
EMBED_EVERY = 1
EMBEDDING_MODEL = "arcface"
ROI_BRIGHTNESS = -8.0
ROI_CONTRAST = 1.2
ROI_GAMMA = 1.1
ROI_CLAHE_CLIP_LIMIT = 1.8
ROI_CLAHE_TILE_SIZE = 8
HAND_LANDMARK_VISIBILITY_MARGIN_PX = 2.0


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
        self.last_hand_observation_signature: tuple[Any, ...] | None = None
        self.last_video_frame_publish_at = 0.0
        self.last_roi_frame_publish_at = 0.0
        self.model_lock = threading.RLock()

        self.cap = cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened() and isinstance(self.camera_source, int):
            logger.warning("Camera %s failed, trying MJPEG bridge at %s", self.camera_source, WEBCAM_BRIDGE_URL)
            self.camera_source = WEBCAM_BRIDGE_URL
            self.cap = cv2.VideoCapture(WEBCAM_BRIDGE_URL)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera source {CAMERA_SOURCE!r}. "
                f"Start the webcam bridge if running in Docker Desktop without device passthrough: {WEBCAM_BRIDGE_URL}"
            )
        self._configure_capture_resolution()
        with suppress(cv2.error):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.landmarker = create_hand_landmarker(DEFAULT_HAND_MODEL_PATH)
        self.device = resolve_torch_device()
        self.roi_tone_settings = RoiToneSettings(
            brightness=ROI_BRIGHTNESS,
            contrast=ROI_CONTRAST,
            gamma=ROI_GAMMA,
            clahe_clip_limit=ROI_CLAHE_CLIP_LIMIT,
            clahe_tile_size=ROI_CLAHE_TILE_SIZE,
        )
        self.cnn = TextureCNN(
            device=self.device,
            embedding_model=EMBEDDING_MODEL,
            roi_tone_settings=self.roi_tone_settings,
        )

        self.geom_history: deque[dict[str, float]] = deque(maxlen=HISTORY_SIZE)
        self.embedding_history: deque[np.ndarray] = deque(maxlen=HISTORY_SIZE)
        self.last_embedding: np.ndarray | None = None
        self.feature_cache = self._empty_feature_cache()
        self._publish_hand_event(HandEvent(
            trigger=HandTrigger.ABSENT,
        ))

    def _configure_capture_resolution(self) -> None:
        if self.camera_fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        if self.width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def _empty_feature_cache(self) -> dict[str, Any]:
        return {
            "hand_lengths": {},
            "embedding_vector": [],
        }

    def start(self) -> None:
        if self.worker is not None:
            return
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._run_loop, name="palmprint-client-runtime", daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.worker is not None and self.worker.is_alive() and self.worker is not threading.current_thread():
            self.worker.join(timeout=3.0)
        self.worker = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def _should_publish_video_frame(self) -> bool:
        return (time.monotonic() - self.last_video_frame_publish_at) >= VIDEO_STREAM_INTERVAL_S

    def _should_publish_roi_frame(self) -> bool:
        return (time.monotonic() - self.last_roi_frame_publish_at) >= VIDEO_STREAM_INTERVAL_S

    def _publish_video_frame(
        self,
        camera_frame_bgr: np.ndarray | None,
    ) -> None:
        if camera_frame_bgr is None:
            return
        self.video_client.send_message(encode_frame_jpeg(camera_frame_bgr, max_width=720, quality=72))
        self.last_video_frame_publish_at = time.monotonic()

    def _publish_roi_frame(self, roi: np.ndarray) -> None:
        model_input = prepare_cnn_input_roi(roi, self.roi_tone_settings)
        self.roi_client.send_message(encode_frame_jpeg(model_input, max_width=256, quality=72))
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
        self._publish_hand_event(HandEvent(
            trigger=trigger,
            hand=hand,
        ))

    def _publish_feature_message(
        self,
        hand: Hand | None,
        camera_frame_bgr: np.ndarray | None = None,
    ) -> None:
        if self._should_publish_video_frame():
            self._publish_video_frame(camera_frame_bgr)

        self._publish_hand_event(HandEvent(
            trigger=HandTrigger.DETECTED,
            hand=hand,
            lengths=dict(self.feature_cache["hand_lengths"]),
            vector=list(self.feature_cache["embedding_vector"]),
        ))

    def _publish_hand_event(self, event: HandEvent) -> None:
        signature = (
            event.trigger,
            event.hand,
            bool(event.vector),
        )
        if signature == self.last_hand_observation_signature:
            return
        self.last_hand_observation_signature = signature
        self.event_client.send_message(event)

    def _update_feature_cache(self, geometry: dict[str, float]) -> None:
        geom_keys = sorted(geometry.keys())
        geom_matrix = np.array([[sample[key] for key in geom_keys] for sample in self.geom_history], dtype=np.float32)
        geom_median = np.median(geom_matrix, axis=0)

        emb_matrix = np.stack(list(self.embedding_history), axis=0)
        emb_median = l2_normalize(np.median(emb_matrix, axis=0).astype(np.float32))
        self.feature_cache = {
            "hand_lengths": {
                key: round(float(value), 6)
                for key, value in zip(geom_keys, geom_median.tolist())
            },
            "embedding_vector": [round(float(value), 6) for value in emb_median.tolist()],
        }

    def _publish_no_hand(self, frame_bgr: np.ndarray) -> None:
        self._publish_hand_status(
            trigger=HandTrigger.ABSENT,
            camera_frame_bgr=frame_bgr,
        )

    def _publish_roi_failure(self, hand: Hand | None, frame_bgr: np.ndarray) -> None:
        self._publish_hand_status(
            trigger=HandTrigger.TILTED,
            hand=hand,
            camera_frame_bgr=frame_bgr,
        )

    def _publish_palm_side_required(self, hand: Hand | None, frame_bgr: np.ndarray) -> None:
        self._publish_hand_status(
            trigger=HandTrigger.WRONG_SIDE,
            hand=hand,
            camera_frame_bgr=frame_bgr,
        )

    def _publish_pose_quality(self, hand: Hand | None, frame_bgr: np.ndarray) -> None:
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
            try:
                ok, raw_frame = self.cap.read()
                if not ok:
                    self._publish_error("Webcam frame read failed.")
                    if self.stop_event.wait(target_period):
                        break
                    continue

                display_frame = cv2.flip(raw_frame, 1)
                self.frame_counter += 1

                rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((time.monotonic() - self.started_at) * 1000.0)
                result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
                index = select_primary_hand_from_tasks(result)

                if index is None:
                    self._publish_no_hand(display_frame)
                else:
                    hand = result.hand_landmarks[index]
                    mediapipe_hand = None
                    if result.handedness and index < len(result.handedness) and result.handedness[index]:
                        with suppress(ValueError):
                            mediapipe_hand = Hand((result.handedness[index][0].category_name or "").strip().lower())
                    display_hand = None
                    if mediapipe_hand is Hand.LEFT:
                        display_hand = Hand.RIGHT
                    elif mediapipe_hand is Hand.RIGHT:
                        display_hand = Hand.LEFT

                    height, width = display_frame.shape[:2]
                    display_pts = np.array([[lm.x * width, lm.y * height, lm.z] for lm in hand], dtype=np.float32)
                    raw_pts = display_pts.copy()
                    raw_pts[:, 0] = (width - 1) - raw_pts[:, 0]

                    overlay_hand_frame = draw_hand_overlay(display_frame, display_pts[:, :2])
                    if not are_hand_landmarks_fully_visible(
                        display_pts[:, :2],
                        width,
                        height,
                        margin_px=HAND_LANDMARK_VISIBILITY_MARGIN_PX,
                    ):
                        self._publish_pose_quality(display_hand, overlay_hand_frame)
                        continue

                    world_pts = None
                    if result.hand_world_landmarks and index < len(result.hand_world_landmarks):
                        world = result.hand_world_landmarks[index]
                        world_pts = np.array([[lm.x, lm.y, lm.z] for lm in world], dtype=np.float32)

                    palm_side_points = world_pts
                    if world_pts is None:
                        palm_side_points = np.array([[lm.x * width, lm.y * height, lm.z] for lm in hand], dtype=np.float32)
                    if not is_palm_side_visible(np.asarray(palm_side_points), mediapipe_hand, min_confidence=0.3):
                        self._publish_palm_side_required(display_hand, overlay_hand_frame)
                        continue

                    points3d = world_pts if world_pts is not None else raw_pts
                    if not is_palm_frontal(np.asarray(palm_side_points), mediapipe_hand):
                        self._publish_pose_quality(display_hand, overlay_hand_frame)
                        continue

                    geometry = extract_geometry_features(points3d)
                    current_roi_pose = estimate_roi_pose(raw_pts[:, :2])
                    if current_roi_pose is None:
                        self._publish_roi_failure(display_hand, overlay_hand_frame)
                        continue

                    raw_roi_quad = roi_quad_from_pose(current_roi_pose)
                    display_roi_quad = mirror_quad_horizontally(raw_roi_quad, width)
                    overlay_frame = draw_roi_quad(overlay_hand_frame, display_roi_quad)
                    roi = warp_roi_from_quad(raw_frame, raw_roi_quad, self.roi_size)

                    if roi is None:
                        self._publish_roi_failure(display_hand, overlay_frame)
                    else:
                        if self._should_publish_roi_frame():
                            self._publish_roi_frame(roi)
                        with self.model_lock:
                            run_embed = (self.frame_counter % self.embed_every == 0) or (self.last_embedding is None)
                            if run_embed:
                                embedding = self.cnn.embed(roi)
                                self.last_embedding = embedding
                                self.embedding_history.append(embedding)
                            else:
                                embedding = self.last_embedding
                                if embedding is None:
                                    self._publish_error("Embedding cache was empty.")
                                    continue

                            self.geom_history.append(geometry)
                            if not self.embedding_history:
                                self.embedding_history.append(embedding)

                            self._update_feature_cache(geometry)
                        self._publish_feature_message(
                            display_hand,
                            overlay_frame,
                        )
            except Exception as exc:
                logger.exception("Palmprint processing loop failed")
                self._publish_error(str(exc))

            elapsed = time.monotonic() - loop_started
            remaining = target_period - elapsed
            if remaining > 0 and self.stop_event.wait(remaining):
                break
