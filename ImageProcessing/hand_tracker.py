from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable

import numpy as np

from shared.events import Hand, HandEvent, HandTrigger

logger = logging.getLogger("image-processing")

HAND_MODEL_PATH = Path(__file__).resolve().parent / "assets" / "models" / "hand_landmarker.task"

CAMERA_FPS = 15.0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FRAME_INTERVAL_S = 1.0 / CAMERA_FPS
VIDEO_STREAM_WIDTH = 640
VIDEO_STREAM_INTERVAL_S = 1.0 / 15.0
ROI_SIZE = 256
HISTORY_SIZE = 45
EMBED_EVERY = 3
EMBEDDING_MODEL = "arcface"
HAND_LANDMARK_VISIBILITY_MARGIN_PX = 2.0
GEOMETRY_FEATURE_KEYS = (
    "palm_width",
    "palm_height",
    "thumb_length",
    "index_length",
    "middle_length",
    "ring_length",
    "pinky_length",
)

cv2: Any = None
mp: Any = None
torch: Any = None
vision: Any = None
models: Any = None
preprocessing: Any = None
l2_normalize: Any = None


def _load_dependencies():
    global cv2, mp, torch, vision, models, preprocessing, l2_normalize
    import cv2 as _cv2
    import mediapipe as _mp
    import torch as _torch

    from ImageProcessing.palm_processing import models as _models
    from ImageProcessing.palm_processing import preprocessing as _preprocessing
    from ImageProcessing.palm_processing import vision as _vision
    from ImageProcessing.palm_processing.utils import l2_normalize as _l2_normalize

    cv2, mp, torch = _cv2, _mp, _torch
    vision, models, preprocessing = _vision, _models, _preprocessing
    l2_normalize = _l2_normalize


def _parse_camera_source(raw: str) -> int | str:
    stripped = raw.strip()
    return int(stripped) if stripped.lstrip("-").isdigit() else stripped


class HandTracker:
    def __init__(self, video_client, roi_client):
        self.video_client = video_client
        self.roi_client = roi_client
        self.camera_source = _parse_camera_source(os.environ["WITCH_CAMERA_SOURCE"])

        self._callback: Callable[[HandEvent], None] | None = None
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None

        self.cap: Any | None = None
        self.landmarker: Any | None = None
        self.cnn: Any | None = None

        self.started_at = time.monotonic()
        self.frame_counter = 0
        self.last_video_publish_at = 0.0
        self.last_roi_publish_at = 0.0
        self.geom_history: deque[dict[str, float]] = deque(maxlen=HISTORY_SIZE)
        self.embedding_history: deque[np.ndarray] = deque(maxlen=HISTORY_SIZE)
        self.feature_cache = self._empty_feature_cache()

    def start(self, callback: Callable[[HandEvent], None]):
        if self.worker is not None:
            return
        _load_dependencies()
        self._callback = callback
        self._open_camera()
        self.landmarker = models.create_hand_landmarker(HAND_MODEL_PATH)
        device = self._resolve_torch_device()
        logger.info("Palm embedding torch device: %s", device)
        self.cnn = models.TextureCNN(device=device, embedding_model=EMBEDDING_MODEL)
        self._emit(HandEvent(trigger=HandTrigger.ABSENT))

        self.stop_event.clear()
        self.worker = threading.Thread(target=self._run, name="hand-tracker", daemon=True)
        self.worker.start()

    def stop(self):
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

    @staticmethod
    def _resolve_torch_device():
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _open_camera(self):
        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            logger.warning("Could not open camera source %r. Continuing without camera.", self.camera_source)
            self.cap = None
            return
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        with suppress(cv2.error):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap

    def _run(self):
        while not self.stop_event.is_set():
            loop_started = time.monotonic()
            if self.cap is None:
                self.stop_event.wait(FRAME_INTERVAL_S)
                continue
            try:
                self._process_next_frame()
            except Exception as exc:
                logger.exception("Hand tracking loop failed")
                self._report_error(str(exc))

            remaining = FRAME_INTERVAL_S - (time.monotonic() - loop_started)
            if remaining > 0 and self.stop_event.wait(remaining):
                break

    def _process_next_frame(self):
        ok, raw_frame = self.cap.read()
        if not ok:
            self._report_error("Webcam frame read failed.")
            self.stop_event.wait(FRAME_INTERVAL_S)
            return

        raw_frame = self._normalize_capture_frame(raw_frame)
        self.frame_counter += 1

        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.monotonic() - self.started_at) * 1000.0)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        index = vision.select_primary_hand_from_tasks(result)

        if index is None:
            if self._should_publish_video():
                self._publish_video(cv2.flip(raw_frame, 1))
            self._report_status(HandTrigger.ABSENT)
            return

        self._process_detected_hand(raw_frame, result, index)

    def _process_detected_hand(self, raw_frame: np.ndarray, result: Any, index: int):
        hand_landmarks = result.hand_landmarks[index]
        hand = Hand(result.handedness[index][0].category_name.strip().lower())
        height, width = raw_frame.shape[:2]
        should_publish_video = self._should_publish_video()

        if not vision.are_normalized_hand_landmarks_fully_visible(
            hand_landmarks, width, height, margin_px=HAND_LANDMARK_VISIBILITY_MARGIN_PX
        ):
            overlay = None
            if should_publish_video:
                display_pts = np.array(
                    [[(1.0 - lm.x) * width, lm.y * height] for lm in hand_landmarks],
                    dtype=np.float32,
                )
                overlay = vision.draw_hand_overlay(cv2.flip(raw_frame, 1), display_pts)
            self._report_status(HandTrigger.NOT_FULLY_IN_VIEW, hand, overlay)
            return

        raw_pts = np.array(
            [[lm.x * width, lm.y * height, lm.z] for lm in hand_landmarks],
            dtype=np.float32,
        )

        overlay = None
        if should_publish_video:
            display_pts = raw_pts.copy()
            display_pts[:, 0] = (width - 1) - display_pts[:, 0]
            overlay = vision.draw_hand_overlay(cv2.flip(raw_frame, 1), display_pts[:, :2])

        world_pts = np.array(
            [[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[index]],
            dtype=np.float32,
        )

        if not vision.is_palm_side_visible(world_pts, hand, min_confidence=0.3):
            self._report_status(HandTrigger.WRONG_SIDE, hand, overlay)
            return

        if not vision.is_palm_frontal(world_pts, hand):
            self._report_status(HandTrigger.TILTED, hand, overlay)
            return

        geometry = vision.extract_geometry_features(world_pts)
        roi_pose = vision.estimate_roi_pose(raw_pts[:, :2])
        if roi_pose is None:
            self._report_status(HandTrigger.TILTED, hand, overlay)
            return

        raw_roi_quad = vision.roi_quad_from_pose(roi_pose)
        if overlay is not None:
            display_roi_quad = vision.mirror_quad_horizontally(raw_roi_quad, width)
            overlay = vision.draw_roi_quad(overlay, display_roi_quad)

        roi = vision.warp_roi_from_quad(raw_frame, raw_roi_quad, ROI_SIZE)
        roi_enhanced = preprocessing.prepare_cnn_input_roi(roi)

        run_embed = (self.frame_counter % EMBED_EVERY == 0) or not self.embedding_history
        if run_embed:
            self.embedding_history.append(self.cnn.embed_preprocessed(roi_enhanced))
            self.geom_history.append(geometry)
            self._update_feature_cache()

        if self._should_publish_roi():
            self._publish_roi(roi_enhanced)
        if overlay is not None:
            self._publish_video(overlay)

        self._emit(
            HandEvent(
                trigger=HandTrigger.DETECTED,
                hand=hand,
                lengths=dict(self.feature_cache["hand_lengths"]),
                vector=list(self.feature_cache["embedding_vector"]) if run_embed else [],
            )
        )

    def _normalize_capture_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        if width == CAMERA_WIDTH and height == CAMERA_HEIGHT:
            return frame_bgr
        return cv2.resize(frame_bgr, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_AREA)

    def _update_feature_cache(self):
        geom_matrix = np.array(
            [[sample[key] for key in GEOMETRY_FEATURE_KEYS] for sample in self.geom_history],
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
            "embedding_vector": [round(float(value), 6) for value in emb_median.tolist()],
        }

    def _report_status(self, trigger: HandTrigger, hand: Hand | None = None, frame_bgr: np.ndarray | None = None):
        if frame_bgr is not None and self._should_publish_video():
            self._publish_video(frame_bgr)
        self._emit(HandEvent(trigger=trigger, hand=hand))

    def _report_error(self, message: str):
        logger.warning("Hand tracker error: %s", message)
        self._emit(HandEvent(trigger=HandTrigger.ABSENT))

    def _emit(self, event: HandEvent):
        if self._callback is not None:
            self._callback(event)

    def _should_publish_video(self) -> bool:
        return (time.monotonic() - self.last_video_publish_at) >= VIDEO_STREAM_INTERVAL_S

    def _should_publish_roi(self) -> bool:
        return (time.monotonic() - self.last_roi_publish_at) >= VIDEO_STREAM_INTERVAL_S

    def _publish_video(self, frame_bgr: np.ndarray):
        self.video_client.send_message(vision.encode_frame_jpeg(frame_bgr, max_width=VIDEO_STREAM_WIDTH, quality=72))
        self.last_video_publish_at = time.monotonic()

    def _publish_roi(self, roi_frame_bgr: np.ndarray):
        self.roi_client.send_message(vision.encode_frame_jpeg(roi_frame_bgr, max_width=ROI_SIZE, quality=72))
        self.last_roi_publish_at = time.monotonic()

    @staticmethod
    def _empty_feature_cache() -> dict[str, Any]:
        return {"hand_lengths": {}, "embedding_vector": []}
