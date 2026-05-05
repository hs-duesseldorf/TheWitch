from __future__ import annotations

import copy
import threading
import time
from collections import deque
from typing import Any, Callable

import cv2
import mediapipe as mp
import numpy as np
import torch

from .config import RuntimeConfig
from .models import TextureCNN, create_hand_landmarker
from .payloads import build_feature_vector_message, build_status_message
from .preprocessing import RoiToneSettings
from .transport import MessageQueue
from .utils import l2_normalize, parse_camera_source
from .vision import (
    draw_hand_overlay,
    draw_roi_quad,
    encode_frame_data_url,
    estimate_roi_pose,
    extract_geometry_features,
    is_palm_frontal,
    is_palm_side_visible,
    mirror_quad_horizontally,
    render_placeholder_frame,
    render_roi_feature_preview,
    roi_quad_from_pose,
    select_primary_hand_from_tasks,
    warp_roi_from_quad,
)

DEBUG_STREAM_INTERVAL_S = 1.0 / 30.0


def resolve_torch_device(requested_device: str) -> tuple[torch.device, str]:
    requested = (requested_device or "auto").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0"), f"cuda ({torch.cuda.get_device_name(0)})"
        if torch.backends.mps.is_available():
            return torch.device("mps"), "mps (Apple Metal)"
        return torch.device("cpu"), "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Torch device 'cuda' was requested, but CUDA is not available on this machine.")
        return torch.device("cuda:0"), f"cuda ({torch.cuda.get_device_name(0)})"

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Torch device 'mps' was requested, but Apple Metal/MPS is not available in this PyTorch build.")
        return torch.device("mps"), "mps (Apple Metal)"

    if requested == "cpu":
        return torch.device("cpu"), "cpu"

    raise ValueError(f"Unsupported torch device: {requested_device!r}")


class HeadlessPalmClient:
    def __init__(
        self,
        config: RuntimeConfig,
        *,
        feature_queue: MessageQueue,
        pipeline_status_provider: Callable[[], dict[str, Any]],
    ):
        self.config = config
        self.feature_queue = feature_queue
        self.pipeline_status_provider = pipeline_status_provider

        self.camera_source = parse_camera_source(config.camera)
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.started_at = time.monotonic()
        self.frame_counter = 0
        self.frame_timestamps: deque[float] = deque(maxlen=90)
        self.last_frame_size = [config.width, config.height]
        self.last_status_publish_at = 0.0
        self.last_status_signature: tuple[Any, ...] | None = None
        self.last_debug_frame_publish_at = 0.0
        self.model_lock = threading.RLock()

        self.cap = cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source {self.camera_source!r}")
        self._configure_capture_resolution()
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.landmarker = create_hand_landmarker(config.hand_model_path)
        self.device, self.device_desc = resolve_torch_device(config.device)
        self.roi_tone_settings = RoiToneSettings(
            brightness=config.roi_brightness,
            contrast=config.roi_contrast,
            gamma=config.roi_gamma,
            clahe_clip_limit=config.roi_clahe_clip_limit,
            clahe_tile_size=config.roi_clahe_tile_size,
        )
        self.cnn = TextureCNN(
            device=self.device,
            embedding_model=config.embedding_model,
            roi_tone_settings=self.roi_tone_settings,
        )

        self.geom_history: deque[dict[str, float]] = deque(maxlen=config.history_size)
        self.embedding_history: deque[np.ndarray] = deque(maxlen=config.history_size)
        self.last_embedding: np.ndarray | None = None
        self.feature_cache = self._empty_feature_cache()
        self.feature_queue.publish(self._ui_payload(self._build_status_message(
            status="starting",
            message="Starting capture pipeline.",
            hand_detected=False,
        )))

    def _reset_feature_state_unlocked(self) -> None:
        self.geom_history.clear()
        self.embedding_history.clear()
        self.last_embedding = None
        self.feature_cache = self._empty_feature_cache()

    def embedding_model_status(self) -> dict[str, Any]:
        with self.model_lock:
            return self.cnn.status_payload()

    def select_embedding_model(self, model_id: str) -> dict[str, Any]:
        with self.model_lock:
            if model_id == self.cnn.model_id:
                return self.cnn.status_payload()

        cnn = TextureCNN(
            device=self.device,
            embedding_model=model_id,
            roi_tone_settings=self.roi_tone_settings,
        )
        with self.model_lock:
            self.cnn = cnn
            self._reset_feature_state_unlocked()

        payload = self._ui_payload(
            self._build_status_message(
                status="starting",
                message=f"Switched embedding model to {cnn.display_name}.",
                hand_detected=False,
            )
        )
        self.feature_queue.publish(payload)
        return cnn.status_payload()

    def handle_command(self, message: dict[str, Any]) -> dict[str, Any] | None:
        command = message.get("type") or message.get("command")
        if command != "select_embedding_model":
            return {"type": "command_result", "ok": False, "error": f"Unknown command: {command}"}

        model_id = message.get("model_id")
        if not isinstance(model_id, str) or not model_id.strip():
            return {"type": "command_result", "ok": False, "error": "model_id is required"}

        try:
            active = self.select_embedding_model(model_id.strip())
        except Exception as exc:
            return {"type": "command_result", "ok": False, "error": str(exc)}

        return {"type": "command_result", "ok": True, "active": active}

    def _configure_capture_resolution(self) -> None:
        if self.config.camera_fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
        if self.config.width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        if self.config.height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)

    def _empty_feature_cache(self) -> dict[str, Any]:
        return {
            "hand_proportions": {},
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

    def _processing_fps(self) -> float:
        if len(self.frame_timestamps) < 2:
            return 0.0
        span = self.frame_timestamps[-1] - self.frame_timestamps[0]
        if span <= 1e-9:
            return 0.0
        return (len(self.frame_timestamps) - 1) / span

    def _build_status_message(
        self,
        *,
        status: str,
        message: str,
        hand_detected: bool,
        hand_label: str = "unknown",
    ) -> dict[str, Any]:
        return build_status_message(
            status=status,
            message=message,
            hand_detected=hand_detected,
            hand_label=hand_label,
        )

    def _ui_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        ui_payload = copy.deepcopy(payload)
        with self.model_lock:
            embedding_model = self.cnn.status_payload()
        ui_payload["fps"] = round(self._processing_fps(), 1)
        ui_payload["pipeline"] = self.pipeline_status_provider()
        ui_payload["runtime"] = {
            "uptime_s": round(time.monotonic() - self.started_at, 2),
            "frame_size": copy.deepcopy(self.last_frame_size),
            "camera_source": str(self.camera_source),
            "camera_fps_request": self.config.camera_fps,
            "device": self.device_desc,
            "cnn_source": embedding_model["source"],
            "embedding_model": embedding_model,
            "embed_every": self.config.embed_every,
        }
        return ui_payload

    def _should_publish_debug_frame(self) -> bool:
        return (time.monotonic() - self.last_debug_frame_publish_at) >= DEBUG_STREAM_INTERVAL_S

    def _debug_message(
        self,
        payload: dict[str, Any],
        camera_frame_bgr: np.ndarray | None,
        roi_frame_bgr: np.ndarray | None,
    ) -> dict[str, Any]:
        if camera_frame_bgr is None and roi_frame_bgr is None:
            return payload

        debug_payload = copy.deepcopy(payload)
        debug: dict[str, str] = {}
        if camera_frame_bgr is not None:
            debug["camera_frame_jpeg"] = encode_frame_data_url(camera_frame_bgr)
        if roi_frame_bgr is not None:
            debug["roi_frame_jpeg"] = encode_frame_data_url(roi_frame_bgr, max_width=320, quality=78)
        debug_payload["debug"] = debug
        self.last_debug_frame_publish_at = time.monotonic()
        return debug_payload

    def _publish_status(
        self,
        *,
        status: str,
        message: str,
        hand_detected: bool,
        hand_label: str = "unknown",
        camera_frame_bgr: np.ndarray | None = None,
        roi_frame_bgr: np.ndarray | None = None,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        signature = (status, message, hand_detected, hand_label)
        frame_due = (camera_frame_bgr is not None or roi_frame_bgr is not None) and self._should_publish_debug_frame()
        if not force and not frame_due and self.last_status_signature == signature and (now - self.last_status_publish_at) < 0.5:
            return

        payload = self._build_status_message(
            status=status,
            message=message,
            hand_detected=hand_detected,
            hand_label=hand_label,
        )
        self.last_status_signature = signature
        self.last_status_publish_at = now
        self.feature_queue.publish(
            self._debug_message(
                self._ui_payload(payload),
                camera_frame_bgr if frame_due or force else None,
                roi_frame_bgr if frame_due or force else None,
            )
        )

    def _publish_feature_message(
        self,
        hand_label: str,
        camera_frame_bgr: np.ndarray | None = None,
        roi_frame_bgr: np.ndarray | None = None,
    ) -> None:
        payload = build_feature_vector_message(
            status="running",
            hand_label=hand_label,
            embedding_vector=self.feature_cache["embedding_vector"],
            hand_proportions=self.feature_cache["hand_proportions"],
        )
        frame_due = self._should_publish_debug_frame()
        self.feature_queue.publish(
            self._debug_message(
                self._ui_payload(payload),
                camera_frame_bgr if frame_due else None,
                roi_frame_bgr if frame_due else None,
            )
        )

    def _update_feature_cache(self, geometry: dict[str, float]) -> None:
        geom_keys = sorted(geometry.keys())
        geom_matrix = np.array([[sample[key] for key in geom_keys] for sample in self.geom_history], dtype=np.float32)
        geom_median = np.median(geom_matrix, axis=0)

        emb_matrix = np.stack(list(self.embedding_history), axis=0)
        emb_median = l2_normalize(np.median(emb_matrix, axis=0).astype(np.float32))
        self.feature_cache = {
            "hand_proportions": {
                key: round(float(value), 6)
                for key, value in zip(geom_keys, geom_median.tolist())
            },
            "embedding_vector": [round(float(value), 6) for value in emb_median.tolist()],
        }

    def _publish_no_hand(self, frame_bgr: np.ndarray) -> None:
        self._publish_status(
            status="idle",
            message="No hand detected.",
            hand_detected=False,
            camera_frame_bgr=frame_bgr,
            roi_frame_bgr=render_placeholder_frame(self.config.roi_size, self.config.roi_size, "", ""),
        )

    def _publish_roi_failure(self, hand_label: str, frame_bgr: np.ndarray) -> None:
        self._publish_status(
            status="degraded",
            message="Hand detected but ROI warp failed.",
            hand_detected=True,
            hand_label=hand_label,
            camera_frame_bgr=frame_bgr,
            roi_frame_bgr=render_placeholder_frame(self.config.roi_size, self.config.roi_size, "ROI failed", "Adjust palm pose"),
        )

    def _publish_palm_side_required(self, hand_label: str, frame_bgr: np.ndarray) -> None:
        self._publish_status(
            status="degraded",
            message="Falsche Handseite. Bitte dreh deine Hand, damit die Handflaeche sichtbar ist.",
            hand_detected=True,
            hand_label=hand_label,
            camera_frame_bgr=frame_bgr,
            roi_frame_bgr=render_placeholder_frame(self.config.roi_size, self.config.roi_size, "Falsche Seite", "Hand drehen"),
        )

    def _publish_pose_quality(self, hand_label: str, frame_bgr: np.ndarray) -> None:
        self._publish_status(
            status="degraded",
            message="Handflaeche ist nicht gerade genug. Bitte sehr frontal zur Kamera halten.",
            hand_detected=True,
            hand_label=hand_label,
            camera_frame_bgr=frame_bgr,
            roi_frame_bgr=render_placeholder_frame(self.config.roi_size, self.config.roi_size, "Hand gerade", "Sehr frontal halten"),
        )

    def _publish_error(self, message: str) -> None:
        self._publish_status(
            status="error",
            message=message,
            hand_detected=False,
            force=True,
        )

    def _handedness(self, result: Any, index: int) -> str:
        if result.handedness and index < len(result.handedness) and result.handedness[index]:
            category = result.handedness[index][0]
            return category.category_name
        return "unknown"

    def _display_handedness(self, hand_label: str) -> str:
        label = (hand_label or "").strip().lower()
        if label == "left":
            return "Right"
        if label == "right":
            return "Left"
        return hand_label

    def _run_loop(self) -> None:
        target_period = max(self.config.interval_ms / 1000.0, 0.0)
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
                self.last_frame_size = [int(display_frame.shape[1]), int(display_frame.shape[0])]
                self.frame_counter += 1
                self.frame_timestamps.append(time.monotonic())

                rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((time.monotonic() - self.started_at) * 1000.0)
                result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
                index = select_primary_hand_from_tasks(result)

                if index is None:
                    self._publish_no_hand(display_frame)
                else:
                    hand = result.hand_landmarks[index]
                    hand_label = self._handedness(result, index)
                    display_hand_label = self._display_handedness(hand_label)

                    height, width = display_frame.shape[:2]
                    display_pts = np.array([[lm.x * width, lm.y * height, lm.z] for lm in hand], dtype=np.float32)
                    raw_pts = display_pts.copy()
                    raw_pts[:, 0] = (width - 1) - raw_pts[:, 0]

                    debug_hand_frame = draw_hand_overlay(display_frame, display_pts[:, :2])
                    world_pts = None
                    if result.hand_world_landmarks and index < len(result.hand_world_landmarks):
                        world = result.hand_world_landmarks[index]
                        world_pts = np.array([[lm.x, lm.y, lm.z] for lm in world], dtype=np.float32)

                    palm_side_points = world_pts
                    if world_pts is None:
                        palm_side_points = np.array([[lm.x * width, lm.y * height, lm.z] for lm in hand], dtype=np.float32)
                    if not is_palm_side_visible(np.asarray(palm_side_points), hand_label, min_confidence=0.3):
                        self._publish_palm_side_required(display_hand_label, debug_hand_frame)
                        continue

                    points3d = world_pts if world_pts is not None else raw_pts
                    if not is_palm_frontal(np.asarray(palm_side_points), hand_label):
                        self._publish_pose_quality(display_hand_label, debug_hand_frame)
                        continue

                    geometry = extract_geometry_features(points3d)
                    current_roi_pose = estimate_roi_pose(raw_pts[:, :2])
                    if current_roi_pose is None:
                        self._publish_roi_failure(display_hand_label, debug_hand_frame)
                        continue

                    raw_roi_quad = roi_quad_from_pose(current_roi_pose)
                    display_roi_quad = mirror_quad_horizontally(raw_roi_quad, width)
                    debug_frame = draw_roi_quad(debug_hand_frame, display_roi_quad)
                    roi = warp_roi_from_quad(raw_frame, raw_roi_quad, self.config.roi_size)

                    if roi is None:
                        self._publish_roi_failure(hand_label, debug_frame)
                    else:
                        with self.model_lock:
                            run_embed = (self.frame_counter % self.config.embed_every == 0) or (self.last_embedding is None)
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
                            display_hand_label,
                            debug_frame,
                            render_roi_feature_preview(roi, self.roi_tone_settings),
                        )
            except Exception as exc:
                self._publish_error(str(exc))

            elapsed = time.monotonic() - loop_started
            remaining = target_period - elapsed
            if remaining > 0 and self.stop_event.wait(remaining):
                break
