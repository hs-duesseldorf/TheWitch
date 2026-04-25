from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
RUNTIME_CACHE_DIR = PROJECT_ROOT / ".runtime-cache"
MPLCONFIG_DIR = RUNTIME_CACHE_DIR / "matplotlib"
DEFAULT_HAND_MODEL_PATH = ASSETS_DIR / "models" / "hand_landmarker.task"
DEFAULT_AI_PIPELINE_WS_URL = "ws://localhost:8001/ws/palmprint"
EMBEDDING_MODEL_CHOICES = ("arcface", "contrastive")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    camera: str
    embedding_model: str
    camera_fps: float = 0.0
    width: int = 0
    height: int = 0
    interval_ms: int = 0
    history_size: int = 45
    roi_size: int = 256
    embed_every: int = 1
    roi_brightness: float = -8.0
    roi_contrast: float = 1.2
    roi_gamma: float = 1.1
    roi_clahe_clip_limit: float = 1.8
    roi_clahe_tile_size: int = 8
    hand_model_path: Path = DEFAULT_HAND_MODEL_PATH


@dataclass(frozen=True, slots=True)
class TransportConfig:
    pipeline_ws_url: str


@dataclass(frozen=True, slots=True)
class AppConfig:
    runtime: RuntimeConfig
    transport: TransportConfig


def prepare_runtime_environment() -> None:
    MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Palmprint runtime websocket client")
    parser.add_argument("--camera", type=str, default="0", help="OpenCV camera index or source string")
    parser.add_argument(
        "--embedding_model",
        type=str,
        choices=EMBEDDING_MODEL_CHOICES,
        default="arcface",
        help="Embedding model id to use at startup",
    )
    parser.add_argument(
        "--pipeline_ws_url",
        type=str,
        default=os.getenv("PALMPRINT_AI_WS_URL", DEFAULT_AI_PIPELINE_WS_URL),
        help=(
            "Remote AI websocket URL for outbound feature vectors "
            f"(local default: PALMPRINT_AI_WS_URL or {DEFAULT_AI_PIPELINE_WS_URL}; "
            "on Jetson use ws://<PC-LAN-IP>:8001/ws/palmprint)"
        ),
    )
    args = parser.parse_args()

    runtime = RuntimeConfig(
        camera=args.camera,
        embedding_model=args.embedding_model,
    )
    transport = TransportConfig(
        pipeline_ws_url=(args.pipeline_ws_url or os.getenv("PALMPRINT_AI_WS_URL", DEFAULT_AI_PIPELINE_WS_URL)).strip()
        or DEFAULT_AI_PIPELINE_WS_URL
    )
    return AppConfig(runtime=runtime, transport=transport)
