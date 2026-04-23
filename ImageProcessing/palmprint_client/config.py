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
DEFAULT_CNN_WEIGHTS_PATH = ASSETS_DIR / "weights" / "tongji_resnet18_arcface_256d.pt"
DEFAULT_AI_PIPELINE_WS_URL = "ws://127.0.0.1:8001/ws/palmprint"


def _non_negative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return value


def _non_negative_float(raw: str) -> float:
    value = float(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    camera: str
    camera_fps: float
    width: int
    height: int
    interval_ms: int
    history_size: int
    roi_size: int
    embed_every: int
    roi_brightness: float
    roi_contrast: float
    roi_gamma: float
    roi_clahe_clip_limit: float
    roi_clahe_tile_size: int
    hand_model_path: Path
    cnn_weights_path: Path


@dataclass(frozen=True, slots=True)
class ServerConfig:
    host: str
    port: int


@dataclass(frozen=True, slots=True)
class TransportConfig:
    pipeline_ws_url: str


@dataclass(frozen=True, slots=True)
class AppConfig:
    runtime: RuntimeConfig
    server: ServerConfig
    transport: TransportConfig
    seed: int


def prepare_runtime_environment() -> None:
    MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Palmprint client with FastAPI debug server")
    parser.add_argument("--camera", type=str, default="0", help="OpenCV camera index or source string")
    parser.add_argument("--camera_fps", type=_non_negative_float, default=0.0, help="Requested camera FPS; 0 leaves the driver default")
    parser.add_argument("--width", type=_non_negative_int, default=0, help="Camera width override; 0 keeps the camera default mode")
    parser.add_argument("--height", type=_non_negative_int, default=0, help="Camera height override; 0 keeps the camera default mode")
    parser.add_argument("--interval_ms", type=_non_negative_int, default=0, help="Processing interval target (ms); 0 runs uncapped")
    parser.add_argument("--history_size", type=_positive_int, default=45, help="Rolling aggregation window")
    parser.add_argument("--roi_size", type=_positive_int, default=256, help="ROI warp output size")
    parser.add_argument("--embed_every", type=_positive_int, default=1, help="Run the CNN every N frames (default: every ROI frame)")
    parser.add_argument("--roi_brightness", type=float, default=-8.0, help="Additive grayscale brightness applied to the CNN ROI input")
    parser.add_argument("--roi_contrast", type=float, default=1.2, help="Multiplicative grayscale contrast applied to the CNN ROI input")
    parser.add_argument("--roi_gamma", type=_positive_float, default=1.1, help="Gamma applied to the CNN ROI input; values above 1 darken highlights")
    parser.add_argument("--roi_clahe_clip_limit", type=_non_negative_float, default=1.8, help="CLAHE clip limit for the CNN ROI input; set 0 to disable")
    parser.add_argument("--roi_clahe_tile_size", type=_positive_int, default=8, help="CLAHE tile size for the CNN ROI input")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="FastAPI bind host")
    parser.add_argument("--port", type=_positive_int, default=8000, help="FastAPI bind port")
    parser.add_argument(
        "--pipeline_ws_url",
        type=str,
        default=os.getenv("PALMPRINT_AI_WS_URL", DEFAULT_AI_PIPELINE_WS_URL),
        help=(
            "Remote AI websocket URL for outbound feature vectors "
            f"(default: PALMPRINT_AI_WS_URL or {DEFAULT_AI_PIPELINE_WS_URL})"
        ),
    )
    parser.add_argument("--seed", type=int, default=1234, help="Deterministic seed")
    parser.add_argument(
        "--hand_model",
        type=Path,
        default=DEFAULT_HAND_MODEL_PATH,
        help="Path to the MediaPipe hand_landmarker.task model",
    )
    parser.add_argument(
        "--cnn_weights",
        type=Path,
        default=DEFAULT_CNN_WEIGHTS_PATH,
        help="Optional local CNN weights file for texture embeddings",
    )
    args = parser.parse_args()

    runtime = RuntimeConfig(
        camera=args.camera,
        camera_fps=args.camera_fps,
        width=args.width,
        height=args.height,
        interval_ms=args.interval_ms,
        history_size=args.history_size,
        roi_size=args.roi_size,
        embed_every=args.embed_every,
        roi_brightness=args.roi_brightness,
        roi_contrast=args.roi_contrast,
        roi_gamma=args.roi_gamma,
        roi_clahe_clip_limit=args.roi_clahe_clip_limit,
        roi_clahe_tile_size=args.roi_clahe_tile_size,
        hand_model_path=Path(args.hand_model),
        cnn_weights_path=Path(args.cnn_weights),
    )
    server = ServerConfig(host=args.host, port=args.port)
    transport = TransportConfig(
        pipeline_ws_url=(args.pipeline_ws_url or os.getenv("PALMPRINT_AI_WS_URL", DEFAULT_AI_PIPELINE_WS_URL)).strip()
        or DEFAULT_AI_PIPELINE_WS_URL
    )
    return AppConfig(runtime=runtime, server=server, transport=transport, seed=args.seed)
