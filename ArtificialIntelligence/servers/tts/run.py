#!/usr/bin/env python3
from __future__ import annotations

import os
import vllm_omni
from pathlib import Path

from dotenv import load_dotenv

SERVER_DIR = Path(__file__).resolve().parent
ROOT = SERVER_DIR.parents[2]


def run() -> None:
    load_dotenv(ROOT / ".env")
    model = os.environ["WITCH_TTS_MODEL"]
    port = os.environ["WITCH_TTS_PORT"]
    print(f"[run_tts] Starting TTS: model={model} port={port}", flush=True)
    python = SERVER_DIR / ".venv" / "bin" / "vllm-omni"
    deploy_config = Path(vllm_omni.__file__).resolve().parent / "deploy" / "qwen3_tts.yaml"
    os.execv(str(python), [str(python), "serve", model,
        "--deploy-config", str(deploy_config),
        "--host", "0.0.0.0",
        "--port", port,
        "--gpu-memory-utilization", os.environ["TTS_GPU_MEMORY_UTILIZATION"],
        "--trust-remote-code",
        "--omni",
    ])


if __name__ == "__main__":
    run()
