#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

SERVER_DIR = Path(__file__).resolve().parent
ROOT = SERVER_DIR.parents[2]
PATCH_DIR = SERVER_DIR / "python_patches"


def run() -> None:
    load_dotenv(ROOT / ".env")
    model = os.environ["WITCH_TTS_MODEL"]
    port = os.environ["WITCH_TTS_PORT"]
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    existing_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        str(PATCH_DIR)
        if not existing_pythonpath
        else os.pathsep.join((str(PATCH_DIR), existing_pythonpath))
    )
    print(f"[run_tts] Starting TTS: model={model} port={port}", flush=True)
    python = SERVER_DIR / ".venv" / "bin" / "vllm-omni"
    deploy_config = SERVER_DIR / "qwen3_tts_witch.yaml"
    os.execv(str(python), [str(python), "serve", model,
        "--deploy-config", str(deploy_config),
        "--host", "0.0.0.0",
        "--port", port,
        "--gpu-memory-utilization", os.environ["TTS_GPU_MEMORY_UTILIZATION"],
        "--generation-config", "vllm",
        "--trust-remote-code",
        "--omni",
    ])


if __name__ == "__main__":
    run()
