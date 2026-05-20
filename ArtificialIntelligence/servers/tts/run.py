#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

SERVER_DIR = Path(__file__).resolve().parent
ROOT = SERVER_DIR.parents[2]


def run() -> None:
    load_dotenv(ROOT / ".env")
    model = os.environ["WITCH_TTS_MODEL"]
    port = os.environ["WITCH_TTS_PORT"]
    print(f"[run_tts] Starting TTS: model={model} port={port}", flush=True)
    python = SERVER_DIR / ".venv" / "bin" / "vllm"
    os.execv(str(python), [str(python), "serve", model,
        "--host", "0.0.0.0",
        "--port", port,
        "--max-model-len", os.environ["TTS_MAX_MODEL_LEN"],
        "--gpu-memory-utilization", os.environ["TTS_GPU_MEMORY_UTILIZATION"],
        "--trust-remote-code",
        "--enforce-eager",
        "--max-num-seqs", "1",
    ])


if __name__ == "__main__":
    run()