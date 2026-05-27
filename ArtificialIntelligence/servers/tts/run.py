#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

SERVER_DIR = Path(__file__).resolve().parent
ROOT = SERVER_DIR.parents[2]
FISH_DIR = SERVER_DIR / "fish-speech"
CHECKPOINT_DIR = SERVER_DIR / "checkpoints" / "s2-pro"


def run() -> None:
    load_dotenv(ROOT / ".env")

    port = os.environ["WITCH_TTS_PORT"]
    codec_path = CHECKPOINT_DIR / "codec.pth"

    if not FISH_DIR.exists():
        raise RuntimeError(f"Fish Speech repo not found: {FISH_DIR}")

    if not CHECKPOINT_DIR.exists():
        raise RuntimeError(
            f"Fish S2 checkpoint folder not found: {CHECKPOINT_DIR}\n"
            "Download it with:\n"
            "  ./ArtificialIntelligence/servers/tts/.venv/bin/hf download fishaudio/s2-pro "
            "--local-dir ArtificialIntelligence/servers/tts/checkpoints/s2-pro"
        )

    if not codec_path.exists():
        raise RuntimeError(f"Fish codec checkpoint not found: {codec_path}")

    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

    print(f"[run_tts] Starting TTS: Model=Fish S2 native server on port={port}", flush=True)

    os.chdir(FISH_DIR)

    os.execvp(
        sys.executable,
        [
            sys.executable,
            "tools/api_server.py",
            "--llama-checkpoint-path",
            str(CHECKPOINT_DIR),
            "--decoder-checkpoint-path",
            str(codec_path),
            "--listen",
            f"0.0.0.0:{port}",
            "--half",
        ],
    )


if __name__ == "__main__":
    run()