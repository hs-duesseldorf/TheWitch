#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

SERVER_DIR = Path(__file__).resolve().parent
ROOT = SERVER_DIR.parents[2]
PATCH_DIR = ROOT / "scripts" / "tts_python_patches"
PATCH_INSTALLER = ROOT / "scripts" / "patch_tts_server_venv.py"
VENV_DIR = SERVER_DIR / ".venv"


def _install_patches() -> None:
    python = VENV_DIR / "bin" / "python"
    if not python.exists():
        raise RuntimeError(f"TTS venv Python not found: {python}")
    subprocess.check_call(
        [
            sys.executable,
            str(PATCH_INSTALLER),
            "--python",
            str(python),
        ],
        cwd=ROOT,
    )


def run() -> None:
    load_dotenv(ROOT / ".env")
    _install_patches()
    model = os.environ["WITCH_TTS_MODEL"]
    port = os.environ["WITCH_TTS_PORT"]
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
    existing_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        str(PATCH_DIR)
        if not existing_pythonpath
        else os.pathsep.join((str(PATCH_DIR), existing_pythonpath))
    )
    print(f"[run_tts] Starting TTS: model={model} port={port}", flush=True)
    python = VENV_DIR / "bin" / "vllm-omni"
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
