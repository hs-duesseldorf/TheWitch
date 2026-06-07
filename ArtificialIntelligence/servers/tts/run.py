#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import httpx
import uvicorn
from dotenv import load_dotenv

from ArtificialIntelligence.servers.tts.gateway import app
from ArtificialIntelligence.servers.tts.server_config import profile

SERVER_DIR = Path(__file__).resolve().parent
ROOT = SERVER_DIR.parents[2]
VENV_DIR = SERVER_DIR / ".venv"


def _wait_for_server(url: str, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM-Omni exited with code {process.returncode}")
        try:
            response = httpx.get(f"{url}/v1/audio/voices", timeout=2)
            if response.status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError("Timed out waiting for vLLM-Omni TTS server")


def _ensure_custom_voice_profile() -> None:
    from ArtificialIntelligence.servers.tts.precompute_voice import precompute_voice
    from ArtificialIntelligence.servers.tts.server_config import (
        BASE_MODEL,
        CUSTOM_VOICE_DIR,
        CUSTOM_VOICE_PROFILE,
        VOICE_ANCHOR,
        VOICE_ANCHOR_TEXT,
        VOICE_DESCRIPTION,
        VOICE_NAME,
    )

    if not VOICE_ANCHOR.is_file():
        return
    if CUSTOM_VOICE_PROFILE.is_file():
        print(f"[run_tts] Reusing custom voice profile {CUSTOM_VOICE_PROFILE}", flush=True)
        return

    print(f"[run_tts] Precomputing voice profile from {VOICE_ANCHOR}...", flush=True)
    precompute_voice(
        model=BASE_MODEL,
        voice_name=VOICE_NAME,
        ref_audio=VOICE_ANCHOR,
        ref_text=VOICE_ANCHOR_TEXT,
        speaker_description=VOICE_DESCRIPTION,
        output_dir=CUSTOM_VOICE_DIR,
    )


def run() -> None:
    load_dotenv(ROOT / ".env")
    _ensure_custom_voice_profile()
    selected = profile()
    public_port = int(os.environ["WITCH_TTS_PORT"])
    internal_port = public_port + 1
    internal_url = f"http://127.0.0.1:{internal_port}"
    os.environ["WITCH_TTS_INTERNAL_PORT"] = str(internal_port)
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
    executable = VENV_DIR / "bin" / "vllm-omni"
    deploy_config = SERVER_DIR / "qwen3_tts_witch.yaml"
    command = [
        str(executable),
        "serve",
        selected.model,
        "--deploy-config",
        str(deploy_config),
        "--host",
        "127.0.0.1",
        "--port",
        str(internal_port),
        "--generation-config",
        "vllm",
        "--max-model-len",
        os.environ["TTS_MAX_MODEL_LEN"],
        "--trust-remote-code",
        "--omni",
    ]

    mode = f"anchored voice '{selected.voice}'" if selected.uses_anchor else "VoiceDesign fallback"
    print(
        f"[run_tts] Starting internal vLLM-Omni: model={selected.model} port={internal_port} mode={mode}",
        flush=True,
    )
    process = subprocess.Popen(command, cwd=ROOT)
    try:
        _wait_for_server(internal_url, process)
        print(f"[run_tts] Starting text-only TTS gateway on port {public_port}", flush=True)
        uvicorn.run(app, host="0.0.0.0", port=public_port)
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    run()
