#!/usr/bin/env python3
from __future__ import annotations

import io
import os
import subprocess
import time
import wave
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

from ArtificialIntelligence.servers.tts.server_config import (
    CUSTOM_VOICE_MANIFEST,
    CUSTOM_VOICE_PROFILE,
    LANGUAGE,
    VOICE_ANCHOR,
    VOICE_ANCHOR_TEXT,
    VOICE_DESIGN_MODEL,
)
SERVER_DIR = Path(__file__).resolve().parent
ROOT = SERVER_DIR.parents[2]
VENV_DIR = SERVER_DIR / ".venv"


def _reference_instructions(description: str) -> str:
    tail = "Keep voice stable. Pronounce clearly. No emotions. Do not add or skip words."
    combined = f"{description.strip()} {tail}"
    if len(combined) > 500:
        combined = combined[:497].rstrip() + "..."
    return combined


def _wait_for_server(url: str, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"VoiceDesign server exited with code {process.returncode}")
        try:
            response = httpx.get(f"{url}/v1/audio/voices", timeout=2)
            if response.status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError("Timed out waiting for VoiceDesign server")


def _trim_trailing_silence(audio: np.ndarray, sr: int, silence_threshold: float = 0.015, min_duration: float = 1.0) -> np.ndarray:
    """Trim trailing silence/static from audio while keeping leading speech intact."""
    window = int(0.05 * sr)
    rms = np.sqrt(np.mean(audio[: len(audio) - len(audio) % window].reshape(-1, window) ** 2, axis=1))
    above = np.flatnonzero(rms > silence_threshold)
    if above.size == 0:
        return audio[: int(min_duration * sr)]

    # Detect gaps of silence > 1 second — speech ended and static follows
    gap_threshold = int(1.0 / 0.05)
    gaps = np.diff(above)
    big_gaps = np.flatnonzero(gaps > gap_threshold)
    if big_gaps.size > 0:
        last_active = above[big_gaps[0]] * window
    else:
        last_active = above[-1] * window

    keep = min(last_active + int(0.3 * sr), len(audio))
    return audio[:keep]


def _generate_reference(url: str, description: str) -> None:
    response = httpx.post(
        f"{url}/v1/audio/speech",
        json={
            "input": VOICE_ANCHOR_TEXT,
            "model": VOICE_DESIGN_MODEL,
            "task_type": "VoiceDesign",
            "language": LANGUAGE,
            "instructions": _reference_instructions(description),
            "response_format": "wav",
            "max_new_tokens": 4096,
            "stream": False,
            "seed": 42,
        },
        timeout=300,
    )
    response.raise_for_status()
    with wave.open(io.BytesIO(response.content), "rb") as audio:
        if (
            audio.getnchannels() != 1
            or audio.getsampwidth() != 2
            or audio.getframerate() != 24_000
            or audio.getnframes() == 0
        ):
            raise RuntimeError("VoiceDesign returned an invalid anchor; expected non-empty mono 24 kHz 16-bit PCM WAV")

    raw_wav, sr = sf.read(io.BytesIO(response.content), dtype="float32")
    trimmed = _trim_trailing_silence(raw_wav, sr)
    sf.write(str(VOICE_ANCHOR), trimmed, sr)
    CUSTOM_VOICE_MANIFEST.unlink(missing_ok=True)
    CUSTOM_VOICE_PROFILE.unlink(missing_ok=True)
    print(
        f"[tts-design] Created reference recording: {VOICE_ANCHOR} "
        f"({raw_wav.shape[0] / sr:.1f}s -> {trimmed.shape[0] / sr:.1f}s)",
        flush=True,
    )


def _stop_server(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run(description: str | None = None) -> None:
    load_dotenv(ROOT / ".env")
    if description is None:
        description = os.environ["WITCH_TTS_VOICE_DESCRIPTION"]
    port = int(os.environ["WITCH_TTS_PORT"]) + 1
    url = f"http://127.0.0.1:{port}"
    executable = VENV_DIR / "bin" / "vllm-omni"
    deploy_config = SERVER_DIR / "qwen3_tts_witch.yaml"
    command = [
        str(executable),
        "serve",
        VOICE_DESIGN_MODEL,
        "--deploy-config",
        str(deploy_config),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--generation-config",
        "vllm",
        "--trust-remote-code",
        "--omni",
    ]

    print(f"[tts-design] Starting VoiceDesign model on port {port}", flush=True)
    process = subprocess.Popen(command, cwd=ROOT)
    try:
        _wait_for_server(url, process)
        _generate_reference(url, description)
    finally:
        _stop_server(process)

    print(f"[tts-design] Reference transcript: {VOICE_ANCHOR_TEXT}", flush=True)
    print(f"[tts-design] Run 'witch-compose tts' to compute the voice profile from this anchor.", flush=True)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
