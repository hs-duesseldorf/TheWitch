from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from omnivoice import OmniVoice
from omnivoice_config import (
    DEVICE,
    LANGUAGE,
    NUM_STEP,
    REF_AUDIO,
    REF_TEXT,
    SAMPLE_RATE,
)


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tts-server")

load_dotenv()

MODEL_NAME = os.getenv("WITCH_TTS_MODEL")
SPEAKER_DIR = Path("/assets")
SPEAKER_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("/tmp/thewitch-tts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class TTSRequest(BaseModel):
    text: str
    speaker_wav: str | None = None
    language: str = LANGUAGE
    speed: float = 1.0


tts_engine: OmniVoice | None = None


def resolve_device() -> str:
    if DEVICE:
        if DEVICE.startswith("cuda") and not torch.cuda.is_available():
            logger.debug("CUDA requested but unavailable, falling back to CPU")
        else:
            return DEVICE

    if torch.cuda.is_available():
        return "cuda:0"
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    
    return "cpu"


def resolve_dtype(device: str):
    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


def get_tts() -> OmniVoice:
    global tts_engine

    if tts_engine is None:
        device = resolve_device()
        dtype = resolve_dtype(device)

        logger.debug(
            "Loading OmniVoice model=%s device=%s dtype=%s",
            MODEL_NAME,
            device,
            dtype,
        )

        tts_engine = OmniVoice.from_pretrained(
            MODEL_NAME,
            device_map=device,
            dtype=dtype,
        )

        logger.debug("OmniVoice model loaded")

    return tts_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # preload the model instead of waiting for first request
    get_tts()
    yield


api = FastAPI(title="The Witch TTS Server", lifespan=lifespan)


@api.get("/health")
async def health():
    device = resolve_device()
    return {
        "ok": True,
        "model": MODEL_NAME,
        "device": device,
        "language": LANGUAGE,
        "sample_rate": SAMPLE_RATE,
        "cuda_available": torch.cuda.is_available(),
    }


@api.post("/api/generate")
async def synthesize(req: TTSRequest):
    try:
        engine = get_tts()

        text = req.text.strip()
        if not text:
            return JSONResponse({"error": "text is empty"}, status_code=400)

        ref_audio = req.speaker_wav or REF_AUDIO
        ref_text = REF_TEXT

        if not ref_audio:
            return JSONResponse({"error": "OmniVoice REF_AUDIO is empty"}, status_code=500)
        if not ref_text:
            return JSONResponse({"error": "OmniVoice REF_TEXT is empty"}, status_code=500)

        logger.debug(
            "Synthesizing OmniVoice voice clone: chars=%d lang=%s speed=%s",
            len(text),
            req.language or LANGUAGE,
            req.speed
        )

        generate_kwargs = {
            "text": text,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "speed": req.speed,
            "num_step": NUM_STEP
        }

        audio_list = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: engine.generate(**generate_kwargs),
        )

        if not audio_list:
            return JSONResponse({"error": "OmniVoice returned no audio"}, status_code=500)

        audio = np.asarray(audio_list[0], dtype=np.float32)

        buffer = io.BytesIO()
        sf.write(buffer, audio, SAMPLE_RATE, format="WAV")
        audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return JSONResponse(
            {
                "audio": audio_b64,
                "sample_rate": SAMPLE_RATE,
                "format": "wav",
            }
        )

    except Exception as e:
        logger.exception("OmniVoice synthesis failed")
        return JSONResponse({"error": str(e)}, status_code=500)


def main():
    port = int(os.getenv("WITCH_TTS_PORT"))
    uvicorn.run(api, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
