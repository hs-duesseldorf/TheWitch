from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from TTS.api import TTS

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
    language: str = "en"
    speed: float = 1.0


tts_engine = None


def get_tts():
    global tts_engine
    if tts_engine is None:
        logger.debug("Loading TTS model: %s", MODEL_NAME)
        use_gpu = torch.cuda.is_available()
        logger.debug("TTS CUDA available: %s", use_gpu)
        tts_engine = TTS(model_name=MODEL_NAME)
        if use_gpu:
            tts_engine.to("cuda")
        logger.debug("TTS model loaded")
    return tts_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


api = FastAPI(title="The Witch TTS Server", lifespan=lifespan)


@api.post("/api/generate")
async def synthesize(req: TTSRequest):
    try:
        engine = get_tts()

        language = (req.language or "de").strip().lower()
        output_path = OUTPUT_DIR / f"output_{id(req)}.wav"

        logger.debug("Synthesizing: lang=%s text=%d chars", language, len(req.text))

        tts_kwargs = {
            "text": req.text,
            "file_path": str(output_path),
        }
        # Only add language if the model is multilingual
        if "multilingual" in MODEL_NAME:
            tts_kwargs["language"] = language

        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: engine.tts_to_file(**tts_kwargs),
        )

        data, samplerate = sf.read(output_path)
        output_path.unlink(missing_ok=True)

        buffer = io.BytesIO()
        sf.write(buffer, data, samplerate, format="WAV")
        audio_b64 = base64.b64encode(buffer.getvalue()).decode()

        return JSONResponse(
            {
                "audio": audio_b64,
                "sample_rate": samplerate,
                "format": "wav",
            }
        )

    except Exception as e:
        logger.exception("TTS synthesis failed")
        return JSONResponse({"error": str(e)}, status_code=500)


def main():
    port = int(os.getenv("WITCH_TTS_PORT"))
    uvicorn.run(api, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
