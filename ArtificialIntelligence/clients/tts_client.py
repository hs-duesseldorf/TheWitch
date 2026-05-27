from __future__ import annotations

import asyncio
import logging
import os
import wave
import ormsgpack
from collections.abc import AsyncIterator
from dataclasses import dataclass
from urllib.parse import urljoin

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_TTS_SAMPLE_RATE = int(os.getenv("WITCH_TTS_SAMPLE_RATE", "44100"))
FISH_SPEECH_PATH = "/v1/tts"
PCM_SAMPLE_WIDTH = 2
PCM_CHUNK_SECONDS = 0.1
DEFAULT_VOICE = "witch"

REQUEST_ATTEMPTS = 5
RETRY_DELAY_SECONDS = 1.0


@dataclass(frozen=True)
class AudioFrame:
    audio: bytes
    sample_rate: int
    format: str
    mime_type: str


class TTSClient:
    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        voice: str | None = None,
        instructions: str | None = None,
        task_type: str | None = None,
    ):
        from pathlib import Path
        
        base_url = base_url.strip()
        if not base_url:
            raise ValueError("TTS base URL must not be empty")

        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.task_type = (task_type or "CustomVoice").strip()

        configured_voice = (voice or "").strip()
        self.voice = (configured_voice or DEFAULT_VOICE).strip()
        self.instructions = ""
        
        stream_setting = os.getenv("WITCH_TTS_STREAM", "").strip().lower()
        self.stream = stream_setting not in {"0", "false", "no", "off"}
        
        voice_root = (
            Path(__file__).resolve().parents[1]
            / "servers"
            / "tts"
            / "voice_refs"
            / self.voice
        )

        self.voice_audio_path = voice_root / "reference.wav"
        transcript_path = voice_root / "reference.lab"

        if not self.voice_audio_path.exists():
            raise RuntimeError(
                f"Missing voice reference audio: {self.voice_audio_path}"
            )

        if not transcript_path.exists():
            raise RuntimeError(
                f"Missing voice reference transcript: {transcript_path}"
            )

        self.voice_transcript = transcript_path.read_text(
            encoding="utf-8"
        ).strip()

        self.voice_audio = self.voice_audio_path.read_bytes()

        logger.info(
            "Loaded Fish voice reference '%s' from %s",
            self.voice,
            voice_root,
        )

    async def stream_synthesize(
        self, text: str, *, seed: int | None = None
    ) -> AsyncIterator[AudioFrame]:
        text = text.strip()
        if not text:
            return

        if len(text) < 5:
            logger.warning("Text too short for TTS (%d chars), skipping", len(text))
            return

        url = urljoin(self.base_url + "/", FISH_SPEECH_PATH.lstrip("/"))
        logger.info(
            "Fish HTTP TTS %s: url=%s text_chars=%d",
            "stream" if self.stream else "full",
            url,
            len(text),
        )

        attempt = 0
        yielded_audio = False
        while True:
            attempt += 1
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        data=ormsgpack.packb(self._speech_request(text, seed=seed)),
                        headers={"content-type": "application/msgpack"},
                        timeout=aiohttp.ClientTimeout(total=None),
                    ) as resp:
                        if resp.status >= 400:
                            body = await resp.text()
                            raise RuntimeError(
                                f"TTS stream failed: HTTP {resp.status}: {body}"
                            )
                        if self.stream:
                            async for chunk in self._pcm_chunks(resp):
                                yielded_audio = True
                                yield AudioFrame(
                                    audio=chunk,
                                    sample_rate=DEFAULT_TTS_SAMPLE_RATE,
                                    format="pcm",
                                    mime_type=resp.headers.get("content-type", "audio/pcm"),
                                )
                        else:
                            audio = await resp.read()
                            content_type = resp.headers.get("content-type", "audio/pcm")
                            if "wav" in content_type.lower():
                                for chunk in self._decode_wav(audio):
                                    yielded_audio = True
                                    yield AudioFrame(
                                        audio=chunk,
                                        sample_rate=DEFAULT_TTS_SAMPLE_RATE,
                                        format="pcm",
                                        mime_type=content_type,
                                    )
                            elif audio:
                                yielded_audio = True
                                yield AudioFrame(
                                    audio=audio,
                                    sample_rate=DEFAULT_TTS_SAMPLE_RATE,
                                    format="pcm",
                                    mime_type=content_type,
                                )
                    return
            except Exception as e:
                if yielded_audio:
                    logger.warning("TTS stream failed after audio started; ending partial stream: %s", e)
                    return
                if attempt >= REQUEST_ATTEMPTS:
                    logger.error("TTS stream failed after %d attempts: %s", attempt, e)
                    raise
                logger.warning(
                    "TTS stream failed on attempt %d; retrying: %s", attempt, e
                )
                await asyncio.sleep(RETRY_DELAY_SECONDS * min(attempt, 10))

    def _speech_request(
        self, text: str, *, seed: int | None = None
    ) -> dict[str, object]:
        request: dict[str, object] = {
            "text": text,
            "format": "pcm",
            "streaming": self.stream,
            "latency": "balanced",
            "max_new_tokens": 1024,
            "chunk_length": 300,
            "top_p": 0.8,
            "repetition_penalty": 1.1,
            "temperature": 0.6,
            "use_memory_cache": "off",
            "normalize": True,
            "references": [
                {
                    "audio": self.voice_audio,
                    "text": self.voice_transcript,
                }
            ],
        }

        if seed is not None:
            request["seed"] = seed
        return request

    async def _pcm_chunks(self, resp: aiohttp.ClientResponse) -> AsyncIterator[bytes]:
        content_type = resp.headers.get("content-type", "").lower()
        if "wav" in content_type:
            audio = await resp.read()
            for chunk in self._decode_wav(audio):
                yield chunk
            return

        min_chunk_bytes = (
            int(DEFAULT_TTS_SAMPLE_RATE * PCM_CHUNK_SECONDS) * PCM_SAMPLE_WIDTH
        )
        buffer = bytearray()
        async for chunk in resp.content.iter_any():
            if not chunk:
                continue
            buffer.extend(chunk)
            emit_len = (len(buffer) // min_chunk_bytes) * min_chunk_bytes
            emit_len -= emit_len % PCM_SAMPLE_WIDTH
            if emit_len <= 0:
                continue
            yield bytes(buffer[:emit_len])
            del buffer[:emit_len]

        if buffer:
            if len(buffer) % PCM_SAMPLE_WIDTH:
                logger.warning("Dropping trailing partial PCM sample from TTS stream")
                buffer = buffer[:-1]
            if buffer:
                yield bytes(buffer)

    def _decode_wav(self, audio: bytes) -> list[bytes]:
        import io

        with wave.open(io.BytesIO(audio), "rb") as wav:
            if wav.getsampwidth() != PCM_SAMPLE_WIDTH:
                raise RuntimeError(
                    f"Unsupported WAV sample width: {wav.getsampwidth()}"
                )
            if wav.getnchannels() != 1:
                raise RuntimeError(
                    f"Unsupported WAV channel count: {wav.getnchannels()}"
                )
            if wav.getframerate() != DEFAULT_TTS_SAMPLE_RATE:
                raise RuntimeError(f"Unsupported WAV sample rate: {wav.getframerate()}")
            pcm = wav.readframes(wav.getnframes())

        chunk_bytes = (
            int(DEFAULT_TTS_SAMPLE_RATE * PCM_CHUNK_SECONDS) * PCM_SAMPLE_WIDTH
        )
        return [pcm[i : i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes)]

    def _task_type_from_model(self, model: str) -> str:
        if model.endswith("-VoiceDesign"):
            return "VoiceDesign"
        if model.endswith("-Base"):
            return "Base"
        return "CustomVoice"
