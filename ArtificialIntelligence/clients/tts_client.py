from __future__ import annotations

import asyncio
import logging
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass
from urllib.parse import urljoin

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_QWEN_SAMPLE_RATE = 24_000
QWEN_SPEECH_PATH = "/v1/audio/speech"
PCM_SAMPLE_WIDTH = 2
PCM_CHUNK_SECONDS = 0.1
REQUEST_ATTEMPTS = 5
RETRY_DELAY_SECONDS = 1.0
HTTP_CONNECT_TIMEOUT_SECONDS = 10
HTTP_READ_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class AudioFrame:
    audio: bytes
    sample_rate: int
    format: str
    mime_type: str


class TTSClient:
    def __init__(self, base_url: str):
        base_url = base_url.strip()
        if not base_url:
            raise ValueError("TTS base URL must not be empty")

        self.base_url = base_url.rstrip("/")

    async def stream_synthesize(self, text: str) -> AsyncIterator[AudioFrame]:
        text = text.strip()
        if not text:
            return

        if len(text) < 5:
            logger.warning("Text too short for TTS (%d chars), skipping", len(text))
            return

        url = urljoin(self.base_url + "/", QWEN_SPEECH_PATH.lstrip("/"))
        logger.info(
            "TTS stream request: url=%s text_chars=%d",
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
                        json={"input": text},
                        timeout=aiohttp.ClientTimeout(
                            total=None,
                            connect=HTTP_CONNECT_TIMEOUT_SECONDS,
                            sock_connect=HTTP_CONNECT_TIMEOUT_SECONDS,
                            sock_read=HTTP_READ_TIMEOUT_SECONDS,
                        ),
                    ) as resp:
                        if resp.status >= 400:
                            body = await resp.text()
                            raise RuntimeError(
                                f"TTS stream failed: HTTP {resp.status}: {body}"
                            )
                        async for chunk in self._pcm_chunks(resp):
                            yielded_audio = True
                            yield AudioFrame(
                                audio=chunk,
                                sample_rate=DEFAULT_QWEN_SAMPLE_RATE,
                                format="pcm",
                                mime_type=resp.headers.get("content-type", "audio/pcm"),
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

    async def _pcm_chunks(self, resp: aiohttp.ClientResponse) -> AsyncIterator[bytes]:
        content_type = resp.headers.get("content-type", "").lower()
        if "wav" in content_type:
            audio = await resp.read()
            for chunk in self._decode_wav(audio):
                yield chunk
            return

        min_chunk_bytes = (
            int(DEFAULT_QWEN_SAMPLE_RATE * PCM_CHUNK_SECONDS) * PCM_SAMPLE_WIDTH
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
            if wav.getframerate() != DEFAULT_QWEN_SAMPLE_RATE:
                raise RuntimeError(f"Unsupported WAV sample rate: {wav.getframerate()}")
            pcm = wav.readframes(wav.getnframes())

        chunk_bytes = (
            int(DEFAULT_QWEN_SAMPLE_RATE * PCM_CHUNK_SECONDS) * PCM_SAMPLE_WIDTH
        )
        return [pcm[i : i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes)]
