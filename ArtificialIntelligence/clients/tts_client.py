from __future__ import annotations

import asyncio
import logging
import os
import wave
import re
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
SENTENCE_STREAM_MAX_CHARS = 260


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
        # attribute kept from old version to not break stuff
        self.instructions = ""
        
        # In this client, WITCH_TTS_STREAM means "stream sentence-sized audio
        # chunks through the app". Fish native streaming is intentionally kept
        # disabled in _speech_request(), because it does not produce usable PCM
        # chunks early enough for this pipeline
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
        
        # If app-level streaming is enabled, split long replies into smaller
        # Fish requests. Each Fish request returns one complete WAV, which we
        # decode to PCM immediately and yield before requesting/playing later
        # sentences. If streaming is disabled, use one request for the whole text.
        text_chunks = self._split_tts_text(text) if self.stream else [text]
        text_chunks = [chunk for chunk in text_chunks if chunk.strip()]
        if not text_chunks:
            return

        logger.info(
            "Fish sentence TTS %s: text_chars=%d chunks=%d",
            "stream" if self.stream else "full",
            len(text),
            len(text_chunks),
        )

        for index, chunk_text in enumerate(text_chunks, start=1):
            logger.info(
                "Fish sentence TTS chunk %d/%d: chars=%d",
                index,
                len(text_chunks),
                len(chunk_text),
            )

            async for frame in self._request_frames(chunk_text, seed=seed):
                yield frame
                
    def _split_tts_text(self, text: str) -> list[str]:
        text = " ".join(text.strip().split())
        if not text:
            return []

        parts = re.split(r"(?<=[.!?…])\s+", text)
        chunks: list[str] = []
        current = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # If one sentence is very long, split it further at commas/semicolons.
            subparts = [part]
            if len(part) > SENTENCE_STREAM_MAX_CHARS:
                subparts = [p.strip() for p in re.split(r"(?<=[,;:])\s+", part) if p.strip()]

            for subpart in subparts:
                if current and len(current) + 1 + len(subpart) > SENTENCE_STREAM_MAX_CHARS:
                    chunks.append(current.strip())
                    current = subpart
                else:
                    current = f"{current} {subpart}".strip()

        if current:
            chunks.append(current.strip())

        return chunks
    
    async def _request_frames(
        self, text: str, *, seed: int | None = None
    ) -> AsyncIterator[AudioFrame]:
        url = urljoin(self.base_url + "/", FISH_SPEECH_PATH.lstrip("/"))

        attempt = 0
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
                        content_type = resp.headers.get("content-type", "audio/wav")

                        if resp.status >= 400:
                            body = await resp.text()
                            raise RuntimeError(
                                f"TTS request failed: HTTP {resp.status}: {body}"
                            )

                        audio = await resp.read()
                        if not audio:
                            raise RuntimeError("TTS request returned empty audio")

                        if "wav" in content_type.lower() or audio.startswith(b"RIFF"):
                            for pcm in self._decode_wav(audio):
                                yield AudioFrame(
                                    audio=pcm,
                                    sample_rate=DEFAULT_TTS_SAMPLE_RATE,
                                    format="pcm",
                                    mime_type=content_type,
                                )
                        else:
                            yield AudioFrame(
                                audio=audio,
                                sample_rate=DEFAULT_TTS_SAMPLE_RATE,
                                format="pcm",
                                mime_type=content_type,
                            )

                    return

            except Exception as exc:
                if attempt >= REQUEST_ATTEMPTS:
                    logger.error(
                        "TTS sentence failed after %d attempts: %s",
                        attempt,
                        exc,
                    )
                    raise

                logger.warning(
                    "TTS sentence failed on attempt %d; retrying: %s",
                    attempt,
                    exc,
                )
                await asyncio.sleep(RETRY_DELAY_SECONDS * min(attempt, 10))

    def _speech_request(
        self, text: str, *, seed: int | None = None
    ) -> dict[str, object]:
        request: dict[str, object] = {
            "text": text,
            "format": "wav",
            "streaming": False, 
            # native streaming is disabled due to not getting it to work correctly with our setup and how fish speech works
            # instead we're doing sentence splitting for now
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

        min_chunk_bytes = (
            int(DEFAULT_TTS_SAMPLE_RATE * PCM_CHUNK_SECONDS) * PCM_SAMPLE_WIDTH
        )

        if "wav" in content_type or self.stream:
            buffer = bytearray()
            in_data = False

            async for chunk in resp.content.iter_any():
                if not chunk:
                    continue

                buffer.extend(chunk)

                if not in_data:
                    riff_pos = buffer.find(b"RIFF")
                    if riff_pos < 0:
                        # Fish may not start exactly at byte 0; keep a little search window.
                        if len(buffer) > 4096:
                            del buffer[:-4]
                        continue

                    if riff_pos > 0:
                        del buffer[:riff_pos]

                    if len(buffer) < 12:
                        continue

                    if buffer[0:4] != b"RIFF" or buffer[8:12] != b"WAVE":
                        raise RuntimeError("Invalid WAV stream header from TTS")

                    offset = 12
                    while True:
                        if len(buffer) < offset + 8:
                            break

                        chunk_id = bytes(buffer[offset : offset + 4])
                        chunk_size = int.from_bytes(
                            buffer[offset + 4 : offset + 8],
                            "little",
                        )
                        chunk_data_start = offset + 8
                        chunk_data_end = chunk_data_start + chunk_size
                        padded_end = chunk_data_end + (chunk_size % 2)

                        if chunk_id == b"fmt ":
                            if len(buffer) < chunk_data_end:
                                break

                            channels = int.from_bytes(
                                buffer[chunk_data_start + 2 : chunk_data_start + 4],
                                "little",
                            )
                            sample_rate = int.from_bytes(
                                buffer[chunk_data_start + 4 : chunk_data_start + 8],
                                "little",
                            )
                            sample_width = int.from_bytes(
                                buffer[chunk_data_start + 14 : chunk_data_start + 16],
                                "little",
                            ) // 8

                            if channels != 1:
                                raise RuntimeError(
                                    f"Unsupported streamed WAV channel count: {channels}"
                                )
                            if sample_rate != DEFAULT_TTS_SAMPLE_RATE:
                                raise RuntimeError(
                                    f"Unsupported streamed WAV sample rate: {sample_rate}"
                                )
                            if sample_width != PCM_SAMPLE_WIDTH:
                                raise RuntimeError(
                                    f"Unsupported streamed WAV sample width: {sample_width}"
                                )

                            offset = padded_end
                            continue

                        if chunk_id == b"data":
                            del buffer[:chunk_data_start]
                            in_data = True
                            break

                        if len(buffer) < padded_end:
                            break

                        offset = padded_end

                    if not in_data:
                        continue

                emit_len = (len(buffer) // min_chunk_bytes) * min_chunk_bytes
                emit_len -= emit_len % PCM_SAMPLE_WIDTH

                if emit_len > 0:
                    yield bytes(buffer[:emit_len])
                    del buffer[:emit_len]

            if buffer:
                if len(buffer) % PCM_SAMPLE_WIDTH:
                    logger.warning("Dropping trailing partial PCM sample from TTS stream")
                    buffer = buffer[: -(len(buffer) % PCM_SAMPLE_WIDTH)]
                if buffer:
                    yield bytes(buffer)

            return

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
