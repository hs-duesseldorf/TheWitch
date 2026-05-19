from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from websockets.asyncio.client import connect

logger = logging.getLogger(__name__)

DEFAULT_QWEN_SAMPLE_RATE = 24_000
QWEN_WS_PATH = "/v1/audio/speech/stream"
DEFAULT_LANGUAGE = "German"
DEFAULT_VOICE = "vivian"
DEFAULT_INSTRUCTIONS = (
    "A young female speaker, around 20 to 30 years old, speaking German with natural prosody. "
    "Her voice has a subtle Asian accent, warm and realistic, never theatrical. "
    "She sounds like she comes from a remote mountain region: calm, sharp-minded, slightly enigmatic. "
    "Her delivery is witty, mysterious, composed, and quietly confident, with a hint of playful irony."
)


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
        language: str = DEFAULT_LANGUAGE,
        task_type: str | None = None,
    ):
        base_url = base_url.strip()
        if not base_url:
            raise ValueError("TTS base URL must not be empty")

        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.language = language.strip()
        self.task_type = (task_type or self._task_type_from_model(self.model)).strip()

        configured_voice = (voice or "").strip()
        if self.task_type == "CustomVoice" and self._looks_like_instructions(configured_voice):
            self.voice = DEFAULT_VOICE
            self.instructions = (instructions or configured_voice).strip()
        else:
            self.voice = (configured_voice or DEFAULT_VOICE).strip()
            self.instructions = (instructions or DEFAULT_INSTRUCTIONS).strip()

    async def stream_synthesize(
        self, text_chunks: AsyncIterable[str]
    ) -> AsyncIterator[AudioFrame]:
        url = self._websocket_url()
        logger.info("Qwen WebSocket TTS stream: url=%s", url)

        async with connect(url, max_size=None) as websocket:
            await websocket.send(json.dumps(self._session_config()))
            producer = asyncio.create_task(self._send_text_chunks(websocket, text_chunks))
            sample_rate = DEFAULT_QWEN_SAMPLE_RATE
            audio_format = "pcm"

            try:
                async for message in websocket:
                    if producer.done():
                        producer.result()

                    if isinstance(message, bytes):
                        yield AudioFrame(
                            audio=message,
                            sample_rate=sample_rate,
                            format=audio_format,
                            mime_type="audio/pcm",
                        )
                        continue

                    event = json.loads(message)
                    event_type = event.get("type")
                    if event_type == "audio.start":
                        sample_rate = int(event.get("sample_rate") or sample_rate)
                        audio_format = event.get("format") or audio_format
                    elif event_type == "session.done":
                        break
                    elif event_type == "error":
                        raise RuntimeError(event.get("message") or "TTS stream failed")
            finally:
                await producer

    def _session_config(self) -> dict[str, object]:
        config: dict[str, object] = {
            "type": "session.config",
            "model": self.model,
            "language": self.language,
            "response_format": "pcm",
            "task_type": self.task_type,
            "stream_audio": True,
            "split_granularity": "sentence",
        }

        if self.task_type == "CustomVoice":
            config["voice"] = self.voice
        if self.task_type in ("CustomVoice", "VoiceDesign") and self.instructions:
            config["instructions"] = self.instructions
        return config

    async def _send_text_chunks(self, websocket, text_chunks: AsyncIterable[str]) -> None:
        async for chunk in text_chunks:
            if not chunk:
                continue
            await websocket.send(json.dumps({"type": "input.text", "text": chunk}))
        await websocket.send(json.dumps({"type": "input.done"}))

    def _websocket_url(self) -> str:
        parsed = urlsplit(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return urlunsplit((scheme, parsed.netloc, QWEN_WS_PATH, "", ""))

    def _task_type_from_model(self, model: str) -> str:
        if model.endswith("-VoiceDesign"):
            return "VoiceDesign"
        if model.endswith("-Base"):
            return "Base"
        return "CustomVoice"

    def _looks_like_instructions(self, voice: str) -> bool:
        return " " in voice or "," in voice or "." in voice
