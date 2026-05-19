from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from websockets.asyncio.client import connect

logger = logging.getLogger(__name__)

DEFAULT_QWEN_SAMPLE_RATE = 24_000
QWEN_WS_PATH = "/v1/audio/speech/stream"
QWEN_TASK_TYPE = "VoiceDesign"
QWEN_LANGUAGE = "German"
QWEN_DRAIN_TIMEOUT_SECONDS = 30.0
DEFAULT_VOICE = (
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
        voice: str | None = None,
    ):
        base_url = base_url.strip()
        if not base_url:
            raise ValueError("TTS base URL must not be empty")

        self.base_url = base_url.rstrip("/")
        self.voice = (voice or DEFAULT_VOICE).strip()

    async def stream_synthesize(
        self, text_chunks: AsyncIterable[str]
    ) -> AsyncIterator[AudioFrame]:
        url = self._websocket_url(QWEN_WS_PATH)
        logger.info("Qwen WebSocket TTS stream: base_url=%s ws_url=%s", self.base_url, url)

        retry_delay = 2.0
        attempt = 0
        while True:
            attempt += 1
            try:
                websocket = await connect(url, max_size=None)
                break
            except Exception as e:
                logger.warning("TTS server not ready (attempt %d): %s", attempt, e)
                await asyncio.sleep(retry_delay * min(attempt, 10))

        async with websocket:
            sender = asyncio.create_task(
                self._send_qwen_websocket_text(websocket, text_chunks)
            )
            try:
                while True:
                    timeout = QWEN_DRAIN_TIMEOUT_SECONDS if sender.done() else None
                    try:
                        message = await asyncio.wait_for(
                            websocket.recv(), timeout=timeout
                        )
                    except TimeoutError:
                        if sender.done():
                            break
                        continue

                    frame = self._audio_frame_from_ws_message(message)
                    if frame is not None:
                        yield frame

                    if (
                        isinstance(message, str)
                        and self._is_done_message(message)
                        and sender.done()
                    ):
                        break
            finally:
                if not sender.done():
                    sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)

    async def _send_qwen_websocket_text(
        self, websocket, text_chunks: AsyncIterable[str]
    ) -> None:
        async for text in text_chunks:
            text = text.strip()
            if not text:
                continue
            await websocket.send(
                json.dumps(self._qwen_payload(text, stream_audio=True))
            )

        await websocket.send(
            json.dumps(
                {
                    "input": "",
                    "text": "",
                    "end": True,
                    "is_final": True,
                    "stream_audio": True,
                    "response_format": "pcm",
                }
            )
        )

    def _qwen_payload(
        self, text: str, *, stream_audio: bool = True
    ) -> dict[str, object]:
        return {
            "input": text,
            "task_type": QWEN_TASK_TYPE,
            "language": QWEN_LANGUAGE,
            "instructions": self.voice,
            "stream": True,
            "stream_audio": stream_audio,
            "response_format": "pcm",
        }

    def _websocket_url(self, path: str) -> str:
        split = urlsplit(self.base_url)
        scheme = "wss" if split.scheme == "https" else "ws"
        return urlunsplit(
            (scheme, split.netloc, path if path.startswith("/") else f"/{path}", "", "")
        )

    def _audio_frame_from_ws_message(self, message: str | bytes) -> AudioFrame | None:
        if isinstance(message, bytes):
            return AudioFrame(
                audio=message,
                sample_rate=DEFAULT_QWEN_SAMPLE_RATE,
                format="pcm_s16le",
                mime_type="audio/L16;rate=24000;channels=1",
            )

        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return None

        audio = (
            data.get("audio")
            or data.get("audio_data")
            or data.get("pcm")
            or data.get("chunk")
            or data.get("data")
        )
        if not audio:
            return None

        if isinstance(audio, list):
            raw = bytes(audio)
        elif isinstance(audio, str):
            try:
                raw = base64.b64decode(audio)
            except Exception:
                return None
        else:
            return None

        return AudioFrame(
            audio=raw,
            sample_rate=int(data.get("sample_rate") or DEFAULT_QWEN_SAMPLE_RATE),
            format="pcm_s16le",
            mime_type="audio/L16;rate=24000;channels=1",
        )

    def _is_done_message(self, message: str) -> bool:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return False
        event = str(data.get("event") or data.get("type") or "").lower()
        return bool(
            data.get("done")
            or data.get("is_final")
            or event in {"done", "audio.done", "speech.done"}
        )
