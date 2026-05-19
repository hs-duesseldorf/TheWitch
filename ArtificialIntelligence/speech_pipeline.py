from __future__ import annotations

import asyncio
import copy
import logging
import os
import platform
import queue
import threading
from typing import Any, Awaitable, Callable

import httpx
import numpy as np

try:
    import sounddevice as sd

    HAS_SOUNDDEVICE = True
except (ImportError, OSError):
    HAS_SOUNDDEVICE = False
    sd = None

from shared.events import AnalysisStartedEvent, ErrorEvent, FortuneEvent, Scene

logger = logging.getLogger(__name__)

SR = 48000

_AUDIO_PLAY_HOST = os.getenv("WITCH_AUDIO_PLAY_HOST")
_AUDIO_BRIDGE_PORT = os.getenv("WITCH_AUDIO_BRIDGE_PORT", "10034")


async def _play_audio_http(audio: bytes) -> bool:
    if not _AUDIO_PLAY_HOST:
        return False
    url = f"http://{_AUDIO_PLAY_HOST}:{_AUDIO_BRIDGE_PORT}/play"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(url, files={"file": ("tts.wav", audio, "audio/wav")})
            return True
    except Exception as e:
        logger.warning(f"HTTP audio error: {e}")
        return False


def find_audio_devices() -> tuple[int | None, int | None]:
    if not HAS_SOUNDDEVICE:
        return None, None

    system = platform.system()
    virtual, default = None, None

    try:
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_output_channels"] <= 0:
                continue
            name = dev["name"].lower()

            if system == "Windows" and "cable input" in name and "vb-audio" in name:
                virtual = i
            elif system == "Linux" and ("virtual" in name or "null" in name):
                virtual = i
            elif system == "Darwin" and "blackhole" in name:
                virtual = i

        if virtual is None:
            default = sd.query_devices(kind="output")["index"]
    except Exception as e:
        logger.warning(f"Error finding devices: {e}")

    return virtual, default

AnalysisCallbacks = Callable[
    [AnalysisStartedEvent | FortuneEvent | ErrorEvent],
    Awaitable[None],
]


class StreamingAudioPlayer:
    def __init__(self, sample_rate: int = 24000):
        self._sample_rate = sample_rate
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=5)
        self._streams: list = []
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if not HAS_SOUNDDEVICE:
            logger.warning("sounddevice not available")
            return
        if self._streams:
            return

        virtual, default = find_audio_devices()

        if virtual is None and default is None:
            logger.warning("No audio device found")
            return

        self._running = True

        for device in (virtual, default):
            if device is None:
                continue
            try:
                stream = sd.OutputStream(
                    samplerate=SR,
                    channels=2,
                    device=device,
                    dtype="float32",
                    callback=self._callback,
                )
                stream.start()
                self._streams.append(stream)
                logger.info(f"Started stream on device {device}")
            except Exception as e:
                logger.warning(f"Failed to start stream on device {device}: {e}")

    def is_active(self) -> bool:
        return len(self._streams) > 0 and HAS_SOUNDDEVICE

    def _callback(
        self, outdata: np.ndarray, frames: int, time: Any, status: Any
    ) -> None:
        try:
            chunk = self._queue.get(timeout=0.1)
            if chunk is None:
                outdata.fill(0)
                return

            data = chunk.copy()

            if data.shape[0] < frames:
                outdata[: data.shape[0]] = data
                outdata[data.shape[0] :].fill(0)
            elif data.shape[0] > frames:
                outdata[:] = data[:frames]
            else:
                outdata[:] = data

        except queue.Empty:
            outdata.fill(0)

    def put(self, audio_chunk: bytes) -> None:
        try:
            data = (
                np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if self._sample_rate != SR:
                from scipy.signal import resample_poly

                data = resample_poly(data, SR, self._sample_rate)

            if len(data.shape) == 1:
                data = np.column_stack((data, data))

            self._queue.put(data, timeout=0.5)
        except queue.Full:
            logger.warning("Audio queue full, dropping chunk")

    def stop(self) -> None:
        self._running = False
        self._queue.put(None)
        for stream in self._streams:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self._streams.clear()


class SpeechPipeline:
    def __init__(
        self,
        llm: Any,
        tts: Any,
        broadcast_callback: AnalysisCallbacks,
    ):
        self._llm = llm
        self._tts = tts
        self._broadcast_callback = broadcast_callback
        self._current_task: asyncio.Task | None = None
        self._stage = "idle"

    @property
    def stage(self) -> str:
        return self._stage

    def is_running(self) -> bool:
        return self._current_task is not None and not self._current_task.done()

    async def run_analysis(self, prompt: str, scene: Scene) -> None:
        if self.is_running():
            logger.debug("Analysis already running, ignoring")
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        self._stage = "queued"
        self._current_task = loop.create_task(self._execute(prompt, scene))

    async def cancel(self) -> None:
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        self._stage = "idle"

    async def _execute(self, prompt: str, scene: Scene) -> None:
        player = StreamingAudioPlayer()

        try:
            self._stage = "llm"
            await self._broadcast_callback(
                AnalysisStartedEvent(scene=scene),
            )

            fortune_parts: list[str] = []

            async def fortune_chunks():
                async for chunk in self._llm.stream_fortune_chunks(prompt):
                    fortune_parts.append(chunk)
                    yield chunk

            self._stage = "tts_stream"
            audio_chunks: list[bytes] = []

            try:
                player.start()
                use_direct = player.is_active()
            except Exception:
                use_direct = False

            if use_direct:
                async for frame in self._tts.stream_synthesize(fortune_chunks()):
                    audio_chunks.append(frame.audio)
                    player.put(frame.audio)
            else:
                logger.info("No audio device, using HTTP bridge")
                async for frame in self._tts.stream_synthesize(fortune_chunks()):
                    audio_chunks.append(frame.audio)

            fortune = " ".join(
                part.strip() for part in fortune_parts if part.strip()
            ).strip()
            logger.info("Analysis stream result: chars=%d", len(fortune))

            if not use_direct and audio_chunks:
                wav_audio = self._pcm_to_wav(b"".join(audio_chunks), frame.sample_rate)
                if wav_audio:
                    await _play_audio_http(wav_audio)
            else:
                player.stop()

            await self._broadcast_callback(
                FortuneEvent(
                    text=fortune,
                    sample_rate=frame.sample_rate,
                ),
            )

            self._stage = "done"

        except httpx.HTTPError as e:
            self._stage = "error"
            logger.error("LLM/TTS request failed: %s", e)
        except Exception as e:
            self._stage = "error"
            logger.exception("Analysis error")
        finally:
            if not self._audio_play_host:
                player.stop()

    def _pcm_to_wav(self, audio: bytes, sample_rate: int) -> bytes:
        import wave

        output = __import__("io").BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio)
        return output.getvalue()
