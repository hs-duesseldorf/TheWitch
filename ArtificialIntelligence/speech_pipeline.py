from __future__ import annotations

import asyncio
import logging
import os
import platform
import queue
import re
import threading
from typing import Any, Awaitable, Callable

import numpy as np
import soundfile as sf

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False
    sd = None

from shared.events import AnalysisStartedEvent, ErrorEvent, Scene
from .clients.tts_client import AudioFrame

logger = logging.getLogger(__name__)

SR = 48000
LINUX_VIRTUAL_CABLE_NAMES = ("witchvirtualcable",)

def _configured_audio_device(devices: list[dict[str, Any]]) -> int | None:
    configured = os.getenv("WITCH_AUDIO_DEVICE", "").strip()
    if not configured:
        return None

    if configured.isdigit():
        index = int(configured)
        if 0 <= index < len(devices) and devices[index]["max_output_channels"] > 0:
            return index
        logger.warning("WITCH_AUDIO_DEVICE=%s is not a valid output device index", configured)
        return None

    needle = configured.lower()
    for index, device in enumerate(devices):
        if device["max_output_channels"] > 0 and needle in device["name"].lower():
            return index

    logger.warning("WITCH_AUDIO_DEVICE=%r did not match any output device", configured)
    return None


def _linux_output_score(name: str) -> int:
    lowered = name.lower()
    if "monitor" in lowered or "null" in lowered:
        return -100
    if "built-in audio" in lowered and "analog stereo" in lowered:
        return 120
    if "analog stereo" in lowered:
        return 115
    if "pipewire" in lowered:
        return 100
    if "pulse" in lowered:
        return 90
    if lowered in {"default", "sysdefault"} or "default" in lowered:
        return 80
    if "hdmi" in lowered:
        return 20
    if "sof" in lowered or "alsa" in lowered:
        return 60
    return 10


def _is_virtual_cable(system: str, name: str) -> bool:
    lowered = name.lower()
    if system == "Windows":
        return "cable input" in lowered and "vb-audio" in lowered
    if system == "Darwin":
        return "blackhole" in lowered
    if system == "Linux":
        return any(cable_name in lowered for cable_name in LINUX_VIRTUAL_CABLE_NAMES)
    return False


def find_audio_devices() -> list[int | None]:
    if not HAS_SOUNDDEVICE:
        return []

    system = platform.system()
    selected: list[int | None] = []

    try:
        devices = list(sd.query_devices())
        configured = _configured_audio_device(devices)
        if configured is not None:
            return [configured]

        output_devices = [
            (index, device)
            for index, device in enumerate(devices)
            if device["max_output_channels"] > 0
        ]
        if output_devices:
            selected.append(None)

        for i, dev in output_devices:
            if _is_virtual_cable(system, dev["name"]):
                selected.append(i)

        if system == "Linux":
            if not selected:
                selected.extend(
                    index
                    for index, _ in sorted(
                        output_devices,
                        key=lambda item: _linux_output_score(item[1]["name"]),
                        reverse=True,
                    )
                    if _linux_output_score(devices[index]["name"]) >= 0
                )
        else:
            selected.extend(index for index, _ in output_devices)
    except Exception as e:
        logger.warning(f"Error finding devices: {e}")

    unique: list[int | None] = []
    for device in selected:
        if device not in unique:
            unique.append(device)
    return unique

AnalysisCallbacks = Callable[
    [AnalysisStartedEvent | ErrorEvent],
    Awaitable[None],
]
AudioCallbacks = Callable[[bytes], Awaitable[None]]


class StreamingAudioPlayer:
    def __init__(self, sample_rate: int = 24000):
        self._sample_rate = sample_rate
        self._pcm_remainder = b""
        self._outputs: list[dict[str, Any]] = []
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if not HAS_SOUNDDEVICE:
            logger.warning("sounddevice import failed")
            return
        if self._outputs:
            return

        devices = find_audio_devices()

        if not devices:
            logger.warning("No audio device found")
            return

        self._running = True

        for device in devices:
            try:
                output = {
                    "device": device,
                    "queue": queue.Queue(maxsize=0),
                    "pending": np.empty((0, 2), dtype=np.float32),
                }
                stream = sd.OutputStream(
                    samplerate=SR,
                    channels=2,
                    device=device,
                    dtype="float32",
                    callback=lambda outdata, frames, callback_time, status, output=output: self._callback(
                        output,
                        outdata,
                        frames,
                        callback_time,
                        status,
                    ),
                )
                output["stream"] = stream
                stream.start()
                self._outputs.append(output)
                if device is None:
                    logger.info("Started audio stream on OS default output device")
                else:
                    logger.info("Started audio stream on device %s", device)
            except Exception as e:
                label = "OS default output device" if device is None else f"device {device}"
                logger.warning("Failed to start audio stream on %s: %s", label, e)

        if not self._outputs:
            logger.warning("No audio output stream could be started")

    def is_active(self) -> bool:
        return len(self._outputs) > 0 and HAS_SOUNDDEVICE

    def _callback(
        self,
        output: dict[str, Any],
        outdata: np.ndarray,
        frames: int,
        time: Any,
        status: Any,
    ) -> None:
        outdata.fill(0)
        written = 0
        pending = output["pending"]
        chunks: queue.Queue[np.ndarray | None] = output["queue"]

        while written < frames:
            if pending.shape[0] == 0:
                try:
                    chunk = chunks.get_nowait()
                except queue.Empty:
                    break

                if chunk is None:
                    output["pending"] = np.empty((0, 2), dtype=np.float32)
                    return
                pending = chunk

            available = min(frames - written, pending.shape[0])
            outdata[written : written + available] = pending[:available]
            written += available

            if available < pending.shape[0]:
                pending = pending[available:]
            else:
                pending = np.empty((0, 2), dtype=np.float32)

        output["pending"] = pending

    def _enqueue(self, output: dict[str, Any], data: np.ndarray) -> None:
        chunks: queue.Queue[np.ndarray | None] = output["queue"]
        try:
            chunks.put_nowait(data.copy())
        except Exception:
            chunks.put(data.copy(), block=True)

    def put(self, audio_chunk: bytes, *, format: str = "pcm", sample_rate: int | None = None) -> None:
        try:
            if format == "wav":
                data, sample_rate = sf.read(
                    __import__("io").BytesIO(audio_chunk),
                    dtype="float32",
                )
            else:
                pcm = self._pcm_remainder + audio_chunk
                if len(pcm) < 2:
                    self._pcm_remainder = pcm
                    return
                if len(pcm) % 2:
                    self._pcm_remainder = pcm[-1:]
                    pcm = pcm[:-1]
                else:
                    self._pcm_remainder = b""
                data = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                sample_rate = sample_rate or self._sample_rate

            if sample_rate != SR:
                from scipy.signal import resample_poly

                data = resample_poly(data, SR, sample_rate)

            if len(data.shape) == 1:
                data = np.column_stack((data, data))

            if data.dtype != np.float32:
                data = data.astype(np.float32)
            if data.shape[1] > 2:
                data = data[:, :2]
            elif data.shape[1] == 1:
                data = np.column_stack((data[:, 0], data[:, 0]))

            for output in self._outputs:
                self._enqueue(output, data)
        except Exception as e:
            logger.warning("Audio playback chunk failed: %s", e)

    async def drain(self, timeout: float = 30.0) -> None:
        if not self._outputs:
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if all(
                output["queue"].empty() and output["pending"].shape[0] == 0
                for output in self._outputs
            ):
                await asyncio.sleep(0.1)
                return
            await asyncio.sleep(0.05)

        logger.warning("Audio drain timed out after %.1fs", timeout)

    def stop(self) -> None:
        self._running = False
        for output in self._outputs:
            try:
                output["queue"].put_nowait(None)
            except queue.Full:
                pass
        for output in self._outputs:
            stream = output["stream"]
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self._outputs.clear()
        self._pcm_remainder = b""


AnalysisDoneCallbacks = Callable[[str], Awaitable[None]]


class SpeechPipeline:
    def __init__(
        self,
        llm: Any,
        tts: Any,
        broadcast_callback: AnalysisCallbacks,
        audio_callback: AudioCallbacks | None = None,
        done_callback: AnalysisDoneCallbacks | None = None,
        tts_seed: int = 42,
    ):
        self._llm = llm
        self._tts = tts
        self._broadcast_callback = broadcast_callback
        self._audio_callback = audio_callback
        self._done_callback = done_callback
        self._current_task: asyncio.Task | None = None
        self._stage = "idle"
        self._tts_seed = tts_seed

    @property
    def stage(self) -> str:
        return self._stage

    def is_running(self) -> bool:
        return self._current_task is not None and not self._current_task.done()

    def stop_player(self) -> None:
        if getattr(self, '_player', None):
            logger.info("Stopping player")
            try:
                self._player.stop()
            except Exception:
                pass
        self._player = None
        logger.info("Player stopped and cleared")

    async def run_analysis(self, prompt: str, scene: Scene) -> None:
        if self._current_task is not None and not self._current_task.done():
            logger.debug("Analysis already running, ignoring")
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        self._stage = "queued"
        self._current_task = None
        self._current_task = loop.create_task(self._execute(prompt, scene))

    async def cancel(self) -> None:
        logger.info("cancel() called")
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self._current_task = None
        self._stage = "idle"
        self._player = None
        logger.info("Cancel complete")

    def cancel_sync(self) -> None:
        logger.info("cancel_sync() called")
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        logger.info("Cancel sync complete")

    INITIAL_RETRY_DELAY = 2.0
    MAX_RETRY_DELAY = 60.0

    async def _execute(self, prompt: str, scene: Scene) -> None:
        player = StreamingAudioPlayer()
        self._player = player
        delay = self.INITIAL_RETRY_DELAY

        try:
            while True:
                try:
                    self._stage = "llm"
                    await self._broadcast_callback(
                        AnalysisStartedEvent(scene=scene),
                    )

                    try:
                        player.start()
                        use_direct = player.is_active()
                        if use_direct:
                            logger.info("Audio playback active")
                    except Exception as e:
                        logger.warning("Audio player failed: %s", e)
                        use_direct = False

                    if not use_direct:
                        logger.info("No local audio device; using websocket audio only")

                    debug_text = (await self._llm.generate_fortune(prompt)).strip()
                    logger.info("Analysis LLM result: chars=%d", len(debug_text))
                    if not debug_text:
                        raise RuntimeError("LLM returned empty response")

                    if len(debug_text) < 10:
                        logger.warning("Text too short for TTS (%d chars), skipping", len(debug_text))
                        self._stage = "done"
                        if self._done_callback and debug_text:
                            await self._done_callback(debug_text)
                        return

                    self._stage = "tts_stream"
                    logger.info("TTS full: %s", debug_text[:80])
                    frame_count = 0
                    async for frame in self._tts.stream_synthesize(debug_text, seed=self._tts_seed):
                        frame_count += 1
                        if use_direct:
                            player.put(frame.audio, format=frame.format, sample_rate=frame.sample_rate)
                    logger.info("TTS complete: %d frames", frame_count)
                    if use_direct:
                        await player.drain()

                    self._stage = "done"
                    self._current_task = None
                    self._player = None
                    if self._done_callback and debug_text:
                        await self._done_callback(debug_text)
                    return

                except Exception as e:
                    logger.warning(f"Analysis failed, retrying in %ss: %s", delay, e)
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, self.MAX_RETRY_DELAY)
        finally:
            self._current_task = None
            self._stage = "idle"
            self._player = None
            try:
                player.stop()
            except Exception:
                pass
