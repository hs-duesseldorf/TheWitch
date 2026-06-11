from __future__ import annotations

import asyncio
import logging
import os
import platform
import queue
import re
import threading
import time
import unicodedata
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import numpy as np
import soundfile as sf

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False
    sd = None

logger = logging.getLogger(__name__)

SR = 48000
TTS_INPUT_SAMPLE_RATE = 24000
UNDERRUN_LOG_INTERVAL_SECONDS = 2.0


@dataclass(frozen=True)
class _QueuedAudio:
    generation: int
    data: np.ndarray


@dataclass
class _AudioOutput:
    device: int | None
    is_virtual_cable: bool
    queue: queue.SimpleQueue[_QueuedAudio]
    pending: np.ndarray
    buffered_frames: int
    playback_started: bool
    producer_done: bool
    underruns: int
    last_underrun_log: float
    generation: int
    lock: threading.Lock
    stream: Any | None = None
    submitted_frames: int = 0
    playback_started_at: float | None = None
    playback_complete_at: float | None = None


def normalize_llm_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    # Handle literal escape sequences from LLM output
    text = text.replace(r"\n", " ").replace(r"\r", " ").replace(r"\t", " ")
    text = text.replace(r"\\", " ")

    # Strip markdown formatting
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    text = (
        text.replace("&", " und ")
        .replace("%", " Prozent ")
        .replace("€", " Euro ")
        .replace("/", " ")
        .replace("\\", " ")
    )

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*\u2022]+\s*", "", line)
        line = re.sub(r"^\d+[.)]\s*", "", line)
        lines.append(line)

    normalized = " ".join(lines) if lines else text
    normalized = re.sub(r"[\[\](){}<>]", " ", normalized)
    normalized = re.sub(r"[*_#`~|^@=+]", " ", normalized)
    normalized = re.sub(r"[\"'“”„‘’«»]", "", normalized)
    normalized = re.sub(r"[^0-9A-Za-zÄÖÜäöüß.,!?;:\-\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\s+([.,!?;:])", r"\1", normalized)
    normalized = re.sub(r"([.,!?;:]){2,}", r"\1", normalized)
    if normalized and normalized[-1] not in ".!?":
        normalized = f"{normalized}."
    return normalized


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
    needle = os.environ["WITCH_VIRTUAL_CABLE_NAME"].strip()
    return bool(needle) and needle.lower() in name.lower()


def find_audio_devices() -> list[int | None]:
    if not HAS_SOUNDDEVICE:
        return []

    system = platform.system()
    selected: list[int | None] = []

    try:
        devices = list(sd.query_devices())
        output_devices = [
            (index, device)
            for index, device in enumerate(devices)
            if device["max_output_channels"] > 0
        ]

        if output_devices:
            selected.append(None)
            
        if system == "Linux":
            for i, dev in output_devices:
                if _is_virtual_cable(system, dev["name"]):
                    selected.append(i)

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
            best_match = None
            for i, dev in output_devices:
                if _is_virtual_cable(system, dev["name"]):
                    hostapi = sd.query_hostapis(dev["hostapi"])["name"]

                    if "WASAPI" in hostapi:
                        best_match = i
                        break

                    if best_match is None:
                        best_match = i

            if best_match is not None:
                selected.append(best_match)
                
    except Exception as e:
        logger.warning(f"Error finding devices: {e}")

    unique: list[int | None] = []
    for device in selected:
        if device not in unique:
            unique.append(device)
    return unique


class StreamingAudioPlayer:
    def __init__(self, *, prebuffer_seconds: float, speaker_delay_seconds: float):
        self._input_sample_rate = TTS_INPUT_SAMPLE_RATE
        self._output_sample_rate = SR
        self._prebuffer_seconds = max(0.0, prebuffer_seconds)
        self._prebuffer_frames = int(self._output_sample_rate * self._prebuffer_seconds)
        self._speaker_delay_seconds = max(0.0, speaker_delay_seconds)
        self._speaker_delay_frames = int(self._output_sample_rate * self._speaker_delay_seconds)
        self._pcm_remainder = b""
        self._source_queue: queue.SimpleQueue[_QueuedAudio] = queue.SimpleQueue()
        self._source_generation = 0
        self._source_lock = threading.Lock()
        self._outputs: list[_AudioOutput] = []
        self._playback_start_callback: Callable[[], None] | None = None
        self._playback_start_notified = False

    def start(self) -> None:
        if not HAS_SOUNDDEVICE:
            logger.warning("sounddevice import failed")
            return
        if self._outputs:
            return

        devices = find_audio_devices()
        logger.info("Audio devices selected: %s", devices)

        if not devices:
            logger.warning("No audio device found")
            return

        speaker_started = False
        for device in devices:
            try:
                is_virtual_cable = False

                if device is not None:
                    try:
                        device_info = sd.query_devices(device)
                        is_virtual_cable = _is_virtual_cable(platform.system(), device_info["name"])
                    except Exception:
                        pass

                if speaker_started and not is_virtual_cable:
                    continue

                output = _AudioOutput(
                    device=device,
                    is_virtual_cable=is_virtual_cable,
                    queue=queue.SimpleQueue(),
                    pending=np.empty((0, 2), dtype=np.float32),
                    buffered_frames=0,
                    playback_started=self._prebuffer_frames <= 0,
                    producer_done=False,
                    underruns=0,
                    last_underrun_log=0.0,
                    generation=0,
                    lock=threading.Lock(),
                    playback_complete_at=None,
                )
                stream = sd.OutputStream(
                    samplerate=self._output_sample_rate,
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
                output.stream = stream
                stream.start()
                self._outputs.append(output)
                if not is_virtual_cable:
                    speaker_started = True
                if device is None:
                    logger.info(
                        "Started audio stream on OS default output device; prebuffer=%.2fs",
                        self._prebuffer_seconds,
                    )
                else:
                    logger.info(
                        "Started audio stream on device %s; prebuffer=%.2fs",
                        device,
                        self._prebuffer_seconds,
                    )
            except Exception as e:
                label = "OS default output device" if device is None else f"device {device}"
                logger.warning("Failed to start audio stream on %s: %s", label, e)

        if not self._outputs:
            raise RuntimeError("No audio output stream could be started")

    def is_active(self) -> bool:
        return len(self._outputs) > 0 and HAS_SOUNDDEVICE

    @property
    def prebuffer_seconds(self) -> float:
        return self._prebuffer_seconds

    def begin(self, on_playback_start: Callable[[], None] | None = None) -> None:
        self.start()
        self._playback_start_callback = on_playback_start
        self._playback_start_notified = False
        with self._source_lock:
            self._source_generation += 1
            generation = self._source_generation
        self._clear_source_queue()
        for output in self._outputs:
            self._clear_output(output, producer_done=False, generation=generation)
            
        if self._speaker_delay_frames > 0:
            silence = np.zeros((self._speaker_delay_frames, 2), dtype=np.float32)

            for output in self._outputs:
                if not output.is_virtual_cable:
                    self._enqueue_output(output, silence, generation=generation)

    def _callback(
        self,
        output: _AudioOutput,
        outdata: np.ndarray,
        frames: int,
        callback_time: Any,
        status: Any,
    ) -> None:
        outdata.fill(0)
        written = 0
        
        with output.lock:
            generation = output.generation
            pending = output.pending
        chunks = output.queue

        while written < frames:
            with output.lock:
                if generation != output.generation:
                    pending = np.empty((0, 2), dtype=np.float32)
                    break
                playback_started = output.playback_started
                buffered_frames = output.buffered_frames
                producer_done = output.producer_done
                if not playback_started:
                    if buffered_frames >= self._prebuffer_frames or producer_done:
                        output.playback_started = True
                        playback_started = True

            if not playback_started:
                break

            if pending.shape[0] == 0:
                try:
                    item = chunks.get_nowait()
                except queue.Empty:
                    with output.lock:
                        producer_done = output.producer_done
                    if producer_done:
                        break
                    with output.lock:
                        output.underruns += 1
                        underruns = output.underruns
                        now = time.perf_counter()
                        should_log = (
                            now - output.last_underrun_log
                            >= UNDERRUN_LOG_INTERVAL_SECONDS
                        )
                        if should_log:
                            output.last_underrun_log = now
                    if should_log:
                        device = output.device
                        label = "OS default output device" if device is None else f"device {device}"
                        message = f"WARNING: Audio underrun detected on {label} (count={underruns})"
                        print(message, flush=True)
                        logger.warning(message)
                    break
                if item.generation != generation:
                    continue
                chunk = item.data
                pending = chunk

            with output.lock:
                if generation != output.generation:
                    pending = np.empty((0, 2), dtype=np.float32)
                    break

            available = min(frames - written, pending.shape[0])
            outdata[written : written + available] = pending[:available]
            written += available
            with output.lock:
                output.buffered_frames = max(0, output.buffered_frames - available)
                if output.playback_started_at is None:
                    output.playback_started_at = time.perf_counter()
                output.submitted_frames += available

            if available < pending.shape[0]:
                pending = pending[available:]
            else:
                pending = np.empty((0, 2), dtype=np.float32)

        if written > 0:
            self._notify_playback_started()

        with output.lock:
            if generation == output.generation:
                output.pending = pending
                if written > 0:
                    output.playback_complete_at = self._playback_complete_at(
                        output,
                        callback_time,
                        written,
                    )

    def _playback_complete_at(
        self,
        output: _AudioOutput,
        callback_time: Any,
        written: int,
    ) -> float:
        now = time.perf_counter()
        try:
            device_delay = max(
                0.0,
                float(callback_time.outputBufferDacTime)
                - float(callback_time.currentTime),
            )
        except (AttributeError, TypeError, ValueError):
            try:
                device_delay = max(0.0, float(output.stream.latency))
            except (AttributeError, TypeError, ValueError):
                device_delay = 0.0
        return now + device_delay + (written / self._output_sample_rate)

    def _notify_playback_started(self) -> None:
        if self._playback_start_notified:
            return
        callback = self._playback_start_callback
        self._playback_start_notified = True
        if callback is not None:
            callback()

    def _drain_source_queue(self) -> None:
        while True:
            try:
                item = self._source_queue.get_nowait()
            except queue.Empty:
                return
            with self._source_lock:
                current_generation = self._source_generation
            if item.generation != current_generation:
                continue
            for output in self._outputs:
                self._enqueue_output(output, item.data, generation=item.generation)

    def _enqueue_source(self, data: np.ndarray) -> None:
        with self._source_lock:
            generation = self._source_generation
        self._source_queue.put(_QueuedAudio(generation=generation, data=data.copy()))
        self._drain_source_queue()

    def _enqueue_output(self, output: _AudioOutput, data: np.ndarray, *, generation: int) -> None:
        chunks = output.queue
        with output.lock:
            if generation != output.generation:
                return
            output.buffered_frames += int(data.shape[0])
            chunks.put(_QueuedAudio(generation=generation, data=data.copy()))

    def _convert_chunk(
        self,
        audio_chunk: bytes,
        *,
        format: str,
        sample_rate: int | None,
    ) -> np.ndarray | None:
        if format == "wav":
            data, sample_rate = sf.read(
                __import__("io").BytesIO(audio_chunk),
                dtype="float32",
            )
        else:
            pcm = self._pcm_remainder + audio_chunk
            if len(pcm) < 2:
                self._pcm_remainder = pcm
                return None
            if len(pcm) % 2:
                self._pcm_remainder = pcm[-1:]
                pcm = pcm[:-1]
            else:
                self._pcm_remainder = b""
            data = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            sample_rate = sample_rate or self._input_sample_rate

        if sample_rate != self._output_sample_rate:
            from scipy.signal import resample_poly

            data = resample_poly(data, self._output_sample_rate, sample_rate)

        if len(data.shape) == 1:
            data = np.column_stack((data, data))

        if data.dtype != np.float32:
            data = data.astype(np.float32)
        if data.shape[1] > 2:
            data = data[:, :2]
        elif data.shape[1] == 1:
            data = np.column_stack((data[:, 0], data[:, 0]))

        return data

    def put(self, audio_chunk: bytes, *, format: str = "pcm", sample_rate: int | None = None) -> None:
        try:
            data = self._convert_chunk(audio_chunk, format=format, sample_rate=sample_rate)
            if data is not None:
                self._enqueue_source(data)
        except Exception as e:
            logger.warning(
                "Audio playback chunk failed (format=%s, sample_rate=%s, len=%d): %s",
                format,
                sample_rate,
                len(audio_chunk),
                e,
            )

    def finish(self) -> None:
        self._drain_source_queue()
        for output in self._outputs:
            with output.lock:
                output.producer_done = True

    async def wait_for_playback(self) -> None:
        """Wait until queued and device-buffered audio has finished playing."""
        if not self._outputs:
            return

        while True:
            now = time.perf_counter()
            all_finished = True
            for output in self._outputs:
                with output.lock:
                    output_finished = (
                        output.producer_done
                        and output.buffered_frames <= 0
                        and output.pending.shape[0] == 0
                        and output.queue.empty()
                        and (
                            output.playback_complete_at is None
                            or now >= output.playback_complete_at
                        )
                    )
                if not output_finished:
                    all_finished = False
                    break

            if all_finished:
                return

            await asyncio.sleep(0.02)

    def _clear_source_queue(self) -> None:
        while True:
            try:
                self._source_queue.get_nowait()
            except queue.Empty:
                break

    def _clear_output(self, output: _AudioOutput, *, producer_done: bool, generation: int) -> None:
        with output.lock:
            output.generation = generation
            output.producer_done = producer_done
            output.playback_started = (
                self._prebuffer_frames <= 0 if not producer_done else True
            )
            output.buffered_frames = 0
            output.pending = np.empty((0, 2), dtype=np.float32)
            output.submitted_frames = 0
            output.playback_started_at = None
            output.playback_complete_at = None
        chunks = output.queue
        while True:
            try:
                chunks.get_nowait()
            except queue.Empty:
                break

    def clear(self) -> None:
        with self._source_lock:
            self._source_generation += 1
            generation = self._source_generation
        self._clear_source_queue()
        for output in self._outputs:
            self._clear_output(output, producer_done=True, generation=generation)
        self._pcm_remainder = b""
        self._playback_start_callback = None
        self._playback_start_notified = False


class SpeechPipeline:
    def __init__(
        self,
        llm: Any,
        tts: Any,
        *,
        prebuffer_seconds: float,
        speaker_delay_seconds: float,
    ):
        self._llm = llm
        self._tts = tts
        self._player = StreamingAudioPlayer(
            prebuffer_seconds=prebuffer_seconds,
            speaker_delay_seconds=speaker_delay_seconds,
        )
        self._stage = "idle"
        self._player_started = False

    @property
    def stage(self) -> str:
        return self._stage

    def stop_player(self) -> None:
        logger.info("Clearing player")
        with suppress(Exception):
            self._player.clear()
        logger.info("Player cleared")

    INITIAL_RETRY_DELAY = 2.0
    MAX_RETRY_DELAY = 60.0

    async def _generate_text(self, prompt: str) -> str:
        async_generate = getattr(self._llm, "async_generate_text", None)
        if async_generate is not None:
            return await async_generate(prompt)
        generate = getattr(self._llm, "generate_text", None)
        if generate is not None:
            return await asyncio.to_thread(generate, prompt)
        return await asyncio.to_thread(self._llm.generate_fortune, prompt)

    async def generate_text(self, prompt: str) -> str:
        delay = self.INITIAL_RETRY_DELAY

        try:
            while True:
                try:
                    self._stage = "llm"
                    text = (await self._generate_text(prompt)).strip()
                    logger.info("Speech LLM result: chars=%d", len(text))
                    if not text:
                        raise RuntimeError("LLM returned empty response")
                    self._stage = "done"
                    return text

                except asyncio.CancelledError:
                    logger.info("Speech text generation cancelled at stage=%s", self._stage)
                    raise
                except Exception as e:
                    logger.warning("Speech text generation failed, retrying in %ss: %s", delay, e)
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, self.MAX_RETRY_DELAY)
        finally:
            self._stage = "idle"

    async def play_text(
        self,
        text: str,
        *,
        on_audio_start: Callable[[], Awaitable[None] | None] | None = None,
    ) -> None:
        if len(text) < 10:
            logger.warning("Text too short for TTS (%d chars), skipping", len(text))
            self._stage = "done"
            self._stage = "idle"
            return

        player = self._player
        self._stage = "tts_stream"
        logger.info("TTS full: %s", text[:80])

        loop = asyncio.get_running_loop()
        audio_start_future: asyncio.Future[None] | None = None
        audio_start_task: asyncio.Task[None] | None = None

        if on_audio_start is not None:
            audio_start_future = loop.create_future()

            async def _wait_and_emit_audio_start() -> None:
                await audio_start_future
                maybe_result = on_audio_start()
                if maybe_result is not None:
                    await maybe_result

            audio_start_task = loop.create_task(_wait_and_emit_audio_start())

            def mark_audio_started() -> None:
                if audio_start_future is None or audio_start_future.done():
                    return
                loop.call_soon_threadsafe(audio_start_future.set_result, None)
        else:
            mark_audio_started = lambda: None

        try:
            try:
                player.begin(on_playback_start=mark_audio_started)
                use_direct = player.is_active()
                if use_direct:
                    logger.info(
                        "Audio output active; prebuffering %.2fs",
                        player.prebuffer_seconds,
                    )
            except Exception as e:
                logger.warning("Audio player failed: %s", e)
                use_direct = False

            if not use_direct:
                raise RuntimeError("No local audio output device is available")

            frame_count = 0
            byte_count = 0
            async for frame in self._tts.stream_synthesize(text):
                frame_count += 1
                byte_count += len(frame.audio)

                if not use_direct:
                    mark_audio_started()

                if use_direct:
                    player.put(
                        frame.audio,
                        format=frame.format,
                        sample_rate=frame.sample_rate,
                    )

            logger.info(
                "TTS complete: frames=%d bytes=%d input_chars=%d",
                frame_count,
                byte_count,
                len(text),
            )

            if frame_count == 0:
                raise RuntimeError("TTS returned no audio frames")

            mark_audio_started()
            if use_direct:
                player.finish()
                await player.wait_for_playback()

            if audio_start_task is not None:
                await audio_start_task

            self._stage = "done"
        finally:
            if audio_start_task is not None and not audio_start_task.done():
                audio_start_task.cancel()
                with suppress(asyncio.CancelledError):
                    await audio_start_task
            self._stage = "idle"
