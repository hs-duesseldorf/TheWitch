from __future__ import annotations

import logging
import os
import platform
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import soundfile as sf

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False
    sd = None

logger = logging.getLogger(__name__)

OUTPUT_SAMPLE_RATE = 48000
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


def _is_virtual_cable(name: str) -> bool:
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
            for index, device in output_devices:
                if _is_virtual_cable(device["name"]):
                    selected.append(index)

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
            for index, device in output_devices:
                if _is_virtual_cable(device["name"]):
                    hostapi = sd.query_hostapis(device["hostapi"])["name"]
                    if "WASAPI" in hostapi:
                        best_match = index
                        break
                    if best_match is None:
                        best_match = index

            if best_match is not None:
                selected.append(best_match)

    except Exception as exc:
        logger.warning("Error finding devices: %s", exc)

    unique: list[int | None] = []
    for device in selected:
        if device not in unique:
            unique.append(device)
    return unique


class AudioPlayer:
    def __init__(self, *, prebuffer_seconds: float, speaker_delay_seconds: float):
        self._input_sample_rate = TTS_INPUT_SAMPLE_RATE
        self._output_sample_rate = OUTPUT_SAMPLE_RATE
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

    def start(self):
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
                        is_virtual_cable = _is_virtual_cable(device_info["name"])
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
                label = "OS default output device" if device is None else f"device {device}"
                logger.info("Started audio stream on %s; prebuffer=%.2fs", label, self._prebuffer_seconds)
            except Exception as exc:
                label = "OS default output device" if device is None else f"device {device}"
                logger.warning("Failed to start audio stream on %s: %s", label, exc)

        if not self._outputs:
            raise RuntimeError("No audio output stream could be started")

    def is_active(self) -> bool:
        return len(self._outputs) > 0 and HAS_SOUNDDEVICE

    @property
    def prebuffer_seconds(self) -> float:
        return self._prebuffer_seconds

    def begin(self, on_playback_start: Callable[[], None] | None = None):
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

    def put(self, audio_chunk: bytes, *, format: str = "pcm", sample_rate: int | None = None):
        try:
            data = self._convert_chunk(audio_chunk, format=format, sample_rate=sample_rate)
            if data is not None:
                self._enqueue_source(data)
        except Exception as exc:
            logger.warning(
                "Audio playback chunk failed (format=%s, sample_rate=%s, len=%d): %s",
                format,
                sample_rate,
                len(audio_chunk),
                exc,
            )

    def finish(self):
        self._drain_source_queue()
        for output in self._outputs:
            with output.lock:
                output.producer_done = True

    async def wait_for_playback(self):
        if not self._outputs:
            return

        import asyncio

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

    def clear(self):
        with self._source_lock:
            self._source_generation += 1
            generation = self._source_generation
        self._clear_source_queue()
        for output in self._outputs:
            self._clear_output(output, producer_done=True, generation=generation)
        self._pcm_remainder = b""
        self._playback_start_callback = None
        self._playback_start_notified = False

    def _callback(self, output, outdata, frames, callback_time, status):
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
                pending = item.data

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

    def _playback_complete_at(self, output, callback_time, written: int) -> float:
        now = time.perf_counter()
        try:
            device_delay = max(
                0.0,
                float(callback_time.outputBufferDacTime) - float(callback_time.currentTime),
            )
        except (AttributeError, TypeError, ValueError):
            try:
                device_delay = max(0.0, float(output.stream.latency))
            except (AttributeError, TypeError, ValueError):
                device_delay = 0.0
        return now + device_delay + (written / self._output_sample_rate)

    def _notify_playback_started(self):
        if self._playback_start_notified:
            return
        callback = self._playback_start_callback
        self._playback_start_notified = True
        if callback is not None:
            callback()

    def _drain_source_queue(self):
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

    def _enqueue_source(self, data: np.ndarray):
        with self._source_lock:
            generation = self._source_generation
        self._source_queue.put(_QueuedAudio(generation=generation, data=data.copy()))
        self._drain_source_queue()

    def _enqueue_output(self, output: _AudioOutput, data: np.ndarray, *, generation: int):
        with output.lock:
            if generation != output.generation:
                return
            output.buffered_frames += int(data.shape[0])
            output.queue.put(_QueuedAudio(generation=generation, data=data.copy()))

    def _convert_chunk(self, audio_chunk: bytes, *, format: str, sample_rate: int | None) -> np.ndarray | None:
        if format == "wav":
            import io

            data, sample_rate = sf.read(io.BytesIO(audio_chunk), dtype="float32")
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

    def _clear_source_queue(self):
        while True:
            try:
                self._source_queue.get_nowait()
            except queue.Empty:
                break

    def _clear_output(self, output: _AudioOutput, *, producer_done: bool, generation: int):
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
        while True:
            try:
                output.queue.get_nowait()
            except queue.Empty:
                break
