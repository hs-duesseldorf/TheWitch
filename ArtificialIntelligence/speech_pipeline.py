from __future__ import annotations

import asyncio
import logging
import os
import platform
import queue
import re
import threading
import time
from contextlib import suppress
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
UNDERRUN_LOG_INTERVAL_SECONDS = 2.0
DRAIN_TIMEOUT = 30.0


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

def normalize_tts_text(text: str) -> str:
    # Remove invisible Unicode characters that break streaming
    text = re.sub(r"[\u200b\u200c\u200d\u2060\uFEFF]", "", text)
    # Remove control chars except newline & tab
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines = []
    for line in text.split("\n"):
        raw = line.rstrip()
        # Preserve empty lines (important for prosody)
        if raw.strip() == "":
            cleaned_lines.append("")
            continue
        cleaned_lines.append(raw)

    # Collapse excessive spaces inside lines
    normalized = "\n".join(cleaned_lines)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    return normalized.strip()

def insert_soft_breaks(text: str) -> str:
    # Insert a newline after long comma clauses to avoid chunk boundaries on commas
    return re.sub(
        r",\s+(?=[A-ZÄÖÜ])",   # comma followed by capital letter
        ",\n",
        text
    )

def segment_for_streaming(text: str, target_len=160, max_len=200):
    parts = []
    current = ""

    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Turns a Text into multiple smaller chunks with full sentences to avoid mid text breaks
    # Flushes on max_length reached and on target_max - keeps sentence length very consistent
    for sentence in sentences:
        if len(current) + len(sentence) > max_len:
            if current.strip():
                parts.append(current.strip())
            current = sentence
        else:
            current += " " + sentence
        if len(current) >= target_len:
            parts.append(current.strip())
            current = ""

    if current.strip():
        parts.append(current.strip())

    return parts


def prepare_tts_streaming_text(text: str):
    cleaned = normalize_tts_text(text)
    softened = insert_soft_breaks(cleaned)
    segments = segment_for_streaming(softened)
    return segments


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
    needle = os.environ.get("WITCH_VIRTUAL_CABLE_NAME", "").strip()
    return bool(needle) and needle.lower() in name.lower()


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
            
        if system == "Linux":
            for i, dev in output_devices:
                if _is_virtual_cable(system, dev["name"]):
                    selected.append(i)
                    
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
    def __init__(self, sample_rate: int = 24000):
        self._sample_rate = sample_rate
        self._prebuffer_seconds = max(
            0.0,
            float(os.environ["WITCH_AUDIO_PREBUFFER_SECONDS"].strip()),
        )
        self._prebuffer_frames = int(SR * self._prebuffer_seconds)
        self._speaker_delay_seconds = max(
            0.0,
            float(os.environ["WITCH_SPEAKER_DELAY_SECONDS"].strip()),
        )
        self._speaker_delay_frames = int(SR * self._speaker_delay_seconds)
        self._pcm_remainder = b""
        self._outputs: list[dict[str, Any]] = []
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

        for device in devices:
            try:
                is_virtual_cable = False

                if device is not None:
                    try:
                        device_info = sd.query_devices(device)
                        is_virtual_cable = _is_virtual_cable(platform.system(), device_info["name"])
                    except Exception:
                        pass

                is_speaker = not is_virtual_cable
                
                output = {
                    "device": device,
                    "queue": queue.Queue(maxsize=0),
                    "pending": np.empty((0, 2), dtype=np.float32),
                    "buffered_frames": 0,
                    "playback_started": self._prebuffer_frames <= 0,
                    "producer_done": False,
                    "underruns": 0,
                    "last_underrun_log": 0.0,
                    "generation": 0,
                    "is_virtual_cable": is_virtual_cable,
                    "lock": threading.Lock(),
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
            logger.warning("No audio output stream could be started")

    def is_active(self) -> bool:
        return len(self._outputs) > 0 and HAS_SOUNDDEVICE

    @property
    def prebuffer_seconds(self) -> float:
        return self._prebuffer_seconds

    def begin(self, on_playback_start: Callable[[], None] | None = None) -> None:
        self.start()
        self._playback_start_callback = on_playback_start
        self._playback_start_notified = False
        for output in self._outputs:
            self._clear_output(output, producer_done=False)
            
        if self._speaker_delay_frames > 0:
            silence = np.zeros((self._speaker_delay_frames, 2), dtype=np.float32)

            for output in self._outputs:
                if not output["is_virtual_cable"]:
                    self._enqueue(output, silence)

    def _callback(
        self,
        output: dict[str, Any],
        outdata: np.ndarray,
        frames: int,
        callback_time: Any,
        status: Any,
    ) -> None:
        outdata.fill(0)
        written = 0
        
        with output["lock"]:
            generation = output["generation"]
            pending = output["pending"]
        chunks: queue.Queue[np.ndarray | None] = output["queue"]

        while written < frames:
            with output["lock"]:
                if generation != output["generation"]:
                    pending = np.empty((0, 2), dtype=np.float32)
                    break
                playback_started = output["playback_started"]
                buffered_frames = output["buffered_frames"]
                if not playback_started and buffered_frames >= self._prebuffer_frames:
                    output["playback_started"] = True
                    playback_started = True

            if not playback_started:
                break

            if pending.shape[0] == 0:
                try:
                    chunk = chunks.get_nowait()
                except queue.Empty:
                    with output["lock"]:
                        producer_done = output["producer_done"]
                    if producer_done:
                        break

                    with output["lock"]:
                        if not output["playback_started"]:
                            output["playback_started"] = self._prebuffer_frames <= 0
                        output["underruns"] += 1
                        underruns = output["underruns"]
                        now = time.perf_counter()
                        should_log = (
                            now - output["last_underrun_log"]
                            >= UNDERRUN_LOG_INTERVAL_SECONDS
                        )
                        if should_log:
                            output["last_underrun_log"] = now
                    if should_log:
                        device = output["device"]
                        label = "OS default output device" if device is None else f"device {device}"
                        logger.warning(
                            "Audio underrun on %s; rebuffering %.2fs (count=%d)",
                            label,
                            self._prebuffer_seconds,
                            underruns,
                        )
                    break

                if chunk is None:
                    with output["lock"]:
                        output["pending"] = np.empty((0, 2), dtype=np.float32)
                    return
                pending = chunk

            available = min(frames - written, pending.shape[0])
            outdata[written : written + available] = pending[:available]
            written += available
            with output["lock"]:
                output["buffered_frames"] = max(0, output["buffered_frames"] - available)

            if available < pending.shape[0]:
                pending = pending[available:]
            else:
                pending = np.empty((0, 2), dtype=np.float32)

        if written > 0:
            self._notify_playback_started()

        with output["lock"]:
            if generation == output["generation"]:
                output["pending"] = pending

    def _notify_playback_started(self) -> None:
        if self._playback_start_notified:
            return
        callback = self._playback_start_callback
        self._playback_start_notified = True
        if callback is not None:
            callback()

    def _enqueue(self, output: dict[str, Any], data: np.ndarray) -> None:
        chunks: queue.Queue[np.ndarray | None] = output["queue"]
        try:
            with output["lock"]:
                output["buffered_frames"] += int(data.shape[0])
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
            logger.warning("Audio playback chunk failed (format=%s, sample_rate=%s, len=%d): %s", format, sample_rate, len(audio_chunk), e)

    def finish(self) -> None:
        for output in self._outputs:
            with output["lock"]:
                output["producer_done"] = True
                output["playback_started"] = True

    async def drain(self, *, timeout: float = DRAIN_TIMEOUT) -> None:
        if not self._outputs:
            return

        deadline = time.perf_counter() + timeout
        while True:
            done = True
            for output in self._outputs:
                with output["lock"]:
                    output_done = (
                        output["producer_done"]
                        and output["buffered_frames"] <= 0
                        and output["pending"].shape[0] == 0
                    )
                if not output_done or not output["queue"].empty():
                    done = False
                    break
            if done:
                await asyncio.sleep(0.1)
                return

            if time.perf_counter() >= deadline:
                logger.error("Audio drain timed out after %.1fs; clearing player", timeout)
                self.clear()
                return

            await asyncio.sleep(0.05)

    def _clear_output(self, output: dict[str, Any], *, producer_done: bool) -> None:
        with output["lock"]:
            output["generation"] += 1
            output["producer_done"] = producer_done
            output["playback_started"] = (
                self._prebuffer_frames <= 0 if not producer_done else True
            )
            output["buffered_frames"] = 0
            output["pending"] = np.empty((0, 2), dtype=np.float32)
        chunks: queue.Queue[np.ndarray | None] = output["queue"]
        while True:
            try:
                chunks.get_nowait()
            except queue.Empty:
                break

    def clear(self) -> None:
        for output in self._outputs:
            self._clear_output(output, producer_done=True)
        self._pcm_remainder = b""
        self._playback_start_callback = None
        self._playback_start_notified = False


class SpeechPipeline:
    def __init__(
        self,
        llm: Any,
        tts: Any,
        tts_seed: int = 42,
    ):
        self._llm = llm
        self._tts = tts
        self._player = StreamingAudioPlayer()
        self._stage = "idle"
        self._tts_seed = tts_seed

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
                    if self._done_callback:
                        await self._done_callback(debug_text)

                    if len(debug_text) < 10:
                        logger.warning("Text too short for TTS (%d chars), skipping", len(debug_text))
                        self._stage = "done"
                        return

                    # Prepare text for streaming TTS: normalize + soft breaks + segmentation
                    segments = prepare_tts_streaming_text(debug_text)
                    segments = [s for s in segments if s.strip()]

                    if not segments:
                        raise RuntimeError("TTS text became empty after normalization/segmentation")

                    logger.info(
                        "TTS segments prepared: count=%d total_chars=%d",
                        len(segments),
                        sum(len(s) for s in segments),
                    )

                    self._stage = "tts_stream"

                    try:
                        player.begin()
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
                        logger.info("No local audio device")

                    frame_count = 0
                    byte_count = 0

                    for idx, tts_text in enumerate(segments, start=1):
                        logger.info(
                            "TTS segment %d/%d: chars=%d preview=%r",
                            idx,
                            len(segments),
                            len(tts_text),
                            tts_text[:80],
                        )

                        async for frame in self._tts.stream_synthesize(tts_text, seed=self._tts_seed):
                            frame_count += 1
                            byte_count += len(frame.audio)

                            if use_direct:
                                player.put(
                                    frame.audio,
                                    format=frame.format,
                                    sample_rate=frame.sample_rate,
                                )

                    logger.info(
                        "TTS complete: frames=%d bytes=%d input_chars=%d segments=%d",
                        frame_count,
                        byte_count,
                        sum(len(s) for s in segments),
                        len(segments),
                    )

                    if frame_count == 0:
                        raise RuntimeError("TTS returned no audio frames")

                    if use_direct:
                        player.finish()
                        await player.drain()

                    logger.info(
                        "TTS complete: frames=%d bytes=%d input_chars=%d",
                        frame_count,
                        byte_count,
                        len(tts_text),
                    )

                    if frame_count == 0:
                        raise RuntimeError("TTS returned no audio frames")

                    if use_direct:
                        player.finish()
                        await player.drain()
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
        tts_text = _normalize_tts_text(text)
        if not tts_text:
            raise RuntimeError("TTS text became empty after normalization")
        if tts_text != text:
            logger.info(
                "Normalized TTS text: llm_chars=%d tts_chars=%d",
                len(text),
                len(tts_text),
            )

        self._stage = "tts_stream"
        logger.info("TTS full: %s", tts_text[:80])

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
                logger.info("No local audio device")

            frame_count = 0
            byte_count = 0
            async for frame in self._tts.stream_synthesize(tts_text, seed=self._tts_seed):
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
                len(tts_text),
            )

            if frame_count == 0:
                raise RuntimeError("TTS returned no audio frames")

            mark_audio_started()
            if use_direct:
                player.finish()
                await player.drain()

            if audio_start_task is not None:
                await audio_start_task

            self._stage = "done"
        finally:
            self._stage = "idle"
