from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from contextlib import suppress
from typing import Any, Awaitable, Callable

from ArtificialIntelligence.speech.audio_player import AudioPlayer

logger = logging.getLogger(__name__)


def normalize_llm_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    text = text.replace(r"\n", " ").replace(r"\r", " ").replace(r"\t", " ")
    text = text.replace(r"\\", " ")

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
        line = re.sub(r"^[-*•]+\s*", "", line)
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


class SpeechPipeline:
    INITIAL_RETRY_DELAY = 2.0
    MAX_RETRY_DELAY = 60.0

    def __init__(self, llm: Any, tts: Any, *, prebuffer_seconds: float, speaker_delay_seconds: float):
        self._llm = llm
        self._tts = tts
        self._player = AudioPlayer(
            prebuffer_seconds=prebuffer_seconds,
            speaker_delay_seconds=speaker_delay_seconds,
        )
        self._stage = "idle"

    @property
    def stage(self) -> str:
        return self._stage

    def stop_player(self):
        logger.info("Clearing player")
        with suppress(Exception):
            self._player.clear()
        logger.info("Player cleared")

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
                except Exception as exc:
                    logger.warning("Speech text generation failed, retrying in %ss: %s", delay, exc)
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, self.MAX_RETRY_DELAY)
        finally:
            self._stage = "idle"

    async def play_text(
        self,
        text: str,
        *,
        on_audio_start: Callable[[], Awaitable[None] | None] | None = None,
    ):
        if len(text) < 10:
            logger.warning("Text too short for TTS (%d chars), skipping", len(text))
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

            async def _wait_and_emit_audio_start():
                await audio_start_future
                maybe_result = on_audio_start()
                if maybe_result is not None:
                    await maybe_result

            audio_start_task = loop.create_task(_wait_and_emit_audio_start())

            def mark_audio_started():
                if audio_start_future is None or audio_start_future.done():
                    return
                loop.call_soon_threadsafe(audio_start_future.set_result, None)
        else:
            def mark_audio_started():
                return None

        try:
            try:
                player.begin(on_playback_start=mark_audio_started)
                use_direct = player.is_active()
                if use_direct:
                    logger.info("Audio output active; prebuffering %.2fs", player.prebuffer_seconds)
            except Exception as exc:
                logger.warning("Audio player failed: %s", exc)
                use_direct = False

            if not use_direct:
                raise RuntimeError("No local audio output device is available")

            frame_count = 0
            byte_count = 0
            async for frame in self._tts.stream_synthesize(text):
                frame_count += 1
                byte_count += len(frame.audio)
                player.put(frame.audio, format=frame.format, sample_rate=frame.sample_rate)

            logger.info(
                "TTS complete: frames=%d bytes=%d input_chars=%d",
                frame_count,
                byte_count,
                len(text),
            )

            if frame_count == 0:
                raise RuntimeError("TTS returned no audio frames")

            mark_audio_started()
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

    async def _generate_text(self, prompt: str) -> str:
        async_generate = getattr(self._llm, "async_generate_text", None)
        if async_generate is not None:
            return await async_generate(prompt)
        generate = getattr(self._llm, "generate_text", None)
        if generate is not None:
            return await asyncio.to_thread(generate, prompt)
        return await asyncio.to_thread(self._llm.generate_fortune, prompt)
