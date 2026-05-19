from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator
from urllib.parse import urljoin

import aiohttp

logger = logging.getLogger(__name__)

REQUEST_ATTEMPTS = 5
RETRY_DELAY_SECONDS = 1.0
BOUNDARY_RE = re.compile(r"([.!?,;:]+)\s+")
MAX_CHUNK_LATENCY_SECONDS = 0.45
FIRST_CHUNK_WORDS = 8
LATER_CHUNK_WORDS = 14


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
    ):
        self.base_url = base_url.strip().rstrip("/")
        self.model = model.strip()
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def generate(self, prompt: str) -> str:
        started_at = time.perf_counter()
        logger.info("LLM request: model=%s url=%s prompt_chars=%d", self.model, self.base_url, len(prompt))
        attempt = 0
        while True:
            attempt += 1
            try:
                session = await self._ensure_session()
                async with session.post(
                    urljoin(self.base_url, "/v1/completions"),
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "max_tokens": 256,
                    },
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                break
            except Exception:
                logger.warning("LLM request failed on attempt %d; retrying", attempt)
                await asyncio.sleep(RETRY_DELAY_SECONDS * min(attempt, 10))
        text = (data.get("choices", [{}])[0].get("text") or "").strip()
        logger.info("LLM response: chars=%d elapsed=%.2fs", len(text), time.perf_counter() - started_at)
        return text

    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        started_at = time.perf_counter()
        logger.info("LLM stream request: model=%s url=%s prompt_chars=%d", self.model, self.base_url, len(prompt))
        attempt = 0
        while True:
            attempt += 1
            try:
                session = await self._ensure_session()
                async with session.post(
                    urljoin(self.base_url, "/v1/completions"),
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "max_tokens": 256,
                        "stream": True,
                    },
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.content:
                        line = line.decode().strip()
                        if not line or not line.startswith("data:"):
                            continue
                        if line == "data: [DONE]":
                            return
                        try:
                            data = line[5:].strip()
                            if not data:
                                continue
                            chunk = data.get("choices", [{}])[0].get("text") or ""
                            if chunk:
                                yield chunk
                        except Exception:
                            continue
                logger.info("LLM stream done: elapsed=%.2fs", time.perf_counter() - started_at)
                return
            except Exception:
                logger.warning("LLM stream failed on attempt %d; retrying", attempt)
                await asyncio.sleep(RETRY_DELAY_SECONDS * attempt)

    async def generate_fortune(self, prompt: str) -> str:
        return await self.generate(prompt)

    async def stream_fortune_chunks(self, prompt: str) -> AsyncIterator[str]:
        async for chunk in self.stream_spoken_chunks(prompt):
            yield chunk

    async def stream_spoken_chunks(self, prompt: str) -> AsyncIterator[str]:
        buffer = ""
        emitted = False
        last_flush_check = time.perf_counter()

        async for token in self.stream_generate(prompt):
            buffer += token
            now = time.perf_counter()
            chunk, buffer = self._pop_spoken_chunk(
                buffer,
                first=not emitted,
                force=(now - last_flush_check) >= MAX_CHUNK_LATENCY_SECONDS,
            )
            if chunk:
                emitted = True
                last_flush_check = now
                yield chunk

        final = re.sub(r"\s+", " ", buffer).strip()
        if final:
            yield final

    def _pop_spoken_chunk(self, buffer: str, *, first: bool, force: bool) -> tuple[str | None, str]:
        cleaned = re.sub(r"\s+", " ", buffer).strip()
        if not cleaned:
            return None, ""

        target_words = FIRST_CHUNK_WORDS if first else LATER_CHUNK_WORDS
        words = cleaned.split()

        boundary = BOUNDARY_RE.search(cleaned)
        if boundary and len(words) >= max(4, target_words // 2):
            end = boundary.end()
            return cleaned[:end].strip(), cleaned[end:].strip()

        if len(words) >= target_words:
            return cleaned, ""

        if force and len(words) >= max(5, target_words // 2):
            return cleaned, ""

        return None, buffer