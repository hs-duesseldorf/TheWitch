from __future__ import annotations

import asyncio
import json
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
MAX_CHUNK_LATENCY_SECONDS = 0.2
FIRST_CHUNK_WORDS = 200
LATER_CHUNK_WORDS = 200
NO_THINKING_SYSTEM_PROMPT = (
    "Antworte ausschliesslich auf Deutsch. "
    "Zeige niemals <think>, interne Ueberlegungen oder Analyse. "
    "Gib nur die finale gesprochene Antwort aus."
)
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)


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
            if self._session:
                try:
                    await self._session.close()
                except Exception:
                    pass
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
                    urljoin(self.base_url, "/v1/chat/completions"),
                    json={
                        "model": self.model,
                        "messages": self._messages(prompt),
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 1024,
                        "think_disable": True,
                    },
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                break
            except Exception:
                logger.warning("LLM request failed on attempt %d; retrying", attempt)
                await asyncio.sleep(RETRY_DELAY_SECONDS * min(attempt, 10))
        message = data.get("choices", [{}])[0].get("message") or {}
        text = self._strip_thinking(message.get("content") or "")
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
                    urljoin(self.base_url, "/v1/chat/completions"),
                    json={
                        "model": self.model,
                        "messages": self._messages(prompt),
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "max_tokens": 1024,
                        "stream": True,
                        "think_disable": True,
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
                            payload = line[5:].strip()
                            if not payload:
                                continue
                            data = json.loads(payload)
                            delta = data.get("choices", [{}])[0].get("delta") or {}
                            chunk = delta.get("content") or ""
                            if chunk:
                                yield chunk
                        except Exception:
                            continue
                    logger.info("LLM stream done: elapsed=%.2fs", time.perf_counter() - started_at)
                    return
            except Exception:
                if attempt >= REQUEST_ATTEMPTS:
                    logger.error("LLM stream failed after %d attempts", attempt)
                    raise
                logger.warning("LLM stream failed on attempt %d; retrying", attempt)
                await asyncio.sleep(RETRY_DELAY_SECONDS * min(attempt, 10))

    async def generate_fortune(self, prompt: str) -> str:
        return await self.generate(prompt)

    async def stream_fortune_chunks(self, prompt: str) -> AsyncIterator[str]:
        async for chunk in self.stream_spoken_chunks(prompt):
            yield chunk

    async def stream_spoken_chunks(self, prompt: str) -> AsyncIterator[str]:
        buffer = ""
        emitted = False
        in_think_block = False
        last_flush_check = time.perf_counter()

        async for token in self.stream_generate(prompt):
            token, in_think_block = self._filter_thinking_chunk(token, in_think_block)
            if not token:
                continue
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

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": NO_THINKING_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    def _strip_thinking(self, text: str) -> str:
        text = THINK_BLOCK_RE.sub("", text)
        text = re.sub(r"</?think\b[^>]*>", "", text, flags=re.IGNORECASE)
        return text.strip()

    def _filter_thinking_chunk(self, chunk: str, in_think_block: bool) -> tuple[str, bool]:
        output = []
        pos = 0
        while pos < len(chunk):
            if in_think_block:
                end = chunk.lower().find("</think>", pos)
                if end == -1:
                    return "".join(output), True
                pos = end + len("</think>")
                in_think_block = False
                continue

            start = chunk.lower().find("<think", pos)
            if start == -1:
                output.append(chunk[pos:])
                break
            output.append(chunk[pos:start])
            close = chunk.find(">", start)
            if close == -1:
                return "".join(output), True
            pos = close + 1
            in_think_block = True

        return "".join(output), in_think_block
