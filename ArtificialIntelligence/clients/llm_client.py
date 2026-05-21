from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections.abc import AsyncIterator

from ollama import AsyncClient

logger = logging.getLogger(__name__)

REQUEST_ATTEMPTS = 5
RETRY_DELAY_SECONDS = 1.0
BOUNDARY_RE = re.compile(r"([.!?,;:]+)\s+")
MAX_CHUNK_LATENCY_SECONDS = 0.2
FIRST_CHUNK_WORDS = 200
LATER_CHUNK_WORDS = 200
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
DEFAULT_NUM_CTX = 2048


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
    ):
        self.base_url = base_url.strip().rstrip("/")
        self.model = model.strip()
        self._client: AsyncClient | None = None

    @property
    def ollama_host(self) -> str:
        root = self.base_url
        if root.endswith("/v1"):
            root = root[:-3]
        if root.endswith("/api"):
            root = root[:-4]
        return root

    def _ensure_client(self) -> AsyncClient:
        if self._client is None:
            self._client = AsyncClient(host=self.ollama_host)
        return self._client

    async def close(self) -> None:
        if self._client is None:
            return
        http_client = getattr(self._client, "_client", None)
        close = getattr(http_client, "aclose", None)
        if close:
            await close()
        self._client = None

    async def generate(self, prompt: str) -> str:
        started_at = time.perf_counter()
        logger.info("LLM request: model=%s url=%s prompt_chars=%d", self.model, self.base_url, len(prompt))
        attempt = 0
        while True:
            attempt += 1
            try:
                data = await self._ensure_client().chat(**self._request_args(prompt, stream=False, temperature=0.55))
                break
            except Exception:
                if attempt >= REQUEST_ATTEMPTS:
                    logger.error("LLM request failed after %d attempts", attempt)
                    raise
                logger.warning("LLM request failed on attempt %d; retrying", attempt)
                await asyncio.sleep(RETRY_DELAY_SECONDS * min(attempt, 10))
        message = data.get("message") or {}
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
                stream = await self._ensure_client().chat(**self._request_args(prompt, stream=True, temperature=0.55))
                async for data in stream:
                    message = data.get("message") or {}
                    chunk = message.get("content") or ""
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        logger.info("LLM stream done: elapsed=%.2fs", time.perf_counter() - started_at)
                        return
                logger.warning("LLM stream ended without done marker")
                return
            except Exception:
                if attempt >= REQUEST_ATTEMPTS:
                    logger.error("LLM stream failed after %d attempts", attempt)
                    raise
                logger.warning("LLM stream failed on attempt %d; retrying", attempt)
                await asyncio.sleep(RETRY_DELAY_SECONDS * min(attempt, 10))

    async def warmup(self) -> None:
        logger.info("LLM warmup: model=%s host=%s", self.model, self.ollama_host)
        attempt = 0
        while True:
            attempt += 1
            try:
                await self._ensure_client().show(self.model)
                return
            except Exception:
                if attempt >= REQUEST_ATTEMPTS:
                    logger.error("LLM warmup failed after %d attempts", attempt)
                    raise
                logger.warning("LLM warmup failed on attempt %d; retrying", attempt)
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
            {"role": "user", "content": prompt},
        ]

    def _request_args(self, prompt: str, *, stream: bool, temperature: float) -> dict[str, object]:
        return {
            "model": self.model,
            "messages": self._messages(prompt),
            "stream": stream,
            "think": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.8,
                "num_ctx": self._num_ctx(),
                "num_predict": 180,
            },
        }

    def _num_ctx(self) -> int:
        raw = os.environ.get("WITCH_LLM_NUM_CTX", "").strip()
        if not raw:
            return DEFAULT_NUM_CTX
        try:
            return max(512, int(raw))
        except ValueError:
            logger.warning("Invalid WITCH_LLM_NUM_CTX=%r; using %d", raw, DEFAULT_NUM_CTX)
            return DEFAULT_NUM_CTX

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
