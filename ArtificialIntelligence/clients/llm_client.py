from __future__ import annotations

import asyncio
import logging
import os
import re
import time

from ollama import AsyncClient, Client

logger = logging.getLogger(__name__)

REQUEST_ATTEMPTS = 5
RETRY_DELAY_SECONDS = 1.0
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
NUM_PREDICT = 512


class LLMClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.strip().rstrip("/")
        self.model = model.strip()
        self._client = Client(host=self.ollama_host)
        self._async_client = AsyncClient(host=self.ollama_host)

    @property
    def ollama_host(self) -> str:
        root = self.base_url
        if root.endswith("/v1"):
            root = root[:-3]
        if root.endswith("/api"):
            root = root[:-4]
        return root

    def close(self) -> None:
        # ollama.Client does not require explicit close in most versions
        pass

    def generate(self, prompt: str) -> str:
        started_at = time.perf_counter()
        logger.info(
            "LLM request: model=%s url=%s prompt_chars=%d",
            self.model,
            self.base_url,
            len(prompt),
        )

        attempt = 0
        while True:
            attempt += 1
            try:
                data = self._client.chat(
                    **self._request_args(prompt, stream=False)
                )
                break
            except Exception:
                if attempt >= REQUEST_ATTEMPTS:
                    logger.error("LLM request failed after %d attempts", attempt)
                    raise
                logger.warning("LLM request failed on attempt %d; retrying", attempt)
                time.sleep(RETRY_DELAY_SECONDS * min(attempt, 10))

        message = data.get("message") or {}
        text = self._strip_thinking(message.get("content") or "")

        logger.info(
            "LLM response: chars=%d elapsed=%.2fs",
            len(text),
            time.perf_counter() - started_at,
        )
        return text

    def stream_generate(self, prompt: str):
        started_at = time.perf_counter()
        logger.info(
            "LLM stream request: model=%s url=%s prompt_chars=%d",
            self.model,
            self.base_url,
            len(prompt),
        )

        attempt = 0
        while True:
            attempt += 1
            try:
                stream = self._client.chat(
                    **self._request_args(prompt, stream=True)
                )

                for data in stream:
                    message = data.get("message") or {}
                    chunk = message.get("content") or ""

                    if chunk:
                        yield chunk

                    if data.get("done"):
                        logger.info(
                            "LLM stream done: elapsed=%.2fs",
                            time.perf_counter() - started_at,
                        )
                        return

                logger.warning("LLM stream ended without done marker")
                return

            except Exception:
                if attempt >= REQUEST_ATTEMPTS:
                    logger.error("LLM stream failed after %d attempts", attempt)
                    raise
                logger.warning("LLM stream failed on attempt %d; retrying", attempt)
                time.sleep(RETRY_DELAY_SECONDS * min(attempt, 10))

    def warmup(self) -> None:
        logger.info("LLM warmup: model=%s host=%s", self.model, self.ollama_host)

        attempt = 0
        while True:
            attempt += 1
            try:
                self._client.show(self.model)
                return
            except Exception:
                if attempt >= REQUEST_ATTEMPTS:
                    logger.error("LLM warmup failed after %d attempts", attempt)
                    raise
                logger.warning("LLM warmup failed on attempt %d; retrying", attempt)
                time.sleep(RETRY_DELAY_SECONDS * min(attempt, 10))

    def generate_text(self, prompt: str) -> str:
        return self.generate(prompt)

    async def async_generate(self, prompt: str) -> str:
        started_at = time.perf_counter()
        logger.info(
            "LLM async request: model=%s url=%s prompt_chars=%d",
            self.model,
            self.base_url,
            len(prompt),
        )

        attempt = 0
        while True:
            attempt += 1
            try:
                data = await self._async_client.chat(
                    **self._request_args(prompt, stream=False)
                )
                break
            except Exception:
                if attempt >= REQUEST_ATTEMPTS:
                    logger.error("LLM async request failed after %d attempts", attempt)
                    raise
                logger.warning(
                    "LLM async request failed on attempt %d; retrying",
                    attempt,
                )
                await asyncio.sleep(RETRY_DELAY_SECONDS * min(attempt, 10))

        message = data.get("message") or {}
        text = self._strip_thinking(message.get("content") or "")

        logger.info(
            "LLM async response: chars=%d elapsed=%.2fs",
            len(text),
            time.perf_counter() - started_at,
        )
        return text

    async def async_generate_text(self, prompt: str) -> str:
        return await self.async_generate(prompt)

    def generate_fortune(self, prompt: str) -> str:
        return self.generate_text(prompt)

    async def async_generate_fortune(self, prompt: str) -> str:
        return await self.async_generate_text(prompt)

    def _messages(self, prompt: str):
        return [{"role": "user", "content": prompt}]

    def _request_args(self, prompt: str, *, stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": self._messages(prompt),
            "stream": stream,
            "think": False,
            "options": {
                "temperature": 0.6,
                "top_p": 0.8,
                "repeat_penalty": 1.06,
                "num_ctx": self._num_ctx(),
                "num_predict": NUM_PREDICT,
            },
        }

    def _num_ctx(self) -> int:
        return max(512, int(os.environ["LLM_MAX_MODEL_LEN"]))

    def _strip_thinking(self, text: str) -> str:
        text = THINK_BLOCK_RE.sub("", text)
        text = re.sub(r"</?think\b[^>]*>", "", text, flags=re.IGNORECASE)
        return text.strip()
