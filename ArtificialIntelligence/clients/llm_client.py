from __future__ import annotations

import asyncio
import logging
import time

import ollama

logger = logging.getLogger(__name__)

REQUEST_ATTEMPTS = 5
RETRY_DELAY_SECONDS = 1.0


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
    ):
        self.base_url = base_url.strip().rstrip("/")
        self.model = model.strip()
        self._client = ollama.AsyncClient(host=self.base_url)

    async def generate(self, prompt: str) -> str:
        started_at = time.perf_counter()
        logger.info("LLM request: model=%s prompt_chars=%d", self.model, len(prompt))
        for attempt in range(1, REQUEST_ATTEMPTS + 1):
            try:
                response = await self._client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "num_ctx": 1024,
                    },
                )
                data = {"response": response["response"]}
                break
            except Exception:
                if attempt == REQUEST_ATTEMPTS:
                    raise
                logger.warning("LLM request failed on attempt %d/%d; retrying", attempt, REQUEST_ATTEMPTS)
                await asyncio.sleep(RETRY_DELAY_SECONDS * attempt)
        text = (data.get("response") or "").strip()
        logger.info("LLM response: chars=%d elapsed=%.2fs", len(text), time.perf_counter() - started_at)
        return text

    async def generate_fortune(self, hand_data: dict) -> str:
        prompt = (
            "Schreibe eine kurze mystische Handlese-Wahrsagung auf Deutsch.\n"
            "Antwort nur mit 2 Saetzen, ohne Einleitung und ohne Aufzaehlung.\n"
            "Ton: weise, leicht dunkel, konkret.\n"
            f"Handdaten: {hand_data}"
        )
        return await self.generate(prompt)