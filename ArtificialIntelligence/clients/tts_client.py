from __future__ import annotations

import asyncio
import base64
import logging

import httpx

logger = logging.getLogger(__name__)

REQUEST_ATTEMPTS = 5
RETRY_DELAY_SECONDS = 1.0


class TTSClient:
    def __init__(
        self,
        base_url: str,
        *,
        speaker_wav: str | None = None,
        language: str = "de",
        timeout: float = 120.0,
    ):
        base_url = base_url.strip()
        if not base_url:
            raise ValueError("TTS base URL must not be empty")

        self.base_url = base_url.rstrip("/")
        self.speaker_wav = speaker_wav
        self.language = language
        self.timeout = timeout

    async def synthesize(self, text: str) -> tuple[bytes, int | None] | None:
        url = f"{self.base_url}/api/generate"
        payload = {
            "text": text,
            "language": self.language,
        }
        if self.speaker_wav:
            payload["speaker_wav"] = self.speaker_wav
        logger.debug("TTS request: POST %s text=%d chars", url, len(text))
        for attempt in range(1, REQUEST_ATTEMPTS + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                break
            except httpx.HTTPError:
                if attempt == REQUEST_ATTEMPTS:
                    raise
                logger.warning("TTS request failed on attempt %d/%d; retrying", attempt, REQUEST_ATTEMPTS)
                await asyncio.sleep(RETRY_DELAY_SECONDS * attempt)
        audio_b64 = data.get("audio")
        if not audio_b64:
            logger.error("TTS response missing audio field")
            return None
        decoded = base64.b64decode(audio_b64)
        logger.debug("TTS response: %d bytes", len(decoded))
        sample_rate = data.get("sample_rate")
        return decoded, int(sample_rate) if sample_rate else None
