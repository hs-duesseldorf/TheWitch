from __future__ import annotations

import os

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .server_config import LANGUAGE, profile

app = FastAPI()
PROFILE = profile()


class SpeechRequest(BaseModel):
    input: str


def _configured_max_new_tokens() -> int | None:
    configured = os.environ["TTS_MAX_NEW_TOKENS"].strip()
    if not configured or configured.lower() in {"0", "none", "unlimited"}:
        return None
    return int(configured)


def _speech_payload(text: str) -> dict[str, object]:
    temperature = float(os.environ["WITCH_TTS_TEMPERATURE"])
    repetition_penalty = float(os.environ["WITCH_TTS_REPETITION_PENALTY"])

    payload: dict[str, object] = {
        "input": text,
        "model": PROFILE.model,
        "task_type": PROFILE.task_type,
        "language": LANGUAGE,
        "response_format": "pcm",
        "stream": True,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
    }
    max_new_tokens = _configured_max_new_tokens()
    if max_new_tokens is not None:
        payload["max_new_tokens"] = max_new_tokens
    if PROFILE.voice:
        payload["voice"] = PROFILE.voice
    if PROFILE.instructions:
        payload["instructions"] = PROFILE.instructions
    return payload


async def _open_upstream(
    client: httpx.AsyncClient,
    upstream_url: str,
    text: str,
) -> httpx.Response:
    upstream = await client.send(
        client.build_request("POST", upstream_url, json=_speech_payload(text)),
        stream=True,
    )
    if upstream.status_code >= 400:
        body = await upstream.aread()
        await upstream.aclose()
        raise HTTPException(upstream.status_code, body.decode(errors="replace"))
    return upstream


@app.post("/v1/audio/speech")
async def speech(request: SpeechRequest) -> StreamingResponse:
    upstream_url = f"http://127.0.0.1:{os.environ['WITCH_TTS_INTERNAL_PORT']}/v1/audio/speech"

    client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
    upstream = await _open_upstream(client, upstream_url, request.input)
    media_type = upstream.headers.get("content-type", "audio/pcm")

    async def audio_stream():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        audio_stream(),
        media_type=media_type,
    )
