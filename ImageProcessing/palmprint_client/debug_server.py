from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from .palmprint_data import PalmprintData
from .transport import LatestMessageBus

INDEX_HTML_PATH = Path(__file__).resolve().parent / "static" / "index.html"


class DebugWebSocketHub:
    def __init__(self):
        self.event_bus = LatestMessageBus({"status": "starting", "message": "Waiting for runtime websocket."})
        self.runtime_sockets: set[WebSocket] = set()
        self.lock = asyncio.Lock()

    async def runtime_socket(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self.lock:
            self.runtime_sockets.add(websocket)
        try:
            async for raw_message in websocket.iter_text():
                message = self._decode_json(raw_message)
                if message is not None:
                    self._to_palmprint_data(message)
                    self.event_bus.publish(message)
        except WebSocketDisconnect:
            return
        finally:
            async with self.lock:
                self.runtime_sockets.discard(websocket)

    async def debug_socket(self, websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            await self._send_latest_messages(websocket)
        except WebSocketDisconnect:
            return

    async def _send_latest_messages(self, websocket: WebSocket) -> None:
        last_sequence = -1
        while True:
            sequence, message = await asyncio.to_thread(self.event_bus.wait_for_message, last_sequence, 1.0)
            if sequence == last_sequence or message is None:
                continue
            last_sequence = sequence
            await websocket.send_json(message)

    async def send_runtime_command(self, message: dict[str, Any]) -> bool:
        async with self.lock:
            sockets = tuple(self.runtime_sockets)
        if not sockets:
            self.event_bus.publish({"type": "command_result", "ok": False, "error": "Runtime websocket is not connected"})
            return False
        for websocket in sockets:
            try:
                await websocket.send_json(message)
            except Exception:
                async with self.lock:
                    self.runtime_sockets.discard(websocket)
        return True

    def _decode_json(self, raw_message: str) -> dict[str, Any] | None:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            return None
        return message if isinstance(message, dict) else None

    def _to_palmprint_data(self, message: dict[str, Any]) -> PalmprintData | None:
        if not self._looks_like_palmprint_data(message):
            return None
        return PalmprintData.from_dict(message)

    def _looks_like_palmprint_data(self, message: dict[str, Any]) -> bool:
        return all(key in message for key in ("status", "hand", "proportions", "vector"))


def create_debug_app() -> FastAPI:
    app = FastAPI(title="Palmprint Debug UI", docs_url=None, redoc_url=None, openapi_url=None)
    hub = DebugWebSocketHub()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(
            INDEX_HTML_PATH.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/healthz")
    async def healthz() -> PlainTextResponse:
        return PlainTextResponse("ok")

    @app.post("/api/embedding-model")
    async def select_embedding_model(request: Request) -> JSONResponse:
        body = await request.json()
        model_id = body.get("model_id") if isinstance(body, dict) else None
        if not isinstance(model_id, str) or not model_id.strip():
            return JSONResponse({"ok": False, "error": "model_id is required"}, status_code=400)

        connected = await hub.send_runtime_command(
            {"type": "select_embedding_model", "model_id": model_id.strip()}
        )
        status = 202 if connected else 503
        return JSONResponse({"ok": connected}, status_code=status)

    @app.websocket("/ws/palmprint")
    async def runtime_socket(websocket: WebSocket) -> None:
        await hub.runtime_socket(websocket)

    @app.websocket("/ws/debug")
    async def debug_socket(websocket: WebSocket) -> None:
        await hub.debug_socket(websocket)

    return app
