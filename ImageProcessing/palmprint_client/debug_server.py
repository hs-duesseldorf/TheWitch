from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from .transport import LatestMessageBus

INDEX_HTML_PATH = Path(__file__).resolve().parent / "static" / "index.html"


def create_debug_app(*, event_bus: LatestMessageBus) -> FastAPI:
    app = FastAPI(title="Palmprint Debug UI")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(
            INDEX_HTML_PATH.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/ui-version")
    async def ui_version() -> JSONResponse:
        stat = INDEX_HTML_PATH.stat()
        return JSONResponse(
            {"mtime_ns": stat.st_mtime_ns},
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/healthz")
    async def healthz() -> PlainTextResponse:
        return PlainTextResponse("ok")

    @app.get("/api/latest")
    async def latest() -> JSONResponse:
        message = event_bus.latest()
        if message is None:
            return JSONResponse({"status": "starting"})
        return JSONResponse(message)

    @app.websocket("/ws/debug")
    async def debug_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        last_sequence = -1
        try:
            while True:
                sequence, message = await asyncio.to_thread(event_bus.wait_for_message, last_sequence, 1.0)
                if sequence == last_sequence or message is None:
                    continue
                last_sequence = sequence
                await websocket.send_json(message)
        except WebSocketDisconnect:
            return

    return app
