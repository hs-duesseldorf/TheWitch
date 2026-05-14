from __future__ import annotations

import base64
import logging
import os
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from main import App

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("debug-ui")

this_dir = Path(__file__).parent
html_path = this_dir / "assets" / "debug_ui.html"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

runtime_lock = threading.Lock()
runtime: App | None = None
state_machine_graph: str | None = None


def get_runtime() -> App:
    global runtime
    with runtime_lock:
        if runtime is None:
            runtime = App()
            ws_thread = threading.Thread(target=_run_ws, name="witch-ws", daemon=True)
            ws_thread.start()
        return runtime


def _run_ws() -> None:
    import asyncio

    try:
        asyncio.run(get_runtime().ws_server.start())
    except Exception:
        logger.exception("WebSocket server thread crashed")


@app.get("/")
@app.get("/debug_ui.html")
def get_ui():
    return FileResponse(html_path, media_type="text/html")


@app.get("/api/state")
def get_state():
    return {"state": get_runtime().runtime.state()}


@app.get("/api/state-machine-graph")
def get_state_machine_graph():
    global state_machine_graph
    if state_machine_graph is not None:
        return {"graph": state_machine_graph}

    sm = get_runtime().state_machine.machine
    try:
        state_machine_graph = sm.get_graph().source.replace("direction LR", "direction TB")
    except Exception as e:
        logger.warning("Failed to generate state machine graph: %s", e)
        state_machine_graph = "graph TD\n  error[Graph unavailable]"
    return {"graph": state_machine_graph}


@app.get("/api/config")
def get_config():
    ws_port = int(os.getenv("WITCH_AI_PORT"))
    return {"ws_port": ws_port}


@app.get("/api/tts-audio")
def get_tts_audio():
    latest = get_runtime().runtime.latest_tts_audio()
    if latest is None:
        return {"audio": None, "sample_rate": None}

    audio, sample_rate = latest
    return {
        "audio": base64.b64encode(audio).decode(),
        "sample_rate": sample_rate,
    }


@app.post("/api/reset")
def reset_state():
    result = get_runtime().runtime.trigger_state_event("reset")
    return {"state": result}


if __name__ == "__main__":
    port = int(os.getenv("WITCH_AI_UI_PORT"))
    logger.info("FastAPI server starting on 0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
