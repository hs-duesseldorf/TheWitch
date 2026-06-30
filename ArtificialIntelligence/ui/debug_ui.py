import logging
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Connection bewteen the debug_ui.html (website) and the runtime.py
# Calls happen via FastApi 
# Communicates with debug_ui.html, runtime.py and state_machine.py
# Only Changes necessary here are when aspects are added to debug_ui.html that require Information from
# the code or send Information (f.e. Button presses) to the code

# debug_ui_manual.html is currently !NOT FUNCTIONING! due to a refactor
# It's usecase was to simulate certain events / state changes to make debugging easier
# the html and necessary Methods here are still provided and can be reinstated
# it called on simulation Methods in runtime.py, which in turn called manual scene triggers in state_machine.py

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("debug-ui")

this_dir = Path(__file__).resolve().parent.parent
html_path = this_dir / "assets" / "debug_ui.html"
html_path_manual = this_dir / "assets" / "debug_ui_manual.html"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_runtime = None
_state_machine = None


def set_runtime(ws_server, state_machine, runtime):
    global _runtime, _state_machine
    _runtime = runtime
    _state_machine = state_machine


def _state_response(state: str):
    return {"state": state}


def _manual_mode_response(enabled: bool):
    return {"manual_mode": enabled}


@app.get("/")
@app.get("/debug_ui.html")
def get_ui():
    return FileResponse(html_path, media_type="text/html")

@app.get("/debug_ui_manual.html")
def get_manual_ui():
    return FileResponse(html_path_manual)


@app.get("/api/state")
def get_state():
    return {"state": _runtime.state}


@app.get("/api/state-machine-graph")
def get_state_machine_graph():
    try:
        graph = _state_machine.machine.get_graph().source.replace("direction LR", "direction TB")
    except Exception as e:
        logger.warning("Failed to generate state machine graph: %s", e)
        graph = "graph TD\n  error[Graph unavailable]"
    return {"graph": graph}


@app.get("/api/config")
def get_config():
    ws_port = int(os.environ["WITCH_AI_PORT"])
    return {"ws_port": ws_port}


@app.post("/api/reset")
async def reset_state():
    return _state_response(await _runtime.trigger_state_event("reset"))


@app.post("/api/trigger/{event}")
async def trigger_event(event: str):
    return _state_response(await _runtime.trigger_state_event(event))

@app.post("/api/sim_hand_event/{event}")
async def simulate_hand_event(event: str):
    return _state_response(await _runtime.simulate_hand_event(event))

@app.post("/api/sim_person_event/{event}")
async def simulate_person_event(event: str):
    return _state_response(await _runtime.simulate_person_event(event))

@app.post("/api/sim_animation_event")
async def simulate_event_done():
    return _state_response(await _runtime.acknowledge_unreal_event())

@app.post("/api/state/{state}")
def set_state(state: str):
    return _state_response(_runtime.force_state(state))

@app.post("/api/manual_mode/on")
def manual_on():
    return _manual_mode_response(_runtime.set_manual_mode(True))

@app.post("/api/manual_mode/off")
def manual_off():
    return _manual_mode_response(_runtime.set_manual_mode(False))


def run():
    port = int(os.environ["WITCH_AI_UI_PORT"])
    logger.info("FastAPI server starting on 0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
