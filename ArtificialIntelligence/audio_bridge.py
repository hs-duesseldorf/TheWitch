from fastapi import FastAPI, UploadFile, File
import io
import platform
import os
import queue
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
SR = 48000
PORT = int(os.environ["WITCH_AUDIO_BRIDGE_PORT"])
playback_queue: queue.Queue[tuple[np.ndarray, int, int]] = queue.Queue()
worker_started = False
worker_lock = threading.Lock()


def find_output_device():
    system = platform.system()

    for i, dev in enumerate(sd.query_devices()):
        name = dev["name"]

        if dev["max_output_channels"] <= 0:
            continue

        if system == "Windows":
            if "CABLE Input" in name and "VB-Audio" in name:
                return i

        if system == "Linux":
            if (
                "WitchVirtualCable" in name
                or "Null Output" in name
            ):
                return i
            if "Virtual" in name:
                return i

        if system == "Darwin":
            if "BlackHole" in name and "2ch" in name:
                return i

    default = sd.query_devices(kind="output")
    if default and default["max_output_channels"] > 0:
        return default["index"]

    raise RuntimeError(f"Could not find audio output device on {system}")


@app.get("/devices")
async def devices():
    return [
        {
            "id": i,
            "name": dev["name"],
            "inputs": dev["max_input_channels"],
            "outputs": dev["max_output_channels"],
        }
        for i, dev in enumerate(sd.query_devices())
    ]


@app.post("/play")
async def play(file: UploadFile = File(...)):
    global worker_started
    audio_bytes = await file.read()

    data, fs = sf.read(io.BytesIO(audio_bytes), dtype="float32")

    if fs != SR:
        data = resample_poly(data, SR, fs)
        fs = SR

    if len(data.shape) == 1:
        data = np.column_stack((data, data))

    device = find_output_device()
    playback_queue.put((data, fs, device))

    with worker_lock:
        if not worker_started:
            threading.Thread(target=_playback_worker, daemon=True).start()
            worker_started = True

    return {"ok": True, "device": device}


def _playback_worker():
    while True:
        data, fs, device = playback_queue.get()
        try:
            sd.play(data, fs, device=device)
            sd.wait()
        finally:
            playback_queue.task_done()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
