from fastapi import FastAPI, UploadFile, File
import io
import platform
import os
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
PORT = int(os.getenv("WITCH_AUDIO_BRIDGE_PORT", "10034"))


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
                or "Virtual" in name
                or "Null Output" in name
            ):
                return i

        if system == "Darwin":
            if "BlackHole" in name and "2ch" in name:
                return i

    raise RuntimeError(f"Could not find virtual audio output device on {system}")


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
    audio_bytes = await file.read()

    data, fs = sf.read(io.BytesIO(audio_bytes), dtype="float32")

    if fs != SR:
        data = resample_poly(data, SR, fs)
        fs = SR

    if len(data.shape) == 1:
        data = np.column_stack((data, data))

    device = find_output_device()

    def _play_audio():
        sd.play(data, fs, device=device)
        sd.wait()

    threading.Thread(target=_play_audio, daemon=True).start()

    return {"ok": True, "device": device}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)