# The Witch

A digital installation featuring a virtual fortune teller that reads visitors' palms in real time using camera detection, AI analysis, text-to-speech, and a 3D animated fortune teller.

## Architecture

```mermaid
flowchart LR
    ip[ImageProcessing<br/>Camera + Hand Tracking]
    ai[AI Service<br/>State Machine + Orchestration]
    llm[LLM Server<br/>Ollama]
    tts[TTS Server<br/>vLLM-Omni]
    debug[Debug UI<br/>http://localhost:10030/]
    unreal[3D / Unreal]

    ip <-- "WebSocket /ws/ip-ai" --> ai
    ip <-- "WebSocket /ws/ip-ai-video" --> ai
    ip <-- "WebSocket /ws/ip-roi" --> ai

    debug -- "HTTP /api/*" --> ai
    ai -- "WebSocket /ws/ip-ai" --> debug
    ai -- "WebSocket /ws/ai-3d" --> debug
    ai -- "WebSocket /ws/ai-3d-video" --> debug
    ai -- "WebSocket /ws/ai-3d-roi" --> debug

    unreal <-- "WebSocket /ws/ai-3d" --> ai
    unreal <-- "WebSocket /ws/ai-3d-video" --> ai
    unreal <-- "WebSocket /ws/ai-3d-roi" --> ai

    ai -- "Ollama /api/chat" --> llm
    ai -- "HTTP /v1/audio/speech/stream" --> tts
    ai -- "audio" --> speakers

    speakers[Speakers + VB-Cable]
```

## Quick Start

Prerequisites: Python with `venv` support, camera/sensor access on the host.

Run the full stack with witch-compose:

```bash
./witch-compose up
```

The orchestrator creates missing venvs and installs requirements automatically.

Run selected services:

```bash
./witch-compose up ai ip
./witch-compose up llm tts
```

Open the debug UI:

```text
http://localhost:10030/
```

For a split setup, run `ai` on the desktop and `ip` on the Jetson or camera machine. Set `WITCH_AI_HOST` in `.env` on the camera machine to the desktop address.

```bash
./witch-compose up ip
```

## Services

- `llm`: Ollama server for LLM inference
- `tts`: vLLM-Omni server for text-to-speech
- `ai`: state machine, LLM/TTS orchestration, WebSocket API, debug UI
- `ip`: camera, hand tracking, palm ROI, and seat sensor client

## Configure

`.env` is the shared configuration file. Local Python loads it with `python-dotenv`; `witch-compose` also injects it into child processes.

For an all-local run on one machine:

```dotenv
WITCH_LLM_HOST=localhost
WITCH_TTS_HOST=localhost
WITCH_AI_HOST=localhost
```

For mixed machines, Jetson, or LAN setups, use an address reachable from every service that needs it:

```dotenv
WITCH_LLM_HOST=192.168.1.20
WITCH_TTS_HOST=192.168.1.21
WITCH_AI_HOST=192.168.1.10
```

The app derives service URLs from those host and port values:

```dotenv
LLM: http://${WITCH_LLM_HOST}:${WITCH_LLM_PORT}
TTS: http://${WITCH_TTS_HOST}:${WITCH_TTS_PORT}
AI: ws://${WITCH_AI_HOST}:${WITCH_AI_PORT}
```

Do not use `localhost` for cross-machine connections. Use the LAN address of the machine running that service.

### Local Audio

The AI plays TTS audio directly to VB-Cable (for Unreal) and default speakers.

For Windows/Mac: download https://vb-audio.com/Cable/ and restart device

For Linux with PipeWire/PulseAudio compatibility, create a persistent null sink:

```bash
./scripts/setup_linux_virtual_audio.sh
```

This writes:

```text
~/.config/pipewire/pipewire-pulse.conf.d/10-witch-virtual-cable.conf
```

and loads a sink named `WitchVirtualCable` immediately. The capture source exposed by PipeWire is:

```text
WitchVirtualCable.monitor
```

Use `WitchVirtualCable.monitor` in Unreal or any other receiver that should listen to the AI voice. The AI process opens the operating system default output plus `WitchVirtualCable` when it is present, so Fedora Settings still controls the physical speaker/headphone output.

Verify the setup:

```bash
pactl list short sinks | grep WitchVirtualCable
pactl list short sources | grep WitchVirtualCable.monitor
```

If PipeWire/PulseAudio is restarted before the sink appears, restart the user service:

```bash
systemctl --user restart pipewire-pulse
```

Audio plays automatically via sounddevice.


## Local Python

Prerequisites:

- `uv` available to create Python 3.12 venvs
- Camera and sensor access configured for the host machine
- `.env` hosts set to values reachable by the local services, usually `localhost` for an all-local run

Create the venvs once:

```bash
./witch-compose --build
```

Start the services together:

```bash
./witch-compose up
```

Or start them in separate terminals:

Linux:

`llm`:

```bash
source ArtificialIntelligence/servers/llm/.venv/bin/activate
python ArtificialIntelligence/servers/llm/run.py
```

`tts`:

```bash
source ArtificialIntelligence/servers/tts/.venv/bin/activate
python ArtificialIntelligence/servers/tts/run.py
```

`ai`:

```bash
source ArtificialIntelligence/.venv/bin/activate
python -m ArtificialIntelligence.main
```

`ip`:

```bash
source ImageProcessing/.venv/bin/activate
python -m ImageProcessing.main
```

Windows PowerShell:

`llm`:

```powershell
cd ArtificialIntelligence\servers\llm
.venv\Scripts\Activate.ps1
python ArtificialIntelligence\servers\llm\run.py
```

`tts`:

```powershell
cd ArtificialIntelligence\servers\tts
.venv\Scripts\Activate.ps1
python ArtificialIntelligence\servers\tts\run.py
```

`ai`:

```powershell
ArtificialIntelligence\.venv\Scripts\Activate.ps1
python -m ArtificialIntelligence.main
```

`ip`:

```powershell
ImageProcessing\.venv\Scripts\Activate.ps1
python -m ImageProcessing.main
```

## Seat Sensor

The `ip` service publishes a `person_detected` event when the VL53L0X seat sensor detects someone sitting down. For local testing without sensor hardware:

```dotenv
WITCH_SEAT_SENSOR_OVERRIDE=true
```

## Custom TTS Voice

Qwen3-TTS uses the built-in `vivian` CustomVoice speaker by default.

## Unreal Engine Connection

Audio goes to VB-Cable which Unreal can capture as input.

```text
# Same PC as AI, native Unreal:
ws://localhost:10031/ws/ai-3d

# Different PC:
ws://<AI-machine-LAN-IP>:10031/ws/ai-3d

# AI video and ROI streams:
ws://<AI-machine-LAN-IP>:10031/ws/ai-3d-video
ws://<AI-machine-LAN-IP>:10031/ws/ai-3d-roi
```

## Useful Commands

```bash
./witch-compose up
./witch-compose up ai ip
./witch-compose up llm tts
./witch-compose --build
./witch-compose kill
```
