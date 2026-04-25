# Palmprint Recognition

Real-time palmprint feature extraction and embedding generation for NVIDIA Jetson Nano deployment.

## Scope

- `main.py` starts the live palmprint client, extracts the palm ROI, builds the embedding, and streams it to an upstream websocket service.
- `debug_server.py` starts the optional local websocket server and debug UI. It can be replaced by the real upstream server later.
- `training/train_tongji_embedding.py` trains interchangeable embedding checkpoints, currently ResNet18 with ArcFace or contrastive cosine loss.

## Repository Layout

- `palmprint_client/` runtime, ROI extraction, model loading, transport, and debug UI server
- `training/` training entrypoints
- `assets/` local-only weights, datasets, and third-party model files

## Runtime Requirements

- Python 3.11+
- NVIDIA Jetson Nano or another CUDA-capable NVIDIA device

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Local, with the runtime and debug server on the same machine:

```bash
python debug_server.py --port 8001
python main.py --camera 0 --pipeline_ws_url ws://localhost:8001/ws/palmprint
```

Jetson runtime sending to the AI websocket server on the PC:

```bash
# PC
python debug_server.py --port 8001

# Jetson
python main.py --camera 0 --pipeline_ws_url ws://<PC-LAN-IP>:8001/ws/palmprint
```

Useful flags:

- `--pipeline_ws_url ws://localhost:8001/ws/palmprint`
- `--embedding_model arcface`
- `--embedding_model contrastive`

The debug server binds to remote clients by default so the Jetson can connect to `/ws/palmprint` and the browser on the PC can open the printed debug UI URL. The browser debug UI receives live updates over `/ws/debug` and may use simple HTTP endpoints on `debug_server.py` for UI commands.

## Training

Expected dataset layout:

```text
assets/datasets/tongji/roi/
  session1/
  session2/
```

Run training:

```bash
python training/train_tongji_embedding.py --loss arcface --label_mode person
python training/train_tongji_embedding.py --loss contrastive --label_mode person
```

Use `--label_mode palm` to reproduce the older behavior where each palm is a separate class. Default output names include model, loss, label mode, and embedding dimension so both training runs can coexist under `assets/weights/`.

`--label_mode person` assumes Tongji ordering of two palms per person with 10 images per palm and groups those 20 images into one identity class.
