# Palmprint Recognition

Real-time palmprint feature extraction and embedding generation for NVIDIA Jetson Nano deployment.

## Scope

- `main.py` starts the live palmprint client, extracts the palm ROI, builds the embedding, and streams it to an upstream websocket service.
- `training/train_tongji_embedding.py` trains the ResNet18 + ArcFace embedding model used by the runtime.

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

```bash
python main.py --camera 0 --host 0.0.0.0 --port 8000
```

Useful flags:

- `--pipeline_ws_url ws://127.0.0.1:8001/ws/palmprint`
- `--cnn_weights assets/weights/tongji_resnet18_arcface_256d.pt`
- `--roi_size 256`
- `--embed_every 1`

## Training

Expected dataset layout:

```text
assets/datasets/tongji/roi/
  session1/
  session2/
```

Run training:

```bash
python training/train_tongji_embedding.py
```