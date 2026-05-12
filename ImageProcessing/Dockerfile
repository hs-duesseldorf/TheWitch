FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libegl1 \
    libgles2 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    websockets \
    msgspec \
    numpy \
    opencv-python-headless \
    Pillow \
    mediapipe

COPY shared/ ./shared/
COPY ImageProcessing/ ./

CMD ["python", "main.py"]
