ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libegl1 \
    libgles2 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ARG INSTALL_TORCH=1
RUN if [ "$INSTALL_TORCH" = "1" ]; then \
      pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1; \
    fi

RUN PIP_CONSTRAINT= pip install --no-cache-dir \
    "numpy<2" \
    "protobuf>=4.25.3,<5" \
    websockets \
    msgspec \
    opencv-python-headless \
    mediapipe \
    adafruit-blinka \
    adafruit-circuitpython-vl53l0x

COPY shared/ ./shared/
COPY ImageProcessing/ ./

CMD ["python", "main.py"]
