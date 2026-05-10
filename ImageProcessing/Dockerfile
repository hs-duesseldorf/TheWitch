FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libegl1 \
    libgles2 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY ImageProcessing/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

COPY shared/ ./shared/
COPY ImageProcessing/palmprint_client/ ./palmprint_client/
COPY ImageProcessing/main.py ImageProcessing/webcam_bridge.py ./

CMD ["python", "main.py"]
