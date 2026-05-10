FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

# Pre-download TTS model to cache
ENV COQUI_TOS_AGREED=1
RUN python -c "from TTS.api import TTS; TTS(model_name='tts_models/de/thorsten/vits'); print('TTS model cached')" || echo "Model download failed, will download at runtime"

COPY main.py .

RUN mkdir -p /assets

EXPOSE 5002

CMD ["python", "main.py"]
