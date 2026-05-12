FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ARG WITCH_TTS_MODEL
ENV WITCH_TTS_MODEL=${WITCH_TTS_MODEL}

RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    uvicorn[standard]>=0.29.0 \
    soundfile>=0.12.1 \
    TTS==0.21.3

ENV COQUI_TOS_AGREED=1

RUN python -c "from TTS.api import TTS; TTS('${WITCH_TTS_MODEL}')"

COPY main.py .

RUN mkdir -p /assets

ARG WITCH_TTS_PORT=8083
EXPOSE ${WITCH_TTS_PORT}

CMD ["python", "main.py"]
