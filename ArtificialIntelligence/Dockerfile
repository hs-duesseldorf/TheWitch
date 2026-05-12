FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    httpx>=0.27.0 \
    msgspec>=0.18.6 \
    transitions>=0.9.3 \
    streamlit>=1.36.0 \
    websockets>=12.0 \
    flask>=3.0.0 \
    flask-cors>=4.0.0 \
    fastapi>=0.110.0 \
    uvicorn[standard]>=0.29.0 \
    ollama>=0.4.0

COPY shared/ ./shared/
COPY ArtificialIntelligence/clients/ ./clients/
COPY ArtificialIntelligence/servers/ ./servers/
COPY ArtificialIntelligence/state_machine/ ./state_machine/
COPY ArtificialIntelligence/websocket_server/ ./websocket_server/
COPY ArtificialIntelligence/message_channels.py .
COPY ArtificialIntelligence/message_parser.py .
COPY ArtificialIntelligence/main.py .
COPY ArtificialIntelligence/runtime.py .

ARG WITCH_AI_WS_PORT
ARG WITCH_AI_UI_PORT
EXPOSE ${WITCH_AI_WS_PORT}
EXPOSE ${WITCH_AI_UI_PORT}

CMD ["python", "main.py"]
