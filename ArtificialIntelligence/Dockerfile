FROM python:3.11-slim

WORKDIR /app

COPY ArtificialIntelligence/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY shared/ ./shared/
COPY ArtificialIntelligence/clients/ ./clients/
COPY ArtificialIntelligence/servers/ ./servers/
COPY ArtificialIntelligence/state_machine/ ./state_machine/
COPY ArtificialIntelligence/websocket_server/ ./websocket_server/
COPY ArtificialIntelligence/main.py .
COPY ArtificialIntelligence/runtime.py .

COPY ArtificialIntelligence/debug_ui.py .
COPY ArtificialIntelligence/assets/ ./assets/

EXPOSE 8765
EXPOSE 8501

CMD ["python", "main.py"]
