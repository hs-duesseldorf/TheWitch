# syntax=docker/dockerfile:1.7

FROM ollama/ollama:latest

ARG WITCH_LLM_MODEL

RUN --mount=type=cache,id=thewitch-ollama-models,target=/tmp/ollama-model-cache,sharing=locked <<'EOF'
set -eu

: "${WITCH_LLM_MODEL:?WITCH_LLM_MODEL build arg is required}"
export OLLAMA_MODELS=/tmp/ollama-model-cache

ollama serve >/tmp/ollama-build.log 2>&1 &
ollama_pid="$!"

cleanup() {
    kill "$ollama_pid" 2>/dev/null || true
    wait "$ollama_pid" 2>/dev/null || true
}
trap cleanup EXIT

until ollama list >/dev/null 2>&1; do
    sleep 1
done

ollama pull "$WITCH_LLM_MODEL"

mkdir -p /root/.ollama/models
cp -a "$OLLAMA_MODELS/." /root/.ollama/models/
EOF

EXPOSE 11434

CMD ["serve"]
