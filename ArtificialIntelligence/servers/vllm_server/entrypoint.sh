#!/bin/bash
set -e

LLM_MODEL="${WITCH_LLM_MODEL:-qwen/qwen3-7b-instruct}"
TTS_MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
LLM_PORT="${WITCH_LLM_PORT:-8082}"
TTS_PORT="${WITCH_TTS_PORT:-8083}"

echo "Starting vLLM LLM server on port $LLM_PORT with model $LLM_MODEL..."
vllm serve "$LLM_MODEL" \
    --host 0.0.0.0 \
    --port "$LLM_PORT" \
    --trust-remote-code &

echo "Starting vLLM TTS server on port $TTS_PORT with model $TTS_MODEL..."
vllm serve "$TTS_MODEL" \
    --omni \
    --host 0.0.0.0 \
    --port "$TTS_PORT" \
    --trust-remote-code \
    --enforce-eager &

wait