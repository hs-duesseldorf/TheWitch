#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Basic paths and config
# ------------------------------------------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"
MODE="${1:-all}"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"

LLM_DIR="$ROOT_DIR/.venv-llm"
TTS_DIR="$ROOT_DIR/.venv-tts"

LLAMA_VERSION="${LLAMA_VERSION:-9222}"
LLAMA_TAG="b${LLAMA_VERSION}"
LLAMA_ARCHIVE="llama-${LLAMA_TAG}-bin-ubuntu-vulkan-x64.tar.gz"

LLAMA_DIR="$LLM_DIR/llama-cpp"
LLAMA_EXTRACTED_DIR="$LLAMA_DIR/llama-${LLAMA_TAG}"
LLAMA_SERVER="$LLAMA_DIR/llama-server"
LLAMA_TARGET_FILE="$LLAMA_DIR/llama-server.target"

MODEL_DIR="$LLM_DIR/models"

TTS_VLLM_VERSION="${TTS_VLLM_VERSION:-0.21.0}"
TTS_VLLM_OMNI_VERSION="${TTS_VLLM_OMNI_VERSION:-0.21.0rc1}"

children=()

# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

cleanup() {
    if ((${#children[@]})); then
        log "Stopping child processes..."
        kill "${children[@]}" 2>/dev/null || true
        wait "${children[@]}" 2>/dev/null || true
    fi
}

download_file() {
    local url="$1"
    local out="$2"

    mkdir -p "$(dirname "$out")"

    log "Downloading:"
    log "  $url"
    log "  -> $out"

    rm -f "$out.tmp"
    curl -fL --retry 3 --retry-delay 2 -o "$out.tmp" "$url"
    mv "$out.tmp" "$out"
}

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

load_env() {
    [[ -f "$ENV_FILE" ]] || die "Missing .env file: $ENV_FILE"

    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a

    : "${WITCH_LLM_HF:?WITCH_LLM_HF is required, e.g. unsloth/Qwen3-4B-GGUF:Qwen3-4B-Q4_K_M.gguf}"
    : "${WITCH_TTS_MODEL:?WITCH_TTS_MODEL is required}"
    : "${WITCH_LLM_PORT:?WITCH_LLM_PORT is required}"
    : "${WITCH_TTS_PORT:?WITCH_TTS_PORT is required}"
    : "${LLM_MAX_MODEL_LEN:?LLM_MAX_MODEL_LEN is required}"
    : "${LLM_N_GPU_LAYERS:?LLM_N_GPU_LAYERS is required}"
    : "${TTS_MAX_MODEL_LEN:?TTS_MAX_MODEL_LEN is required}"
    : "${TTS_GPU_MEMORY_UTILIZATION:?TTS_GPU_MEMORY_UTILIZATION is required}"
}

parse_llm_model() {
    [[ "$WITCH_LLM_HF" == *":"* ]] || die "WITCH_LLM_HF must be repo:file"

    WITCH_LLM_REPO="${WITCH_LLM_HF%%:*}"
    WITCH_LLM_FILE="${WITCH_LLM_HF#*:}"

    [[ -n "$WITCH_LLM_REPO" && -n "$WITCH_LLM_FILE" ]] || die "Invalid WITCH_LLM_HF: $WITCH_LLM_HF"
}

# ------------------------------------------------------------
# System packages
# ------------------------------------------------------------

install_system_packages() {
    command -v apt-get >/dev/null 2>&1 || return 0

    local missing=()

    # Packages that provide commands we actually use.
    for item in \
        "curl:curl" \
        "tar:tar" \
        "gzip:gzip" \
        "find:findutils" \
        "ffmpeg:ffmpeg" \
        "ninja:ninja-build"
    do
        local cmd="${item%%:*}"
        local pkg="${item##*:}"

        command -v "$cmd" >/dev/null 2>&1 || missing+=("$pkg")
    done

    # ca-certificates is a package, not a command.
    dpkg -s ca-certificates >/dev/null 2>&1 || missing+=("ca-certificates")

    if ! "$PYTHON_BIN" -m venv --help >/dev/null 2>&1; then
        missing+=("python3.12-venv" "python3.12-dev")
    fi

    ((${#missing[@]})) || return 0

    log "Installing system packages: ${missing[*]}"
    export DEBIAN_FRONTEND=noninteractive

    apt-get update
    apt-get install -y --no-install-recommends "${missing[@]}"
    rm -rf /var/lib/apt/lists/*
}

# ------------------------------------------------------------
# LLM setup
# ------------------------------------------------------------

write_llama_wrapper() {
    local real_server="$1"

    chmod +x "$real_server"
    printf '%s\n' "$real_server" > "$LLAMA_TARGET_FILE"

    cat > "$LLAMA_SERVER" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

wrapper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
target="$(cat "$wrapper_dir/llama-server.target")"
target_dir="$(cd "$(dirname "$target")" && pwd)"

export LD_LIBRARY_PATH="$target_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$target_dir"
exec "$target" "$@"
EOF

    chmod +x "$LLAMA_SERVER"

    log "llama-server installed at: $LLAMA_SERVER"
}

install_llama_cpp() {
    if [[ -x "$LLAMA_SERVER" && -f "$LLAMA_TARGET_FILE" ]]; then
        local installed_target
        installed_target="$(<"$LLAMA_TARGET_FILE")"

        if [[ -x "$installed_target" ]]; then
            return 0
        fi

        log "llama-server target is missing or stale:"
        log "  $installed_target"
    fi

    mkdir -p "$LLAMA_DIR"

    local real_server="$LLAMA_EXTRACTED_DIR/llama-server"

    if [[ -f "$real_server" ]]; then
        write_llama_wrapper "$real_server"
        return 0
    fi

    local archive="$LLAMA_DIR/$LLAMA_ARCHIVE"
    local url="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/${LLAMA_ARCHIVE}"

    log "Installing llama.cpp $LLAMA_TAG..."
    download_file "$url" "$archive"

    log "Extracting llama.cpp..."
    tar -xzf "$archive" -C "$LLAMA_DIR"

    if [[ ! -f "$real_server" ]]; then
        log "Archive contents:"
        find "$LLAMA_DIR" -maxdepth 3 -type f | sort | sed 's/^/  /'
        die "Could not find llama-server after extracting llama.cpp."
    fi

    write_llama_wrapper "$real_server"
}

download_llm_model() {
    mkdir -p "$MODEL_DIR"

    local model_path="$MODEL_DIR/$WITCH_LLM_FILE"
    local url="https://huggingface.co/${WITCH_LLM_REPO}/resolve/main/${WITCH_LLM_FILE}"

    if [[ ! -f "$model_path" ]]; then
        log "Downloading LLM model..."
        download_file "$url" "$model_path"
    fi

    [[ -s "$model_path" ]] || die "Model file is missing or empty: $model_path"

    echo "$model_path"
}

start_llm() {
    install_llama_cpp
    parse_llm_model

    local model_path
    model_path="$(download_llm_model)"

    log "Starting LLM:"
    log "  model:      $model_path"
    log "  port:       $WITCH_LLM_PORT"
    log "  ctx:        $LLM_MAX_MODEL_LEN"
    log "  gpu layers: $LLM_N_GPU_LAYERS"

    "$LLAMA_SERVER" \
        --model "$model_path" \
        --host 0.0.0.0 \
        --port "$WITCH_LLM_PORT" \
        --alias "$WITCH_LLM_FILE" \
        --ctx-size "$LLM_MAX_MODEL_LEN" \
        --n-gpu-layers "$LLM_N_GPU_LAYERS" \
        --threads "$(nproc)" \
        --parallel 1 \
        &

    children+=("$!")
}

# ------------------------------------------------------------
# TTS setup
# ------------------------------------------------------------

install_tts_venv() {
    if [[ ! -x "$TTS_DIR/bin/python" ]]; then
        log "Creating TTS venv at $TTS_DIR"

        "$PYTHON_BIN" -m venv "$TTS_DIR"
        "$TTS_DIR/bin/python" -m pip install -U pip wheel
        "$TTS_DIR/bin/python" -m pip install "setuptools<81,>=77.0.3" uv ninja
    fi

    if [[ -x "$TTS_DIR/bin/vllm" ]]; then
        return 0
    fi

    log "Installing vLLM and vLLM-Omni into TTS venv..."

    "$TTS_DIR/bin/uv" pip install \
        --python "$TTS_DIR/bin/python" \
        "vllm==$TTS_VLLM_VERSION" \
        --torch-backend=auto

    "$TTS_DIR/bin/uv" pip install \
        --python "$TTS_DIR/bin/python" \
        "vllm-omni==$TTS_VLLM_OMNI_VERSION"

    "$TTS_DIR/bin/uv" pip install \
        --python "$TTS_DIR/bin/python" \
        aenum \
        accelerate \
        diffusers \
        soundfile \
        scipy \
        pydub \
        openai-whisper \
        omegaconf \
        ninja
}

start_tts() {
    install_tts_venv

    log "Starting TTS:"
    log "  model:    $WITCH_TTS_MODEL"
    log "  port:     $WITCH_TTS_PORT"
    log "  max len:  $TTS_MAX_MODEL_LEN"
    log "  gpu util: $TTS_GPU_MEMORY_UTILIZATION"

    "$TTS_DIR/bin/vllm" serve "$WITCH_TTS_MODEL" \
        --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
        --omni \
        --host 0.0.0.0 \
        --port "$WITCH_TTS_PORT" \
        --max-model-len "$TTS_MAX_MODEL_LEN" \
        --gpu-memory-utilization "$TTS_GPU_MEMORY_UTILIZATION" \
        --no-enable-prefix-caching \
        --trust-remote-code \
        --tensor-parallel-size 1 \
        --pipeline-parallel-size 1 \
        --max-num-seqs 1 \
        --enforce-eager \
        &

    children+=("$!")
}

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

main() {
    load_env
    install_system_packages

    case "$MODE" in
        all)
            start_llm
            start_tts
            ;;
        llm)
            start_llm
            ;;
        tts)
            start_tts
            ;;
        *)
            echo "Usage: $0 [all|llm|tts]"
            echo
            echo "Examples:"
            echo "  $0       # starts LLM and TTS"
            echo "  $0 all   # starts LLM and TTS"
            echo "  $0 llm   # starts only LLM"
            echo "  $0 tts   # starts only TTS"
            exit 1
            ;;
    esac

    wait "${children[@]}"
}

trap cleanup EXIT INT TERM
main
