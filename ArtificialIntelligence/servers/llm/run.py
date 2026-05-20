#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

SERVER_DIR = Path(__file__).resolve().parent
ROOT = SERVER_DIR.parents[2]


def run() -> None:
    load_dotenv(ROOT / ".env")

    for key in ("WITCH_LLM_HF", "WITCH_LLM_PORT", "LLM_MAX_MODEL_LEN", "LLM_N_GPU_LAYERS"):
        if key not in os.environ:
            sys.exit(f"ERROR: {key} required")

    version = os.environ.get("LLAMA_VERSION", "9222")
    tag = f"b{version}"

    llama_dir = SERVER_DIR / "llama-cpp"
    llama_server = llama_dir / "llama-server"
    models_dir = SERVER_DIR / "models"

    os.environ.update({
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "1"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8"),
        "TORCH_NUM_THREADS": os.environ.get("TORCH_NUM_THREADS", "8"),
        "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "false"),
        "HF_HUB_DISABLE_SYMLINKS_WARNING": os.environ.get("HF_HUB_DISABLE_SYMLINKS_WARNING", "1"),
    })

    if not llama_server.exists():
        llama_dir.mkdir(parents=True, exist_ok=True)
        print(f"[run_llm] Downloading llama.cpp {tag}...", flush=True)
        archive = llama_dir / f"llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz"
        url = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{archive.name}"
        urllib.request.urlretrieve(url, archive)
        subprocess.run(["tar", "-xzf", str(archive), "-C", str(llama_dir)], check=True)
        subprocess.check_call(["chmod", "+x", str(llama_server)])

    repo, file = os.environ["WITCH_LLM_HF"].split(":", 1)
    model = models_dir / file
    if not model.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        print("[run_llm] Downloading LLM model...", flush=True)
        urllib.request.urlretrieve(f"https://huggingface.co/{repo}/resolve/main/{file}", model)

    print(f"[run_llm] Starting LLM: model={model} port={os.environ['WITCH_LLM_PORT']}", flush=True)
    threads = os.environ.get("LLM_THREADS", "8")
    os.execv(str(llama_server), [
        str(llama_server),
        "--model", str(model),
        "--host", "0.0.0.0",
        "--port", os.environ["WITCH_LLM_PORT"],
        "--alias", file,
        "--ctx-size", os.environ["LLM_MAX_MODEL_LEN"],
        "--n-gpu-layers", os.environ["LLM_N_GPU_LAYERS"],
        "--threads", threads,
        "--parallel", "1",
    ])


if __name__ == "__main__":
    run()