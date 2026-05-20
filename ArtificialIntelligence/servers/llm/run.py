#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

SERVER_DIR = Path(__file__).resolve().parent
ROOT = SERVER_DIR.parents[2]


def remote_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request) as response:
            length = response.headers.get("Content-Length") or response.headers.get("X-Linked-Size")
    except Exception:
        return None
    return int(length) if length and length.isdigit() else None


def download_file(url: str, destination: Path, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = remote_size(url)
    if destination.exists() and expected_size is not None:
        actual_size = destination.stat().st_size
        if actual_size == expected_size:
            return
        print(
            f"[run_llm] Existing {label} has size {actual_size}, expected {expected_size}; re-downloading...",
            flush=True,
        )
    elif destination.exists():
        return

    part = destination.with_suffix(destination.suffix + ".part")
    if part.exists():
        part.unlink()
    print(f"[run_llm] Downloading {label}...", flush=True)
    with urllib.request.urlopen(url) as response, part.open("wb") as output:
        shutil.copyfileobj(response, output)

    if expected_size is not None:
        actual_size = part.stat().st_size
        if actual_size != expected_size:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"{label} download incomplete: got {actual_size} bytes, expected {expected_size}")

    part.replace(destination)


def run() -> None:
    load_dotenv(ROOT / ".env")

    data_dir = SERVER_DIR / ".venv"
    llama_dir = data_dir / "llama-cpp"
    llama_server = llama_dir / "llama-server"
    models_dir = data_dir / "models"

    os.environ.update({
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "8",
        "TORCH_NUM_THREADS": "8",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    })

    if not llama_server.exists():
        tag = "b9222"
        llama_dir.mkdir(parents=True, exist_ok=True)
        archive = llama_dir / f"llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz"
        url = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{archive.name}"
        download_file(url, archive, f"llama.cpp {tag}")
        subprocess.run(["tar", "-xzf", str(archive), "-C", str(llama_dir), "--strip-components=1"], check=True)
        subprocess.check_call(["chmod", "+x", str(llama_server)])

    repo, file = os.environ["WITCH_LLM_HF"].split(":", 1)
    model = models_dir / file
    download_file(f"https://huggingface.co/{repo}/resolve/main/{file}", model, "LLM model")

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
