#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

SERVER_DIR = Path(__file__).resolve().parent
ROOT = SERVER_DIR.parents[2]
DEFAULT_OLLAMA_MODEL = "hf.co/unsloth/Qwen3-4B-GGUF:Qwen3-4B-Q4_K_M.gguf"


def log(message: str) -> None:
    print(message, flush=True)


def ollama_bin() -> str:
    path = shutil.which("ollama")
    if path:
        return path

    raise SystemExit(
        "ollama is required. Install it with:\n"
        "curl -fsSL https://ollama.com/install.sh | sh"
    )


def wait_for_ollama(host: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://{host}/api/tags"

    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
            time.sleep(0.5)

    raise RuntimeError(f"Ollama did not become ready at {url}: {last_error}")


def run_ollama_command(args: list[str], env: dict[str, str]) -> None:
    log(f"[run_llm] Running: ollama {' '.join(args)}")
    subprocess.run(["ollama", *args], cwd=ROOT, env=env, check=True)


def terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def run() -> None:
    load_dotenv(ROOT / ".env")

    ollama = ollama_bin()

    port = os.environ.get("WITCH_LLM_PORT", "10032")
    host = os.environ.get("OLLAMA_BIND_HOST", "0.0.0.0")
    ollama_host = f"{host}:{port}"

    model = (
        os.environ.get("WITCH_OLLAMA_MODEL", "").strip()
        or os.environ.get("OLLAMA_MODEL", "").strip()
        or DEFAULT_OLLAMA_MODEL
    )

    env = os.environ.copy()
    env.update(
        {
            "OLLAMA_HOST": ollama_host,
            "OLLAMA_KEEP_ALIVE": os.environ.get("OLLAMA_KEEP_ALIVE", "24h"),
            "OLLAMA_NUM_PARALLEL": os.environ.get("OLLAMA_NUM_PARALLEL", "1"),
            "OLLAMA_FLASH_ATTENTION": os.environ.get("OLLAMA_FLASH_ATTENTION", "1"),
        }
    )

    log(f"[run_llm] Starting Ollama on {ollama_host}")
    log(f"[run_llm] Ollama model={model}")

    process = subprocess.Popen(
        [ollama, "serve"],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
    )

    try:
        wait_for_ollama(ollama_host)

        run_ollama_command(["pull", model], env)

        log(f"[run_llm] OpenAI-compatible API ready: http://{ollama_host}/v1")
        log(f"[run_llm] Native Ollama API ready: http://{ollama_host}")

        raise SystemExit(process.wait())

    except KeyboardInterrupt:
        log("[run_llm] Stopping Ollama")
        terminate(process)
        raise SystemExit(130)

    except Exception:
        terminate(process)
        raise


if __name__ == "__main__":
    run()
