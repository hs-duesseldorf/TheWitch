#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

SERVER_DIR = Path(__file__).resolve().parent
ROOT = SERVER_DIR.parents[2]
OLLAMA_BIND_HOST = "0.0.0.0"
OLLAMA_KEEP_ALIVE = "24h"
OLLAMA_NUM_PARALLEL = "1"
OLLAMA_FLASH_ATTENTION = "1"


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


def api_host(host: str) -> str:
    bind_host, _, port = host.rpartition(":")
    if bind_host in ("", "0.0.0.0", "::"):
        bind_host = "127.0.0.1"
    return f"{bind_host}:{port}" if port else bind_host


def is_ollama_ready(host: str) -> bool:
    url = f"http://{api_host(host)}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def wait_for_ollama(
    host: str,
    *,
    process: subprocess.Popen[str] | None = None,
    timeout: float = 60.0,
) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://{api_host(host)}/api/tags"

    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"Ollama process exited with code {process.returncode} before {url} became ready"
            )
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


def wait_for_stop_signal() -> signal.Signals:
    stop = threading.Event()
    received = signal.SIGTERM

    def stop_with(signum: int, _frame: object) -> None:
        nonlocal received
        received = signal.Signals(signum)
        stop.set()

    previous_handlers: dict[signal.Signals, object] = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, stop_with)

    try:
        stop.wait()
        return received
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


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

    port = os.environ["WITCH_LLM_PORT"]
    ollama_host = f"{OLLAMA_BIND_HOST}:{port}"

    model = os.environ["WITCH_OLLAMA_MODEL"].strip()

    env = os.environ.copy()
    env.update(
        {
            "OLLAMA_HOST": ollama_host,
            "OLLAMA_KEEP_ALIVE": OLLAMA_KEEP_ALIVE,
            "OLLAMA_NUM_PARALLEL": OLLAMA_NUM_PARALLEL,
            "OLLAMA_FLASH_ATTENTION": OLLAMA_FLASH_ATTENTION,
            "OLLAMA_CONTEXT_LENGTH": os.environ["LLM_MAX_MODEL_LEN"],
        }
    )

    log(f"[run_llm] Starting Ollama on {ollama_host}")
    log(f"[run_llm] Ollama model={model}")

    if is_ollama_ready(ollama_host):
        log(f"[run_llm] Reusing existing Ollama at http://{api_host(ollama_host)}")
        run_ollama_command(["pull", model], env)
        log(f"[run_llm] OpenAI-compatible API ready: http://{ollama_host}/v1")
        log(f"[run_llm] Native Ollama API ready: http://{ollama_host}")
        stop_signal = wait_for_stop_signal()
        if stop_signal == signal.SIGINT:
            log("[run_llm] Leaving existing Ollama running")
            raise SystemExit(130)
        log("[run_llm] Leaving existing Ollama running")
        raise SystemExit(0)

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
        wait_for_ollama(ollama_host, process=process)

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
