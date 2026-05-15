from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv


def find_ollama() -> Path:
    configured = os.getenv("OLLAMA_BINARY")
    if configured:
        path = Path(configured)
        if path.exists():
            return path
        raise RuntimeError(f"OLLAMA_BINARY does not exist: {path}")

    system_ollama = shutil.which("ollama")
    if system_ollama:
        return Path(system_ollama)

    raise RuntimeError(
        "Ollama is not installed or not on PATH. Install Ollama for your OS, "
        "or set OLLAMA_BINARY=/path/to/ollama."
    )


def main() -> None:
    load_dotenv()

    port = os.getenv("WITCH_LLM_PORT", "8082")
    model = os.getenv("WITCH_LLM_MODEL")
    if not model:
        raise RuntimeError("WITCH_LLM_MODEL is not set")

    server_env = os.environ.copy()
    server_env["OLLAMA_HOST"] = f"0.0.0.0:{port}"

    client_env = os.environ.copy()
    client_env["OLLAMA_HOST"] = f"127.0.0.1:{port}"

    ollama = str(find_ollama())
    server = subprocess.Popen([ollama, "serve"], env=server_env)

    def stop_server(*_: object) -> None:
        server.terminate()

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)

    try:
        time.sleep(2)
        subprocess.run([ollama, "pull", model], env=client_env, check=True)
        server.wait()
    finally:
        if server.poll() is None:
            server.terminate()
            server.wait(timeout=10)


if __name__ == "__main__":
    main()
