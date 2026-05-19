from __future__ import annotations

import os
import subprocess
import threading

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    llm_model = os.getenv("WITCH_LLM_MODEL", "qwen/qwen3-7b-instruct")
    tts_model = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    llm_port = os.getenv("WITCH_LLM_PORT", "8082")
    tts_port = os.getenv("WITCH_TTS_PORT", "8083")

    llm_process = subprocess.Popen([
        "vllm", "serve", llm_model,
        "--host", "0.0.0.0",
        "--port", llm_port,
        "--trust-remote-code",
    ])

    tts_process = subprocess.Popen([
        "vllm", "serve", tts_model,
        "--omni",
        "--host", "0.0.0.0",
        "--port", tts_port,
        "--trust-remote-code",
        "--enforce-eager",
    ])

    try:
        llm_process.wait()
    except KeyboardInterrupt:
        llm_process.terminate()
        tts_process.terminate()


if __name__ == "__main__":
    main()