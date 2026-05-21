import asyncio
import logging
import os
import threading

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai")
logging.getLogger("transitions").setLevel(logging.ERROR)

from .clients.llm_client import LLMClient
from .clients.tts_client import TTSClient
from .debug_ui import set_runtime as _set_debug_runtime, run as _run_debug_ui
from .runtime import WitchRuntime
from .state_machine.state_machine import WitchStateMachine
from .websocket_server.websocket_server import WebSocketServer

DEFAULT_OLLAMA_MODEL = "hf.co/unsloth/Qwen3-4B-GGUF:Qwen3-4B-Q4_K_M.gguf"


def _llm_model_name() -> str:
    return os.getenv("WITCH_OLLAMA_MODEL", "").strip() or os.getenv("OLLAMA_MODEL", "").strip() or DEFAULT_OLLAMA_MODEL


def _get_llm_url() -> str:
    return f"http://{os.environ['WITCH_LLM_HOST']}:{os.environ['WITCH_LLM_PORT']}"


def _get_tts_url() -> str:
    return f"http://{os.environ['WITCH_TTS_HOST']}:{os.environ['WITCH_TTS_PORT']}"


class App:
    def __init__(self):
        ws_port = int(os.getenv("WITCH_AI_PORT"))
        self.ws_server = WebSocketServer(host="0.0.0.0", port=ws_port)
        self.state_machine = WitchStateMachine()
        self.runtime = WitchRuntime(
            ws_server=self.ws_server,
            state_machine=self.state_machine,
            llm=LLMClient(
                base_url=_get_llm_url(),
                model=_llm_model_name(),
            ),
            tts=TTSClient(
                base_url=_get_tts_url(),
                model=os.environ["WITCH_TTS_MODEL"],
            ),
        )


def main() -> None:
    app = App()
    _set_debug_runtime(app.ws_server, app.state_machine, app.runtime)
    debug_thread = threading.Thread(target=_run_debug_ui, daemon=True)
    debug_thread.start()

    asyncio.run(app.ws_server.start())


if __name__ == "__main__":
    main()
