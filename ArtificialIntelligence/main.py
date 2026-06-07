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

def _llm_model_name() -> str:
    return os.environ["WITCH_OLLAMA_MODEL"].strip()


def _get_llm_url() -> str:
    return f"http://{os.environ['WITCH_LLM_HOST']}:{os.environ['WITCH_LLM_PORT']}"


def _get_tts_url() -> str:
    return f"http://{os.environ['WITCH_TTS_HOST']}:{os.environ['WITCH_TTS_PORT']}"


class App:
    def __init__(self):
        ws_port = int(os.environ["WITCH_AI_PORT"])
        self.ws_server = WebSocketServer(host="0.0.0.0", port=ws_port)
        self.state_machine = WitchStateMachine()
        self.runtime = WitchRuntime(
            ws_server=self.ws_server,
            state_machine=self.state_machine,
            llm=LLMClient(
                base_url=_get_llm_url(),
                model=_llm_model_name(),
            ),
            tts=TTSClient(base_url=_get_tts_url()),
            audio_prebuffer_seconds=float(os.environ["WITCH_AUDIO_PREBUFFER_SECONDS"]),
            speaker_delay_seconds=float(os.environ["WITCH_SPEAKER_DELAY_SECONDS"]),
        )


def main() -> None:
    app = App()
    _set_debug_runtime(app.ws_server, app.state_machine, app.runtime)
    debug_thread = threading.Thread(target=_run_debug_ui, daemon=True)
    debug_thread.start()

    asyncio.run(app.ws_server.start())


if __name__ == "__main__":
    main()
