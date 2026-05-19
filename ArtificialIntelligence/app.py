import asyncio
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from .clients.llm_client import LLMClient
from .clients.tts_client import TTSClient
from .runtime import WitchRuntime
from .state_machine.state_machine import WitchStateMachine
from .websocket_server.websocket_server import WebSocketServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai")
logging.getLogger("transitions").setLevel(logging.ERROR)

def llm_model_name() -> str:
    llm_hf = os.environ["WITCH_LLM_HF"]
    if ":" not in llm_hf:
        raise RuntimeError(
            "WITCH_LLM_HF must be repo:file, e.g. "
            "unsloth/Qwen3-4B-GGUF:Qwen3-4B-Q4_K_M.gguf"
        )

    repo, model_file = llm_hf.split(":", 1)
    if not repo or not model_file:
        raise RuntimeError(f"Invalid WITCH_LLM_HF: {llm_hf}")
    return model_file


LLM_MODEL = llm_model_name()


def http_url(host_var: str, port_var: str) -> str:
    return f"http://{os.environ[host_var]}:{os.environ[port_var]}"


def get_llm_url() -> str:
    host = os.getenv("WITCH_LLM_HOST", "")
    if host in ("localhost", "127.0.0.1", "ai", ""):
        return http_url("WITCH_LLM_HOST", "WITCH_LLM_PORT")
    return http_url("WITCH_LLM_HOST", "WITCH_LLM_PORT_EXT")


def get_tts_url() -> str:
    host = os.getenv("WITCH_TTS_HOST", "")
    if host in ("localhost", "127.0.0.1", "ai", ""):
        return http_url("WITCH_TTS_HOST", "WITCH_TTS_PORT")
    return http_url("WITCH_TTS_HOST", "WITCH_TTS_PORT_EXT")


class App:
    def __init__(self):
        ws_port = int(os.getenv("WITCH_AI_PORT"))
        self.ws_server = WebSocketServer(host="0.0.0.0", port=ws_port)
        self.state_machine = WitchStateMachine()
        self.runtime = WitchRuntime(
            ws_server=self.ws_server,
            state_machine=self.state_machine,
            llm=LLMClient(
                base_url=get_llm_url(),
                model=LLM_MODEL,
            ),
            tts=TTSClient(
                base_url=get_tts_url(),
                model=os.environ["WITCH_TTS_MODEL"],
            ),
        )

    async def run(self) -> None:
        await self.ws_server.start()


def run_websocket_server() -> None:
    asyncio.run(App().run())
