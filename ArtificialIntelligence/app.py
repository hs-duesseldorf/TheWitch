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


def http_url(host_var: str, port_var: str) -> str:
    return f"http://{os.getenv(host_var)}:{os.getenv(port_var)}"


class App:
    def __init__(self):
        ws_port = int(os.getenv("WITCH_AI_PORT"))
        self.ws_server = WebSocketServer(host="0.0.0.0", port=ws_port)
        self.state_machine = WitchStateMachine()
        self.runtime = WitchRuntime(
            ws_server=self.ws_server,
            state_machine=self.state_machine,
            llm=LLMClient(
                base_url=http_url("WITCH_LLM_HOST", "WITCH_LLM_PORT"),
                model=os.getenv("WITCH_LLM_MODEL"),
            ),
            tts=TTSClient(
                base_url=http_url("WITCH_TTS_HOST", "WITCH_TTS_PORT"),
                speaker_wav=None,
                language="de",
            ),
        )

    async def run(self) -> None:
        await self.ws_server.start()


def run_websocket_server() -> None:
    asyncio.run(App().run())
