from __future__ import annotations

from contextlib import suppress

from .config import parse_args, prepare_runtime_environment


class RuntimeApp:
    def __init__(self):
        self.config = parse_args()

        from .runtime import HeadlessPalmClient
        from .transport import MessageQueue, PipelineWebSocketClient

        self.feature_queue = MessageQueue(maxlen=256)
        self.pipeline_client = PipelineWebSocketClient(self.config.transport.pipeline_ws_url, self.feature_queue)
        self.runtime = HeadlessPalmClient(
            self.config.runtime,
            feature_queue=self.feature_queue,
            pipeline_status_provider=self.pipeline_client.snapshot,
        )
        self.pipeline_client.command_handler = self.runtime.handle_command

    def run(self) -> None:
        try:
            self.runtime.start()
            self.pipeline_client.start()

            print(f"Palmprint runtime websocket target: {self.config.transport.pipeline_ws_url}")
            print("Start debug_server.py on the PC if you want the debug UI and local AI websocket server.")
            with suppress(KeyboardInterrupt):
                self._wait_forever()
        finally:
            self.pipeline_client.stop()
            self.runtime.stop()

    def _wait_forever(self) -> None:
        import time

        while True:
            time.sleep(1.0)


def main() -> None:
    prepare_runtime_environment()
    RuntimeApp().run()
