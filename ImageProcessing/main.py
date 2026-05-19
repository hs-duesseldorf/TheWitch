from __future__ import annotations

import os
import time
from contextlib import suppress

from dotenv import load_dotenv

load_dotenv()

from .palm_processing.seat_sensor import SeatPresenceMonitor
from .palm_processing.pipeline import HeadlessPalmClient
from .palm_processing.transport import WebSocketClient


def ws_url(host_var: str, port_var: str) -> str:
    return f"ws://{os.environ[host_var]}:{os.environ[port_var]}"


class App:
    def __init__(self):
        ai_ws_base_url = ws_url("WITCH_AI_HOST", "WITCH_AI_PORT")

        pipeline_ws_url = f"{ai_ws_base_url}/ws/ip-ai"
        video_ws_url = f"{ai_ws_base_url}/ws/ip-ai-video"
        roi_ws_url = f"{ai_ws_base_url}/ws/ip-roi"

        self.event_client = WebSocketClient(pipeline_ws_url, max_queue=64)
        self.video_client = WebSocketClient(video_ws_url, max_queue=4)
        self.roi_client = WebSocketClient(roi_ws_url, max_queue=4)
        self.runtime = HeadlessPalmClient(
            event_client=self.event_client,
            video_client=self.video_client,
            roi_client=self.roi_client,
        )
        self.seat_monitor = SeatPresenceMonitor(
            event_client=self.event_client,
        )

    def run(self) -> None:
        try:
            self.event_client.start()
            self.video_client.start()
            self.roi_client.start()
            self.seat_monitor.start()
            self.runtime.start()

            with suppress(KeyboardInterrupt):
                self._wait_forever()
        finally:
            self.runtime.stop()
            self.seat_monitor.stop()
            self.roi_client.stop()
            self.video_client.stop()
            self.event_client.stop()

    def _wait_forever(self) -> None:
        while True:
            time.sleep(1.0)


if __name__ == "__main__":
    App().run()
