from __future__ import annotations

import os
import time
from contextlib import suppress

from palmprint_client.runtime import HeadlessPalmClient
from palmprint_client.transport import WebSocketClient


class App:
    def __init__(self):
        ai_ws_base_url = os.getenv("IP_AI_BASE_URL").strip().rstrip("/")

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

    def run(self) -> None:
        try:
            self.event_client.start()
            self.video_client.start()
            self.roi_client.start()
            self.runtime.start()

            with suppress(KeyboardInterrupt):
                self._wait_forever()
        finally:
            self.runtime.stop()
            self.roi_client.stop()
            self.video_client.stop()
            self.event_client.stop()

    def _wait_forever(self) -> None:
        while True:
            time.sleep(1.0)


if __name__ == "__main__":
    App().run()
