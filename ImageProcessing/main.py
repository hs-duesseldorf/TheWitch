from __future__ import annotations

import logging
import os
import time
from contextlib import suppress
import Jetson.GPIO as GPIO

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("image-processing")

from .seat_sensor import SeatPresenceMonitor
from .palm_processing.pipeline import HeadlessPalmClient
from .palm_processing.transport import WebSocketClient


def _ws_url(host_var: str, port_var: str) -> str:
    return f"ws://{os.environ[host_var]}:{os.environ[port_var]}"


def main() -> None:
    ai_ws_base_url = _ws_url("WITCH_AI_HOST", "WITCH_AI_PORT")

    event_client = WebSocketClient(f"{ai_ws_base_url}/ws/ip-ai", max_queue=64)
    video_client = WebSocketClient(f"{ai_ws_base_url}/ws/ip-ai-video", max_queue=4)
    roi_client = WebSocketClient(f"{ai_ws_base_url}/ws/ip-roi", max_queue=4)

    runtime = HeadlessPalmClient(
        event_client=event_client,
        video_client=video_client,
        roi_client=roi_client,
    )
    seat_monitor = SeatPresenceMonitor(event_client=event_client)
    led_pin = int(os.environ["WITCH_LED_PIN"])

    try:
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(led_pin, GPIO.OUT)
        GPIO.output(led_pin, GPIO.LOW)

        event_client.start()
        video_client.start()
        roi_client.start()
        seat_monitor.start()
        runtime.start()

        logger.info("ImageProcessing started")
        with suppress(KeyboardInterrupt):
            while True:
                time.sleep(1.0)
    finally:
        runtime.stop()
        seat_monitor.stop()
        roi_client.stop()
        video_client.stop()
        event_client.stop()
        logger.info("ImageProcessing stopped")


if __name__ == "__main__":
    main()
