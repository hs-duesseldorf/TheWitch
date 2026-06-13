from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import suppress

from dotenv import load_dotenv

from ImageProcessing.observation_stabilizer import ObservationStabilizer
from ImageProcessing.palm_processing.transport import WebSocketClient
from ImageProcessing.seat_sensor.seat_sensor_override import SeatSensorOverride
from ImageProcessing.stone_leds import StoneLeds
from shared.events import PersonEvent, PersonTrigger, WitchEvent

load_dotenv()
logger = logging.getLogger("image-processing")


class Runtime:
    def __init__(self):
        ai_base_url = (
            f"ws://{os.environ['WITCH_AI_HOST']}:{os.environ['WITCH_AI_PORT']}"
        )
        self.event_client = WebSocketClient(f"{ai_base_url}/ws/ip-ai", max_queue=64)
        self.video_client = WebSocketClient(
            f"{ai_base_url}/ws/ip-ai-video", max_queue=4
        )
        self.roi_client = WebSocketClient(f"{ai_base_url}/ws/ip-roi", max_queue=4)

        self.stabilizer = ObservationStabilizer(
            float(os.environ["WITCH_EVENT_STABILITY_SECONDS"])
        )

        self.heartbeat_seconds = float(
            os.environ.get("WITCH_EVENT_HEARTBEAT_SECONDS", "0.5")
        )

        self.hand_tracker = self._create_hand_tracker()
        self.seat_sensor = self._create_seat_sensor()
        self.stone_leds = self._create_leds()

        self.stop_event = threading.Event()

        self._last_send: dict[str, float] = {}
        self._type_interval = 1.0

    def run(self):
        try:
            self.start()
            logger.info("ImageProcessing started")

            with suppress(KeyboardInterrupt):
                while not self.stop_event.wait(self.heartbeat_seconds):
                    self.publish_current_events()
        finally:
            self.stop()
            logger.info("ImageProcessing stopped")

    def start(self):
        self.event_client.start()
        self.video_client.start()
        self.roi_client.start()

        with suppress(Exception):
            self.stone_leds.open()

        self.hand_tracker.start(self.handle_observation)
        self.seat_sensor.start(self.handle_observation)

    def stop(self):
        self.stop_event.set()

        with suppress(Exception):
            self.hand_tracker.stop()

        with suppress(Exception):
            self.seat_sensor.stop()

        with suppress(Exception):
            self.stone_leds.close()

        self.roi_client.stop()
        self.video_client.stop()
        self.event_client.stop()

    def handle_observation(self, event: WitchEvent):
        changed = self.stabilizer.observe(event, time.monotonic())

        if changed:
            self._apply_leds(event)

    def publish_current_events(self):
        now = time.monotonic()

        for event in self.stabilizer.current_events:
            key = self._event_key(event)

            if now - self._last_send.get(key, 0.0) >= self._type_interval:
                self.event_client.send_message(event)
                self._last_send[key] = now

    def _event_key(self, event: WitchEvent) -> str:
        if hasattr(event, "trigger"):
            return f"{type(event).__name__}:{event.trigger}"

        return repr(event)

    def _apply_leds(self, event: WitchEvent):
        if not isinstance(event, PersonEvent):
            return

        if event.trigger is PersonTrigger.SEATED:
            self.stone_leds.show_seated()
        elif event.trigger is PersonTrigger.DETECTED:
            self.stone_leds.show_present()
        else:
            self.stone_leds.show_absent()

    def _create_seat_sensor(self):
        if os.environ.get("WITCH_SEAT_SENSOR_OVERRIDE", "").strip().lower() == "true":
            return SeatSensorOverride()

        from ImageProcessing.seat_sensor.seat_sensor import SeatSensor

        return SeatSensor()

    def _create_hand_tracker(self):
        from ImageProcessing.hand_tracker import HandTracker

        return HandTracker(self.video_client, self.roi_client)

    def _create_leds(self):
        if os.environ.get("WITCH_SEAT_SENSOR_OVERRIDE", "").strip().lower() == "true":
            return _NoopStoneLeds()

        try:
            return StoneLeds(int(os.environ["WITCH_LED_PIN"]))
        except Exception as exc:
            logger.warning("Stone LEDs unavailable: %s", exc)
            return _NoopStoneLeds()


class _NoopStoneLeds:
    def open(self):
        return None

    def show_absent(self):
        return None

    def show_present(self):
        return None

    def show_seated(self):
        return None

    def close(self):
        return None
