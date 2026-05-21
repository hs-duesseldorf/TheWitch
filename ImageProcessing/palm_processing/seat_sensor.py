from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

from .transport import WebSocketClient
from shared.events import ErrorEvent, PersonEvent, PersonTrigger


@dataclass(frozen=True, slots=True)
class SeatSensorConfig:
    i2c_address: int = 0x29
    threshold_mm: int = 600
    reset_threshold_mm: int = 750
    poll_ms: int = 100
    required_hits: int = 3


class VL53L0XDistanceSensor:
    def __init__(self, *, i2c_address: int):
        try:
            import board
            import busio
            import adafruit_vl53l0x
        except ImportError as exc:
            raise RuntimeError(
                "VL53L0X support is missing. Install adafruit-blinka and "
                "adafruit-circuitpython-vl53l0x on the Jetson."
            ) from exc

        i2c = busio.I2C(board.SCL, board.SDA)
        self._sensor = adafruit_vl53l0x.VL53L0X(i2c, address=i2c_address)

    def read_distance_mm(self) -> int | None:
        distance = self._sensor.range
        if distance is None:
            return None
        return int(distance)


class SeatPresenceMonitor:
    def __init__(
        self,
        *,
        event_client: WebSocketClient,
        config: SeatSensorConfig | None = None,
    ):
        self.config = config or SeatSensorConfig()
        self.event_client = event_client
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None

    def start(self) -> None:
        if self.worker is not None:
            return
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._run, name="vl53l0x-seat-monitor", daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.worker is not None and self.worker.is_alive() and self.worker is not threading.current_thread():
            self.worker.join(timeout=2.0)
        self.worker = None

    def _run(self) -> None:
        if os.getenv("WITCH_SEAT_SENSOR_OVERRIDE") == "true":
            self._publish_person_seated()
            return

        try:
            sensor = VL53L0XDistanceSensor(i2c_address=self.config.i2c_address)
        except Exception as exc:
            self._publish_error(str(exc))
            return

        seated = False
        seated_hits = 0
        free_hits = 0
        poll_s = max(self.config.poll_ms / 1000.0, 0.02)
        required_hits = max(1, self.config.required_hits)

        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                distance_mm = sensor.read_distance_mm()
            except Exception as exc:
                self._publish_error(f"VL53L0X read failed: {exc}")
                if self.stop_event.wait(poll_s):
                    break
                continue

            if distance_mm is not None:
                if distance_mm <= self.config.threshold_mm:
                    seated_hits += 1
                    free_hits = 0
                elif distance_mm >= self.config.reset_threshold_mm:
                    free_hits += 1
                    seated_hits = 0
                else:
                    seated_hits = 0
                    free_hits = 0

                if not seated and seated_hits >= required_hits:
                    seated = True
                    self._publish_person_seated()
                elif seated and free_hits >= required_hits:
                    seated = False

            elapsed = time.monotonic() - started
            if self.stop_event.wait(max(0.0, poll_s - elapsed)):
                break

    def _publish_person_seated(self) -> None:
        self.event_client.send_message(
            PersonEvent(
                trigger=PersonTrigger.DETECTED,
            )
        )

    def _publish_error(self, message: str) -> None:
        self.event_client.send_message(
            ErrorEvent(
                message=f"Seat sensor error: {message}",
            )
        )
