from __future__ import annotations

from contextlib import suppress


class SeatSensorLed:
    def __init__(self, *, pin: int):
        self._pin = pin
        self._gpio = self._load_gpio()
        self._is_open = False
        self._seated = False

    def open(self) -> None:
        if self._is_open:
            return
        self._gpio.setmode(self._gpio.BOARD)
        self._gpio.setup(self._pin, self._gpio.OUT)
        self._gpio.output(self._pin, self._gpio.LOW)
        self._seated = False
        self._is_open = True

    def set_seated(self, seated: bool) -> None:
        if not self._is_open:
            raise RuntimeError("Seat sensor LED is not open")
        if seated == self._seated:
            return
        self._gpio.output(self._pin, self._gpio.HIGH if seated else self._gpio.LOW)
        self._seated = seated

    def close(self) -> None:
        if not self._is_open:
            return
        with suppress(Exception):
            self._gpio.output(self._pin, self._gpio.LOW)
        with suppress(Exception):
            self._gpio.cleanup(self._pin)
        self._seated = False
        self._is_open = False

    @staticmethod
    def _load_gpio():
        try:
            import Jetson.GPIO as gpio
        except ImportError as exc:
            raise RuntimeError(
                "Jetson.GPIO is unavailable. Enable WITCH_SEAT_SENSOR_OVERRIDE=true when not running on a Jetson."
            ) from exc
        return gpio
