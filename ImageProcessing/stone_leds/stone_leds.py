from __future__ import annotations

from contextlib import suppress


class StoneLeds:
    def __init__(self, pin: int):
        self.pin = pin
        self.gpio = self._load_gpio()
        self.is_open = False

    def open(self):
        if self.is_open:
            return
        self.gpio.setmode(self.gpio.BOARD)
        self.gpio.setup(self.pin, self.gpio.OUT)
        self.gpio.output(self.pin, self.gpio.LOW)
        self.is_open = True

    def show_absent(self):
        self._set(False)

    def show_present(self):
        self._set(False)

    def show_seated(self):
        self._set(True)

    def close(self):
        if not self.is_open:
            return
        with suppress(Exception):
            self.gpio.output(self.pin, self.gpio.LOW)
        with suppress(Exception):
            self.gpio.cleanup(self.pin)
        self.is_open = False

    def _set(self, enabled: bool):
        if not self.is_open:
            return
        self.gpio.output(self.pin, self.gpio.HIGH if enabled else self.gpio.LOW)

    @staticmethod
    def _load_gpio():
        try:
            import Jetson.GPIO as gpio
        except ImportError as exc:
            raise RuntimeError(
                "Jetson.GPIO is unavailable. Enable WITCH_SEAT_SENSOR_OVERRIDE=true when not running on a Jetson."
            ) from exc
        return gpio
