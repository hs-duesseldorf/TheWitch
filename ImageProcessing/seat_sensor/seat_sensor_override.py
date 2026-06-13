from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from shared.events import PersonEvent, PersonTrigger, WitchEvent

logger = logging.getLogger(__name__)

OVERRIDE_DETECTED_DELAY_SECONDS = 10
OVERRIDE_SEATED_DELAY_SECONDS = 10
OVERRIDE_POLL_SECONDS = 0.1


class SeatSensorOverride:
    def __init__(self):
        self._callback: Callable[[WitchEvent], None] | None = None
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None

    def start(self, callback: Callable[[WitchEvent], None]):
        if self.worker is not None:
            return
        self._callback = callback
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._run, name="seat-sensor-override", daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        if (
            self.worker is not None
            and self.worker.is_alive()
            and self.worker is not threading.current_thread()
        ):
            self.worker.join(timeout=2.0)
        self.worker = None

    def _run(self):
        logger.info(
            "Seat sensor override: detected after %ss, seated after a further %ss",
            OVERRIDE_DETECTED_DELAY_SECONDS,
            OVERRIDE_SEATED_DELAY_SECONDS,
        )
        started = time.monotonic()
        while not self.stop_event.is_set():
            self._emit(self._current_trigger(time.monotonic() - started))
            self.stop_event.wait(OVERRIDE_POLL_SECONDS)

    @staticmethod
    def _current_trigger(elapsed: float) -> PersonTrigger:
        if elapsed >= OVERRIDE_DETECTED_DELAY_SECONDS + OVERRIDE_SEATED_DELAY_SECONDS:
            return PersonTrigger.SEATED
        if elapsed >= OVERRIDE_DETECTED_DELAY_SECONDS:
            return PersonTrigger.DETECTED
        return PersonTrigger.ABSENT

    def _emit(self, trigger: PersonTrigger):
        if self._callback is not None:
            self._callback(PersonEvent(trigger=trigger))
