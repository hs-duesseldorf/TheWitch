from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shared.events import HandEvent, WitchEvent


@dataclass
class _State:
    observed: WitchEvent | None = None
    observed_since: float = 0.0
    current: WitchEvent | None = None


class ObservationStabilizer:
    def __init__(self, stability_seconds: float):
        self.stability_seconds = max(0.0, stability_seconds)
        self._states: dict[str, _State] = {}

    def observe(self, event: WitchEvent, now: float) -> bool:
        cls_name = event.__class__.__name__
        state = self._states.get(cls_name)

        if state is None or not self._same_event(event, state.observed):
            self._states[cls_name] = _State(observed=event, observed_since=now)
            return False

        if (now - state.observed_since) + 1e-9 < self.stability_seconds:
            return False

        if self._same_event(event, state.current):
            return False

        state.current = event
        return True

    @property
    def current_events(self) -> list[WitchEvent]:
        return [
            state.current
            for state in self._states.values()
            if state.current is not None
        ]

    @staticmethod
    def _same_event(a: WitchEvent | None, b: WitchEvent | None) -> bool:
        if a is None and b is None:
            return True

        if a is None or b is None:
            return False

        if isinstance(a, HandEvent) and isinstance(b, HandEvent):
            return a.hand == b.hand and a.trigger == b.trigger

        return a == b
