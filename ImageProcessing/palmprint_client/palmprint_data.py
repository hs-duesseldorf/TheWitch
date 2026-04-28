from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(slots=True)
class PalmprintData:
    """Typed representation of a palmprint message payload."""

    status: str
    hand: str
    proportions: dict[str, float]
    vector: list[float]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PalmprintData":
        # Validate the JSON shape before converting values into Python types.
        if not isinstance(payload, Mapping):
            raise TypeError("Palmprint payload must be a mapping")

        status = payload.get("status")
        hand = payload.get("hand")
        proportions = payload.get("proportions")
        vector = payload.get("vector")

        if not isinstance(status, str) or not status.strip():
            raise ValueError("Palmprint payload requires a non-empty status")
        if not isinstance(hand, str) or not hand.strip():
            raise ValueError("Palmprint payload requires a non-empty hand")
        if not isinstance(proportions, Mapping):
            raise ValueError("Palmprint payload requires proportions as an object")
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
            raise ValueError("Palmprint payload requires vector as an array")

        return cls(
            status=status.strip(),
            hand=hand.strip(),
            proportions={str(key): float(value) for key, value in proportions.items()},
            vector=[float(value) for value in vector],
        )

    @classmethod
    def from_json(cls, raw_json: str) -> "PalmprintData":
        # Decode a JSON string and reuse the dictionary validator.
        payload = json.loads(raw_json)
        if not isinstance(payload, dict):
            raise ValueError("Palmprint JSON must decode to an object")
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        # Convert the dataclass back into a JSON-ready dictionary.
        return asdict(self)

    def to_json(self) -> str:
        # Emit a JSON string for websocket transport or persistence.
        return json.dumps(self.to_dict(), ensure_ascii=False)



def build_palmprint_data(
    *,
    status: str,
    hand: str,
    proportions: Mapping[str, float],
    vector: Sequence[float],
) -> PalmprintData:
    # Normalize numeric values so downstream consumers get consistent precision.
    return PalmprintData(
        status=status,
        hand=hand,
        proportions={key: round(float(value), 6) for key, value in proportions.items()},
        vector=[round(float(value), 6) for value in vector],
    )