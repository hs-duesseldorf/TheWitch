from __future__ import annotations

import logging

import msgspec

from shared.events import WitchEvent

logger = logging.getLogger(__name__)


def decode_event(message: str | bytes, *, source: str) -> WitchEvent | None:
    if isinstance(message, bytes):
        return None
    try:
        return msgspec.json.decode(message, type=WitchEvent)
    except (msgspec.DecodeError, TypeError, ValueError) as exc:
        logger.warning("Failed to decode %s message: %s", source, exc)
        return None
