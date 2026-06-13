from __future__ import annotations

import logging

import msgspec

from shared.events import WitchEvent

logger = logging.getLogger(__name__)


def decode_event(message: str | bytes, *, source: str) -> WitchEvent | None:
    if isinstance(message, bytes):
        try:
            message = message.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning("Failed to decode %s binary message as utf-8", source)
            return None
    try:
        return msgspec.json.decode(message, type=WitchEvent)
    except (msgspec.DecodeError, TypeError, ValueError) as exc:
        logger.warning("Failed to decode %s message: %s -- raw: %s", source, exc, message)
        return None
