from __future__ import annotations

import logging

import msgspec

from shared.events import WitchEvent

logger = logging.getLogger(__name__)


def decode_event(message: str | bytes, *, source: str) -> WitchEvent | None:
    """Parse a raw WebSocket message into a WitchEvent.

    Returns None instead of raising if the message is not valid utf-8 or not a
    well-formed WitchEvent, so a single bad message can't crash the receiver.
    The `source` label is only used for logging to identify where the message came from.
    """
    # Binary frames arrive as bytes; decode them to text before JSON parsing.
    if isinstance(message, bytes):
        try:
            message = message.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning("Failed to decode %s binary message as utf-8", source)
            return None
    # Decode the JSON text directly into a typed WitchEvent.
    try:
        return msgspec.json.decode(message, type=WitchEvent)
    except (msgspec.DecodeError, TypeError, ValueError) as exc:
        # Malformed JSON or a payload that doesn't match the WitchEvent schema.
        logger.warning("Failed to decode %s message: %s -- raw: %s", source, exc, message)
        return None
