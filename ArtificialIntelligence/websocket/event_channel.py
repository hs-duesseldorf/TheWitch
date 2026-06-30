from __future__ import annotations

import logging
from typing import Any

import msgspec

from ArtificialIntelligence.websocket.message_parser import decode_event
from ArtificialIntelligence.websocket.websocket_server import WebSocketServer
from shared.events import WitchEvent

logger = logging.getLogger(__name__)


class EventChannel:
    """Wraps a WebSocketServer to send and receive WitchEvents on a single path.

    Handles decoding incoming messages into events and encoding outgoing events
    into JSON, stamping each one with an origin so receivers know where it came from.
    """

    def __init__(self, ws_server: WebSocketServer, path: str, default_origin: str, decode_source: str):
        self.ws_server = ws_server
        # WebSocket path this channel operates on (used to scope broadcasts).
        self.path = path
        # Origin tag applied to outgoing events when none is provided explicitly.
        self.default_origin = default_origin
        # Source label passed to the decoder for incoming messages.
        self.decode_source = decode_source

    def decode(self, message: str | bytes) -> WitchEvent | None:
        # Parse a raw incoming WebSocket message into a WitchEvent (or None if invalid).
        return decode_event(message, source=self.decode_source)

    async def broadcast(self, event: WitchEvent, *, exclude: Any | None = None, origin: str | None = None):
        # Send the event to every connection on this path, optionally skipping one (e.g. the sender).
        await self.ws_server.broadcast(self._encode(event, origin), path=self.path, exclude=exclude)

    async def send_to(self, connection: Any, event: WitchEvent, *, origin: str | None = None):
        # Send the event to a single specific connection.
        await self.ws_server.send_to(connection, self._encode(event, origin))

    def _encode(self, event: WitchEvent, origin: str | None = None) -> str:
        # Convert the event to a plain dict, then to a JSON string for transport.
        data = msgspec.to_builtins(event)
        # Resolve the origin: keep an existing one, else use the override, else the default.
        data["origin"] = data.get("origin") or origin or self.default_origin
        return msgspec.json.encode(data).decode("utf-8")
