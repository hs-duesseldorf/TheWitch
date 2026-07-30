from __future__ import annotations

import logging
from typing import Any

import msgspec

from ArtificialIntelligence.websocket.message_parser import decode_event
from ArtificialIntelligence.websocket.websocket_server import WebSocketServer
from shared.events import WitchEvent

logger = logging.getLogger(__name__)


class EventChannel:
    def __init__(self, ws_server: WebSocketServer, path: str, default_origin: str, decode_source: str):
        self.ws_server = ws_server
        self.path = path
        self.default_origin = default_origin
        self.decode_source = decode_source

    def decode(self, message: str | bytes) -> WitchEvent | None:
        return decode_event(message, source=self.decode_source)

    async def broadcast(self, event: WitchEvent, *, exclude: Any | None = None, origin: str | None = None):
        await self.ws_server.broadcast(self._encode(event, origin), path=self.path, exclude=exclude)

    async def send_to(self, connection: Any, event: WitchEvent, *, origin: str | None = None):
        await self.ws_server.send_to(connection, self._encode(event, origin))

    def _encode(self, event: WitchEvent, origin: str | None = None) -> str:
        data = msgspec.to_builtins(event)
        data["origin"] = data.get("origin") or origin or self.default_origin
        return msgspec.json.encode(data).decode("utf-8")
