from __future__ import annotations

from typing import Any

import msgspec

from message_parser import decode_event
from shared.events import WitchEvent
from websocket_server.websocket_server import WebSocketServer


class EventChannel:
    def __init__(
        self,
        *,
        ws_server: WebSocketServer,
        path: str,
        default_origin: str,
        decode_source: str,
    ) -> None:
        self._ws_server = ws_server
        self._path = path
        self._default_origin = default_origin
        self._decode_source = decode_source

    def decode(self, message: str | bytes) -> WitchEvent | None:
        return decode_event(message, source=self._decode_source)

    async def broadcast(
        self,
        event: WitchEvent,
        *,
        exclude: Any | None = None,
        origin: str | None = None,
    ) -> None:
        await self._ws_server.broadcast(
            self._encode(event, origin=origin),
            path=self._path,
            exclude=exclude,
        )

    async def send_to(self, connection: Any, event: WitchEvent, *, origin: str | None = None) -> None:
        await self._ws_server.send_to(connection, self._encode(event, origin=origin))

    def _encode(self, event: WitchEvent, *, origin: str | None = None) -> str:
        data = msgspec.to_builtins(event)
        data["origin"] = data.get("origin") or origin or self._default_origin
        return msgspec.json.encode(data).decode("utf-8")
