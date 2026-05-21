import asyncio
import logging
from typing import Awaitable, Callable
from urllib.parse import urlsplit

from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK
try:
    from websockets.asyncio.server import ServerConnection, serve
except ModuleNotFoundError:
    from websockets.server import WebSocketServerProtocol as ServerConnection, serve

logger = logging.getLogger(__name__)

MessageCallback = Callable[["WebSocketServer", ServerConnection, str | bytes], Awaitable[None]]


class WebSocketServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self._routes: dict[str, tuple[set[ServerConnection], MessageCallback]] = {}
        self._server = None

    def add_route(self, path: str, callback: MessageCallback) -> None:
        self._routes[path] = (set(), callback)

    async def send_to(self, connection: ServerConnection, message: str) -> None:
        await connection.send(message)

    async def broadcast(
        self,
        message: str | bytes,
        path: str | None = None,
        exclude: ServerConnection | None = None,
    ) -> None:
        if path:
            clients, _ = self._routes.get(path, (set(), None))
            targets = set(clients)
        else:
            targets = set()
            for c, _ in self._routes.values():
                targets.update(c)
        if exclude is not None:
            targets.discard(exclude)
        if not targets:
            return
        results = await asyncio.gather(
            *[c.send(message) for c in targets],
            return_exceptions=True,
        )
        for connection, result in zip(targets, results):
            if isinstance(result, ConnectionClosed):
                self._discard_client(connection)
            elif isinstance(result, Exception):
                logger.warning("Failed to send websocket message: %s", result)

    def client_count(self, path: str) -> int:
        clients, _ = self._routes.get(path, (set(), None))
        return len(clients)

    def _discard_client(self, connection: ServerConnection) -> None:
        for clients, _ in self._routes.values():
            clients.discard(connection)

    async def _handler(self, websocket: ServerConnection, path: str | None = None) -> None:
        request = getattr(websocket, "request", None)
        raw_path = path or getattr(request, "path", None) or getattr(websocket, "path", "/")
        request_path = urlsplit(raw_path).path
        route = self._routes.get(request_path)
        if route is None:
            logger.warning("No route for path %s", raw_path)
            await websocket.close()
            return
        clients, callback = route
        clients.add(websocket)
        logger.info("Client connected on %s. Total: %s", request_path, len(clients))
        try:
            async for message in websocket:
                if callback is not None:
                    await callback(self, websocket, message)
        except ConnectionClosedOK:
            logger.debug("WebSocket closed on %s", request_path)
        except ConnectionClosedError as exc:
            logger.debug("WebSocket closed abruptly on %s: %s", request_path, exc)
        except Exception as exc:
            logger.error("WebSocket error on %s: %s", request_path, exc)
        finally:
            clients.discard(websocket)
            logger.debug("Client disconnected from %s. Total: %s", request_path, len(clients))

    async def start(self) -> None:
        logger.debug("Starting WebSocket server on %s:%s", self.host, self.port)
        async with serve(self._handler, self.host, self.port) as server:
            self._server = server
            logger.debug("WebSocket server running on ws://%s:%s", self.host, self.port)
            await server.serve_forever()
