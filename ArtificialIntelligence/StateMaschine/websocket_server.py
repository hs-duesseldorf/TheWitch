import asyncio
import logging
from typing import Awaitable, Callable, Optional, Set

from websockets.asyncio.server import ServerConnection, serve

logger = logging.getLogger(__name__)

MessageCallback = Callable[[str, "WebSocketServer", ServerConnection], Awaitable[None]]


class WebSocketServer:
    """WebSocket server that only manages clients and message transport."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self._clients: Set[ServerConnection] = set()
        self._on_message: Optional[MessageCallback] = None
        self._server = None

    def register_message_callback(self, callback: MessageCallback) -> None:
        """Register a callback for incoming messages."""
        self._on_message = callback

    async def send_to_client(self, websocket: ServerConnection, message: str) -> None:
        """Send a message to a single client."""
        await websocket.send(message)

    async def broadcast(self, message: str) -> None:
        """Send a message to all connected clients."""
        if not self._clients:
            return
        await asyncio.gather(
            *[client.send(message) for client in self._clients],
            return_exceptions=True,
        )

    async def _register_client(self, websocket: ServerConnection) -> None:
        self._clients.add(websocket)
        logger.info("Client connected. Total clients: %s", len(self._clients))

    async def _unregister_client(self, websocket: ServerConnection) -> None:
        self._clients.discard(websocket)
        logger.info("Client disconnected. Total clients: %s", len(self._clients))

    async def _handle_message(self, websocket: ServerConnection, message: str) -> None:
        if self._on_message is None:
            logger.debug("No message callback registered. Dropping message.")
            return
        await self._on_message(message, self, websocket)

    async def _handler(self, websocket: ServerConnection) -> None:
        await self._register_client(websocket)
        try:
            async for message in websocket:
                await self._handle_message(websocket, message)
        except Exception as exc:
            logger.error("WebSocket handler error: %s", exc)
        finally:
            await self._unregister_client(websocket)

    async def start(self) -> None:
        """Start the WebSocket server and serve forever."""
        logger.info("Starting WebSocket server on %s:%s", self.host, self.port)
        async with serve(self._handler, self.host, self.port) as server:
            self._server = server
            logger.info("WebSocket server running on ws://%s:%s", self.host, self.port)
            await server.serve_forever()
