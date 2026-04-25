from __future__ import annotations

import asyncio
import copy
import json
import threading
from collections import deque
from typing import Any, Callable

from .utils import iso_timestamp


class LatestMessageBus:
    def __init__(self, initial_message: dict[str, Any] | None = None):
        self._condition = threading.Condition()
        self._sequence = 0
        self._message = copy.deepcopy(initial_message)

    def publish(self, message: dict[str, Any]) -> None:
        with self._condition:
            self._sequence += 1
            self._message = copy.deepcopy(message)
            self._condition.notify_all()

    def latest(self) -> dict[str, Any] | None:
        with self._condition:
            return copy.deepcopy(self._message)

    def wait_for_message(self, last_sequence: int, timeout: float = 1.0) -> tuple[int, dict[str, Any] | None]:
        with self._condition:
            if self._sequence == last_sequence:
                self._condition.wait(timeout=timeout)
            return self._sequence, copy.deepcopy(self._message)


class MessageQueue:
    def __init__(self, maxlen: int = 256):
        self._condition = threading.Condition()
        self._items: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._maxlen = maxlen
        self._dropped = 0

    def publish(self, message: dict[str, Any]) -> None:
        with self._condition:
            if len(self._items) >= self._maxlen:
                self._dropped += 1
            self._items.append(copy.deepcopy(message))
            self._condition.notify_all()

    def get(self, timeout: float = 1.0) -> dict[str, Any] | None:
        with self._condition:
            if not self._items:
                self._condition.wait(timeout=timeout)
            if not self._items:
                return None
            return self._items.popleft()

    def snapshot(self) -> dict[str, int]:
        with self._condition:
            return {
                "queued": len(self._items),
                "dropped": self._dropped,
            }


class PipelineWebSocketClient:
    def __init__(
        self,
        url: str,
        queue: MessageQueue,
        *,
        command_handler: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    ):
        self.url = url
        self.queue = queue
        self.command_handler = command_handler
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._status: dict[str, Any] = {
            "enabled": bool(self.url),
            "url": self.url or None,
            "state": "disabled" if not self.url else "idle",
            "attempts": 0,
            "sent": 0,
            "last_error": None,
            "last_sent_at": None,
        }

    def start(self) -> None:
        if not self.url or self.worker is not None:
            return
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._run_thread, name="pipeline-websocket-client", daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.worker is not None and self.worker.is_alive() and self.worker is not threading.current_thread():
            self.worker.join(timeout=3.0)
        self.worker = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            status = copy.deepcopy(self._status)
        status.update(self.queue.snapshot())
        return status

    def _update_status(self, **changes: Any) -> None:
        with self._lock:
            self._status.update(changes)

    def _increment_status(self, key: str) -> int:
        with self._lock:
            self._status[key] = int(self._status[key]) + 1
            return int(self._status[key])

    def _run_thread(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        try:
            from websockets.asyncio.client import connect
        except Exception:
            from websockets import connect  # type: ignore[no-redef]

        while not self.stop_event.is_set():
            self._update_status(
                state="connecting",
                attempts=self._increment_status("attempts"),
            )
            try:
                async with connect(self.url) as websocket:
                    self._update_status(state="connected", last_error=None)
                    sender = asyncio.create_task(self._send_loop(websocket))
                    receiver = asyncio.create_task(self._receive_loop(websocket))
                    done, pending = await asyncio.wait(
                        {sender, receiver},
                        return_when=asyncio.FIRST_EXCEPTION,
                    )
                    for task in pending:
                        task.cancel()
                    for task in done:
                        task.result()
            except Exception as exc:
                self._update_status(state="error", last_error=str(exc))
                await self._sleep_or_stop(2.0)

        if self.url:
            self._update_status(state="stopped")

    async def _sleep_or_stop(self, seconds: float) -> None:
        steps = max(1, int(seconds * 10))
        for _ in range(steps):
            if self.stop_event.is_set():
                return
            await asyncio.sleep(seconds / steps)

    async def _send_loop(self, websocket) -> None:
        while not self.stop_event.is_set():
            message = await asyncio.to_thread(self.queue.get, 0.2)
            if message is None:
                continue
            await websocket.send(json.dumps(message))
            self._update_status(
                sent=self._increment_status("sent"),
                last_sent_at=iso_timestamp(),
            )

    async def _receive_loop(self, websocket) -> None:
        async for raw_message in websocket:
            if self.command_handler is None:
                continue
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                response = {"type": "command_result", "ok": False, "error": "Invalid JSON command"}
            else:
                if not isinstance(message, dict):
                    response = {"type": "command_result", "ok": False, "error": "Command must be a JSON object"}
                else:
                    response = await asyncio.to_thread(self.command_handler, message)
            if response is not None:
                self.queue.publish(response)
