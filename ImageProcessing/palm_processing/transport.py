from __future__ import annotations

import asyncio
import copy
import threading
from collections import deque
from typing import Any

import msgspec


class WebSocketClient:
    def __init__(self, url: str, *, max_queue: int = 256):
        self.url = url
        self.max_queue = max_queue
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self._condition = threading.Condition()
        self._queue: deque[Any] = deque(maxlen=max_queue)
        self._lock = threading.Lock()
        self._status: dict[str, Any] = {
            "enabled": bool(url),
            "url": url or None,
            "state": "disabled" if not url else "idle",
            "sent": 0,
            "dropped": 0,
            "last_error": None,
        }

    def start(self) -> None:
        if not self.url or self.worker is not None:
            return
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._run_thread, name=f"ws-client:{self.url}", daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self.worker is not None and self.worker.is_alive() and self.worker is not threading.current_thread():
            self.worker.join(timeout=3.0)
        self.worker = None

    def send_message(self, message: Any) -> None:
        if not self.url:
            return
        with self._condition:
            if len(self._queue) >= self.max_queue:
                self._increment_status("dropped")
            self._queue.append(message if isinstance(message, bytes) else copy.deepcopy(message))
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            status = copy.deepcopy(self._status)
        with self._condition:
            status["queued"] = len(self._queue)
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
            try:
                self._update_status(state="connecting")
                async with connect(self.url) as websocket:
                    self._update_status(state="connected", last_error=None)
                    while not self.stop_event.is_set():
                        message = await asyncio.to_thread(self._next_message, 0.2)
                        if message is None:
                            continue
                        await websocket.send(message if isinstance(message, bytes) else msgspec.json.encode(message).decode("utf-8"))
                        self._increment_status("sent")
            except Exception as exc:
                self._update_status(state="error", last_error=str(exc))
                await self._sleep_or_stop(2.0)

        self._update_status(state="stopped")

    def _next_message(self, timeout: float) -> Any | None:
        with self._condition:
            if not self._queue:
                self._condition.wait(timeout=timeout)
            if not self._queue:
                return None
            return self._queue.popleft()

    async def _sleep_or_stop(self, seconds: float) -> None:
        steps = max(1, int(seconds * 10))
        for _ in range(steps):
            if self.stop_event.is_set():
                return
            await asyncio.sleep(seconds / steps)
