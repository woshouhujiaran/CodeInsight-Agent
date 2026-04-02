from __future__ import annotations

from queue import Empty, Queue
from threading import Event, Thread
from typing import Callable, Iterator

from app.contracts import ServiceEvent
from app.sandbox.runner import bind_cancellation_event
from app.web.chat_components import StreamCancelled


class StreamWorker:
    def __init__(self, worker: Callable[[Event, Callable[[ServiceEvent], None]], None]) -> None:
        self._worker = worker
        self._queue: Queue[ServiceEvent | None] = Queue()
        self._cancel_event = Event()
        self._thread: Thread | None = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True

        def run() -> None:
            with bind_cancellation_event(self._cancel_event):
                try:
                    self._worker(self._cancel_event, self._queue.put)
                except StreamCancelled:
                    self._queue.put({"event": "error", "data": {"message": "\u5df2\u53d6\u6d88"}})
                except Exception as exc:  # noqa: BLE001
                    self._queue.put({"event": "error", "data": {"message": str(exc)}})
                finally:
                    self._queue.put(None)

        self._thread = Thread(target=run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._cancel_event.set()

    def close(self, timeout: float = 2.0) -> None:
        self.cancel()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def iter_events(
        self,
        *,
        should_stop: Callable[[], bool] | None = None,
        poll_interval: float = 0.1,
    ) -> Iterator[ServiceEvent]:
        self.start()
        try:
            while True:
                if should_stop is not None and should_stop():
                    self.cancel()
                    break
                try:
                    item = self._queue.get(timeout=poll_interval)
                except Empty:
                    continue
                if item is None:
                    break
                yield item
        finally:
            self.close()
