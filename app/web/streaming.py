from __future__ import annotations

from queue import Queue
from threading import Event, Thread
from typing import Callable, Iterator

from app.contracts import ServiceEvent
from app.sandbox.runner import bind_cancellation_event
from app.web.chat_components import StreamCancelled


class StreamWorker:
    def __init__(self, worker: Callable[[Event, Callable[[ServiceEvent], None]], None]) -> None:
        self._worker = worker

    def iter_events(self) -> Iterator[ServiceEvent]:
        queue: Queue[ServiceEvent | None] = Queue()
        cancel_event = Event()

        def run() -> None:
            with bind_cancellation_event(cancel_event):
                try:
                    self._worker(cancel_event, queue.put)
                except StreamCancelled:
                    queue.put({"event": "error", "data": {"message": "\u5df2\u53d6\u6d88"}})
                except Exception as exc:  # noqa: BLE001
                    queue.put({"event": "error", "data": {"message": str(exc)}})
                finally:
                    queue.put(None)

        thread = Thread(target=run, daemon=True)
        thread.start()
        try:
            while True:
                item = queue.get()
                if item is None:
                    break
                yield item
        finally:
            cancel_event.set()
            thread.join(timeout=2.0)
