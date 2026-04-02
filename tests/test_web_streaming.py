from __future__ import annotations

from pathlib import Path
import time
from threading import Event

from app.web.streaming import StreamWorker


def test_stream_worker_should_stop_cancels_background_turn() -> None:
    started = Event()
    cancelled = Event()

    def worker(cancel_event: Event, emit) -> None:
        emit({"event": "mode", "data": {"mode": "agentic"}})
        started.set()
        while not cancel_event.is_set():
            time.sleep(0.02)
        cancelled.set()

    stop_requested = {"value": False}
    stream = StreamWorker(worker)
    iterator = stream.iter_events(should_stop=lambda: stop_requested["value"], poll_interval=0.02)

    assert next(iterator)["event"] == "mode"
    assert started.wait(timeout=1.0) is True

    stop_requested["value"] = True
    assert list(iterator) == []

    deadline = time.time() + 1.0
    while time.time() < deadline and not cancelled.is_set():
        time.sleep(0.02)

    assert cancelled.is_set() is True


def test_stream_endpoint_checks_request_disconnect() -> None:
    source = Path("app/web/main.py").read_text(encoding="utf-8")

    assert "request.is_disconnected()" in source
    assert "create_stream_chat_worker" in source
