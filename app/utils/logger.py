from __future__ import annotations

import contextvars
import json
import logging
from typing import Any

_TRACE_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


def set_trace_id(trace_id: str) -> None:
    _TRACE_ID_CTX.set((trace_id or "-").strip() or "-")


def get_trace_id() -> str:
    return _TRACE_ID_CTX.get()


class _TraceIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = get_trace_id()
        return True


def log_event(
    logger: logging.Logger,
    *,
    module: str,
    action: str,
    status: str,
    duration_ms: int = 0,
    **fields: Any,
) -> None:
    payload: dict[str, Any] = {
        "trace_id": get_trace_id(),
        "module": module,
        "action": action,
        "status": status,
        "duration_ms": int(duration_ms),
    }
    payload.update(fields)
    logger.info(json.dumps(payload, ensure_ascii=False))


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | trace_id=%(trace_id)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    handler.addFilter(_TraceIdFilter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger
