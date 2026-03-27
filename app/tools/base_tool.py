from __future__ import annotations

from abc import ABC, abstractmethod
import json
from typing import Any, Literal, TypedDict


class BaseTool(ABC):
    """
    Base class for all tools.

    Every concrete tool should define:
    - name: stable unique identifier for registry lookup
    - description: short explanation of tool capability
    - run(input): execute tool logic and return output text
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, input: dict[str, Any] | str) -> dict[str, Any] | str:
        """Execute tool logic with structured args or legacy string input."""
        raise NotImplementedError


class ToolResult(TypedDict):
    status: Literal["ok", "error"]
    data: Any
    error: str
    meta: dict[str, Any]


def make_tool_result(
    *,
    status: Literal["ok", "error"],
    data: Any = None,
    error: str = "",
    meta: dict[str, Any] | None = None,
) -> ToolResult:
    return {
        "status": status,
        "data": data,
        "error": error,
        "meta": dict(meta or {}),
    }


def ensure_tool_result(result: dict[str, Any] | str) -> ToolResult:
    """
    Normalize mixed tool outputs to ToolResult.
    Backward compatibility:
    - plain string output => {"status":"ok","data": <string>, ...}
    - dict without status/data/error/meta => wrapped as data payload
    """
    if isinstance(result, str):
        return make_tool_result(status="ok", data=result)

    if not isinstance(result, dict):
        return make_tool_result(status="ok", data=str(result))

    has_protocol_keys = {"status", "data", "error", "meta"}.issubset(result.keys())
    if has_protocol_keys:
        status = result.get("status")
        if status not in {"ok", "error"}:
            status = "error"
        data = result.get("data")
        error = result.get("error")
        if not isinstance(error, str):
            error = str(error or "")
        meta = result.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        return make_tool_result(status=status, data=data, error=error, meta=meta)

    return make_tool_result(status="ok", data=result)


def tool_result_to_legacy_output(result: ToolResult) -> str:
    """Render ToolResult data into the old text-friendly output field."""
    data = result.get("data")
    if isinstance(data, str):
        return data
    if data is None:
        return ""
    try:
        return json.dumps(data, ensure_ascii=False)
    except TypeError:
        return str(data)
