from __future__ import annotations

import time
from typing import Any

from app.agent.executor import Executor
from app.agent.tool_registry import ToolRegistry
from app.tools.base_tool import BaseTool, make_tool_result


class FlakyTransientTool(BaseTool):
    name = "search_tool"
    description = "fails once then succeeds"

    def __init__(self) -> None:
        self._calls = 0

    def run(self, input: dict[str, Any] | str) -> dict[str, Any]:
        self._calls += 1
        if self._calls == 1:
            raise TimeoutError("network timeout")
        return make_tool_result(status="ok", data={"ok": True}, meta={"calls": self._calls})


class PermanentFailTool(BaseTool):
    name = "analyze_tool"
    description = "always permanent failure"

    def run(self, input: dict[str, Any] | str) -> dict[str, Any]:
        raise ValueError("invalid input format")


class SlowTool(BaseTool):
    name = "optimize_tool"
    description = "slow tool for timeout test"

    def run(self, input: dict[str, Any] | str) -> dict[str, Any]:
        time.sleep(0.2)
        return make_tool_result(status="ok", data={"slow": True})


def test_executor_retries_transient_then_succeeds() -> None:
    registry = ToolRegistry()
    registry.register(FlakyTransientTool())
    executor = Executor(registry=registry)
    results = executor.execute_tools(
        [
            {
                "id": "s1",
                "deps": [],
                "tool": "search_tool",
                "args": {"query": "q"},
                "success_criteria": "ok",
                "max_retries": 2,
            }
        ]
    )
    assert results[0]["status"] == "ok"
    assert results[0]["attempts"] == 2
    assert results[0]["error_type"] == ""
    assert results[0]["timed_out"] is True
    assert results[0]["duration_ms"] >= 300


def test_executor_stops_retry_on_permanent_error() -> None:
    registry = ToolRegistry()
    registry.register(PermanentFailTool())
    executor = Executor(registry=registry)
    results = executor.execute_tools(
        [
            {
                "id": "s1",
                "deps": [],
                "tool": "analyze_tool",
                "args": {"input": "x"},
                "success_criteria": "ok",
                "max_retries": 2,
            }
        ]
    )
    assert results[0]["status"] == "error"
    assert results[0]["attempts"] == 1
    assert results[0]["error_type"] == "permanent"
    assert results[0]["timed_out"] is False


def test_executor_timeout_failure_after_retries() -> None:
    registry = ToolRegistry()
    registry.register(SlowTool())
    executor = Executor(registry=registry, step_timeout_seconds=0.05)
    results = executor.execute_tools(
        [
            {
                "id": "s1",
                "deps": [],
                "tool": "optimize_tool",
                "args": {"input": "x"},
                "success_criteria": "ok",
                "max_retries": 1,
            }
        ]
    )
    assert results[0]["status"] == "error"
    assert results[0]["attempts"] == 2
    assert results[0]["error_type"] == "transient"
    assert results[0]["timed_out"] is True
    assert results[0]["duration_ms"] >= 400
