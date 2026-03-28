from __future__ import annotations

from typing import Any
import json

from app.agent.executor import Executor
from app.agent.tool_registry import ToolRegistry
from app.agent.tool_specs import (
    compact_tool_specs_for_prompt,
    validate_agentic_tool_call,
)
from app.tools.base_tool import BaseTool, make_tool_result


class DummyAnalyzeTool(BaseTool):
    name = "analyze_tool"
    description = "测试用分析工具"

    def run(self, input: dict[str, Any] | str) -> dict[str, Any]:
        return make_tool_result(status="ok", data="ok", error="", meta={})


def test_execute_agentic_rejects_invalid_analyze_arguments() -> None:
    registry = ToolRegistry()
    registry.register(DummyAnalyzeTool())
    executor = Executor(registry=registry)

    results = executor.execute_agentic_calls([{"name": "analyze_tool", "arguments": {}}])

    assert len(results) == 1
    assert results[0]["status"] == "error"
    assert results[0]["tool_result"]["meta"].get("invalid_arguments") is True
    out = str(results[0]["output"])
    assert "参数" in out or "input" in out.lower() or "required" in out.lower()


def test_execute_agentic_rejects_search_without_query_or_input() -> None:
    registry = ToolRegistry()
    registry.register(DummyAnalyzeTool(), name="search_tool")

    executor = Executor(registry=registry)
    results = executor.execute_agentic_calls([{"name": "search_tool", "arguments": {}}])

    assert results[0]["status"] == "error"
    assert results[0]["tool_result"]["meta"].get("invalid_arguments") is True
    assert "参数" in str(results[0]["output"]) or "query" in str(results[0]["output"])


def test_validate_agentic_tool_call_unknown_tool() -> None:
    registry = ToolRegistry()
    registry.register(DummyAnalyzeTool())
    err = validate_agentic_tool_call(registry, "missing_tool", {})
    assert err is not None
    assert "未注册" in err


def test_compact_tool_specs_for_prompt_is_valid_json() -> None:
    registry = ToolRegistry()
    registry.register(DummyAnalyzeTool())
    text = compact_tool_specs_for_prompt(registry.list_specs())
    parsed = json.loads(text)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "analyze_tool"
    assert parsed[0]["parameters"]["type"] == "object"
