from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from app.agent.agent import AgenticTurnResult, CodeAgent
from app.agent.executor import Executor
from app.agent.memory import ConversationMemory
from app.agent.planner import Planner
from app.agent.tool_registry import ToolRegistry
from app.llm.llm import LLMClient, parse_agentic_turn, strip_json_fences
from app.tools.base_tool import BaseTool, make_tool_result


class EchoTool(BaseTool):
    name = "echo_tool"
    description = "回显参数，用于测试。"

    def run(self, input: dict[str, Any] | str) -> dict[str, Any]:
        return make_tool_result(status="ok", data=input, error="", meta={})


def test_run_agentic_two_tool_rounds_then_final() -> None:
    llm = LLMClient(provider="deepseek", model="test-model")
    llm.generate_agentic_json_turn = MagicMock(
        side_effect=[
            {"type": "tool_calls", "calls": [{"name": "echo_tool", "arguments": {"round": 1}}]},
            {"type": "tool_calls", "calls": [{"name": "echo_tool", "arguments": {"round": 2}}]},
            {"type": "final", "content": "两轮工具已完成。"},
        ]
    )

    registry = ToolRegistry()
    registry.register(EchoTool())
    executor = Executor(registry=registry)
    planner = Planner(llm=llm)
    memory = ConversationMemory()
    agent = CodeAgent(planner=planner, executor=executor, llm=llm, memory=memory, workspace_root="/tmp/demo")

    result = agent.run_agentic("请分两轮调用 echo", max_turns=8)

    assert isinstance(result, AgenticTurnResult)
    assert result.answer == "两轮工具已完成。"
    assert len(result.tool_trace) == 2
    assert all(r.get("tool") == "echo_tool" for r in result.tool_trace)
    assert llm.generate_agentic_json_turn.call_count == 3

    msgs = memory.get_messages()
    assert msgs[-2]["role"] == "user"
    assert msgs[-1]["role"] == "assistant"
    assert "两轮工具已完成" in msgs[-1]["content"]


def test_parse_agentic_turn_accepts_fenced_json() -> None:
    raw = '```json\n{"type":"final","content":"OK"}\n```'
    assert strip_json_fences(raw) == '{"type":"final","content":"OK"}'
    parsed = parse_agentic_turn(raw)
    assert parsed == {"type": "final", "content": "OK"}


def test_parse_agentic_turn_tool_calls() -> None:
    raw = '{"type":"tool_calls","calls":[{"name":"echo_tool","arguments":{"a":1}}]}'
    assert parse_agentic_turn(raw) == {
        "type": "tool_calls",
        "calls": [{"name": "echo_tool", "arguments": {"a": 1}}],
    }
