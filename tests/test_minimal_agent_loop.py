from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from app.agent.agent import CodeAgent
from app.agent.executor import Executor
from app.agent.memory import ConversationMemory
from app.agent.planner import Planner
from app.agent.tool_registry import ToolRegistry
from app.llm.llm import LLMClient
from app.tools.base_tool import BaseTool, make_tool_result


class EchoTool(BaseTool):
    name = "echo_tool"
    description = "Echo input for minimal loop tests."

    def run(self, input: dict[str, Any] | str) -> dict[str, Any]:
        return make_tool_result(status="ok", data=input, error="", meta={})


def test_run_minimal_agent_loop_plans_then_executes_tools() -> None:
    llm = LLMClient(provider="deepseek", model="test-model")
    llm.generate_agentic_json_turn = MagicMock(
        side_effect=[
            {"type": "tool_calls", "calls": [{"name": "echo_tool", "arguments": {"task": 1}}]},
            {"type": "final", "content": "task-1 done"},
            {"type": "tool_calls", "calls": [{"name": "echo_tool", "arguments": {"task": 2}}]},
            {"type": "final", "content": "task-2 done"},
        ]
    )
    llm.generate_answer = MagicMock(return_value="all tasks completed")

    planner = Planner(llm=llm)
    planner.make_task_board = MagicMock(
        return_value=[
            {
                "id": "t1",
                "title": "Locate files",
                "description": "Find relevant files.",
                "acceptance": "Key paths are identified.",
                "depends_on": [],
                "status": "pending",
            },
            {
                "id": "t2",
                "title": "Summarize findings",
                "description": "Summarize based on tool outputs.",
                "acceptance": "Summary is grounded in tool output.",
                "depends_on": ["t1"],
                "status": "pending",
            },
        ]
    )

    registry = ToolRegistry()
    registry.register(EchoTool())
    executor = Executor(registry=registry)
    memory = ConversationMemory()
    agent = CodeAgent(planner=planner, executor=executor, llm=llm, memory=memory, workspace_root=".")

    result = agent.run_minimal_agent_loop(
        "Please inspect and summarize.",
        max_tasks=2,
        max_turns_per_task=4,
    )

    assert result.answer == "all tasks completed"
    assert len(result.tool_trace) == 2
    assert all(item.get("tool") == "echo_tool" for item in result.tool_trace)
    planner.make_task_board.assert_called_once()
    llm.generate_answer.assert_called_once()

    turn_meta = memory.get_turn_metadata()
    assert turn_meta
    assert turn_meta[-1].get("minimal_loop") is True
    assert turn_meta[-1].get("mode") == "minimal_loop"
    assert turn_meta[-1]["repro_manifest"]["mode"] == "minimal_loop"
    assert len(turn_meta[-1].get("plan") or []) == 2
