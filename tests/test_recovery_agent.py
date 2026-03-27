from __future__ import annotations

from app.agent.agent import CodeAgent
from app.agent.executor import Executor
from app.agent.planner import Planner
from app.agent.tool_registry import ToolRegistry
from app.llm.llm import LLMClient
from app.tools.base_tool import BaseTool


class EmptySearchTool(BaseTool):
    name = "search_tool"
    description = "always empty hits"

    def run(self, input: str) -> str:
        return "[]"


class DummyAnalyzeTool(BaseTool):
    name = "analyze_tool"
    description = "dummy"

    def run(self, input: str) -> str:
        return f"analysis:{input[:80]}"


def test_agent_triggers_recovery_replan_once() -> None:
    llm = LLMClient(provider="none", model="dummy")
    planner = Planner(llm=llm)
    registry = ToolRegistry()
    registry.register(EmptySearchTool())
    registry.register(DummyAnalyzeTool())
    executor = Executor(registry=registry)
    agent = CodeAgent(planner=planner, executor=executor, llm=llm)

    result = agent.run("分析登录模块")

    assert result.recovery_applied is True
    assert len(result.plan) >= 4
    assert len(result.tool_results) >= 4
    rounds = {r.get("replan_round") for r in result.tool_results}
    assert rounds == {1, 2}
    recovery_items = [r for r in result.tool_results if r.get("replan_round") == 2]
    assert recovery_items
    assert all(r.get("recovery_trigger_reason") == "empty_search_hits" for r in recovery_items)
    assert all(r.get("recovery_strategy") == "split_search" for r in recovery_items)
    assert "Round 1" in result.context
    assert "Round 2" in result.context
