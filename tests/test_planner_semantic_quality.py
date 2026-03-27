from __future__ import annotations

import json

from app.agent.planner import Planner
from app.llm.llm import LLMClient


class FixedPlannerLLM(LLMClient):
    def __init__(self, payload: list[dict]) -> None:
        super().__init__(provider="none", model="fixed-planner")
        self._payload = payload

    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        return json.dumps(self._payload, ensure_ascii=False)


def test_planner_fallback_when_test_intent_without_test_tool() -> None:
    llm = FixedPlannerLLM(
        [
            {
                "id": "s1",
                "deps": [],
                "tool": "search_tool",
                "args": {"query": "login"},
                "success_criteria": "x",
            },
            {
                "id": "s2",
                "deps": ["s1"],
                "tool": "analyze_tool",
                "args": {"input": "login"},
                "success_criteria": "x",
            },
        ]
    )
    planner = Planner(llm=llm)
    plan = planner.make_plan("请帮我写测试并生成 pytest 用例", history=[])
    assert any(step["tool"] == "test_tool" for step in plan)
    assert planner.last_plan_score is not None
    assert planner.last_plan_score["fallback_reason"] in ("", "low_plan_score")


def test_planner_score_keeps_good_test_plan() -> None:
    llm = FixedPlannerLLM(
        [
            {
                "id": "s1",
                "deps": [],
                "tool": "search_tool",
                "args": {"query": "login"},
                "success_criteria": "x",
            },
            {
                "id": "s2",
                "deps": ["s1"],
                "tool": "test_tool",
                "args": {"input": "login"},
                "success_criteria": "x",
            },
        ]
    )
    planner = Planner(llm=llm)
    plan = planner.make_plan("写测试", history=[])
    assert any(step["tool"] == "test_tool" for step in plan)
    assert planner.last_plan_score is not None
    assert planner.last_plan_score["overall"] >= 0.65
