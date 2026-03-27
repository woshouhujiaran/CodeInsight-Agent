from __future__ import annotations

import jsonschema
import pytest

from app.agent.plan_schema import (
    coerce_legacy_plan,
    topological_sort_steps,
    validate_plan_json_schema,
    validate_step_graph,
    validate_tool_args,
)


def test_plan_json_schema_accepts_minimal_valid_plan() -> None:
    plan = [
        {
            "id": "a",
            "deps": [],
            "tool": "search_tool",
            "args": {"query": "hello"},
            "success_criteria": "non-empty hits",
        }
    ]
    validate_plan_json_schema(plan)


def test_plan_json_schema_rejects_extra_property() -> None:
    plan = [
        {
            "id": "a",
            "deps": [],
            "tool": "search_tool",
            "args": {"query": "x"},
            "success_criteria": "ok",
            "extra": 1,
        }
    ]
    with pytest.raises(jsonschema.ValidationError):
        validate_plan_json_schema(plan)


def test_topological_order_respects_deps() -> None:
    steps = [
        {"id": "s2", "deps": ["s1"], "tool": "analyze_tool", "args": {"input": "x"}, "success_criteria": "c"},
        {"id": "s1", "deps": [], "tool": "search_tool", "args": {"query": "x"}, "success_criteria": "c"},
    ]
    ordered = topological_sort_steps(steps)
    assert [s["id"] for s in ordered] == ["s1", "s2"]


def test_cycle_raises() -> None:
    steps = [
        {"id": "a", "deps": ["b"], "tool": "search_tool", "args": {"query": "x"}, "success_criteria": "c"},
        {"id": "b", "deps": ["a"], "tool": "search_tool", "args": {"query": "y"}, "success_criteria": "c"},
    ]
    with pytest.raises(ValueError, match="cycle"):
        topological_sort_steps(steps)


def test_coerce_legacy_plan() -> None:
    raw = [
        {"tool": "search_tool", "input": "q1"},
        {"tool": "analyze_tool", "input": "q2"},
    ]
    out = coerce_legacy_plan(raw, "fallback")
    assert out is not None
    validate_plan_json_schema(out)
    assert out[0]["id"] == "step_1"
    assert out[1]["deps"] == ["step_1"]


def test_validate_tool_args_search() -> None:
    assert validate_tool_args("search_tool", {"query": "a"}) is None
    assert validate_tool_args("search_tool", {}) is not None


def test_validate_step_graph_unknown_dep() -> None:
    steps = [
        {"id": "a", "deps": ["missing"], "tool": "search_tool", "args": {"query": "x"}, "success_criteria": "c"},
    ]
    assert validate_step_graph(steps) is not None
