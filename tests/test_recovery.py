from __future__ import annotations

from app.agent.recovery import (
    apply_recovery_strategy,
    evaluate_recovery,
    search_tool_output_is_empty,
    should_recovery_replan,
)


def test_search_tool_output_is_empty() -> None:
    assert search_tool_output_is_empty("[]") is True
    assert search_tool_output_is_empty("  []  ") is True
    assert search_tool_output_is_empty("") is True
    assert search_tool_output_is_empty('[{"x":1}]') is False


def test_should_recovery_replan_empty_hits() -> None:
    assert (
        should_recovery_replan(
            [{"tool": "search_tool", "status": "ok", "output": "[]"}],
        )
        is True
    )


def test_should_recovery_replan_search_error() -> None:
    assert (
        should_recovery_replan(
            [{"tool": "search_tool", "status": "error", "output": "boom"}],
        )
        is True
    )


def test_should_recovery_replan_no_trigger_for_analyze_only() -> None:
    assert (
        should_recovery_replan(
            [{"tool": "analyze_tool", "status": "ok", "output": "[]"}],
        )
        is False
    )


def test_should_recovery_replan_ok_hits() -> None:
    assert (
        should_recovery_replan(
            [{"tool": "search_tool", "status": "ok", "output": '[{"file_path":"a.py","content":"x"}]'}],
        )
        is False
    )


def test_evaluate_recovery_analyze_low_information() -> None:
    decision = evaluate_recovery(
        [{"tool": "analyze_tool", "status": "ok", "output": "信息不足，无法分析"}]
    )
    assert decision["triggered"] is True
    assert decision["reason"] == "analyze_low_information"
    assert decision["strategy"] == "analyze_with_assumption"


def test_apply_recovery_strategy_split_search() -> None:
    plan = [{"tool": "search_tool", "args": {"query": "登录模块"}}]
    out = apply_recovery_strategy(plan, strategy="split_search", user_query="登录 模块 异常")
    assert "query" in out[0]["args"]
    assert "登录" in str(out[0]["args"]["query"])
