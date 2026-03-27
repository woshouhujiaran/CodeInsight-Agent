from __future__ import annotations

from app.agent.recovery import search_tool_output_is_empty, should_recovery_replan


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
