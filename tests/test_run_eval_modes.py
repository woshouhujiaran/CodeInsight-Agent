from __future__ import annotations

from unittest.mock import MagicMock

from scripts.run_eval import run_one_eval_task, task_success, task_use_agentic


def test_task_use_agentic_per_task_overrides_cli() -> None:
    assert task_use_agentic({}, cli_agentic=False) is False
    assert task_use_agentic({}, cli_agentic=True) is True
    assert task_use_agentic({"agentic": False}, cli_agentic=True) is False
    assert task_use_agentic({"agentic": True}, cli_agentic=False) is True


def test_run_one_eval_task_planner_path_mock() -> None:
    agent = MagicMock()
    agent.run.return_value = MagicMock(answer="登录流程与风险建议", recovery_applied=True)

    row = run_one_eval_task(
        agent,
        {"id": "T", "category": "analysis", "query": "q", "expected_keywords": ["登录", "风险"]},
        cli_agentic=False,
        max_turns=8,
    )
    assert row["success"] is True
    assert row["recovery_applied"] is True
    assert row["agentic"] is False
    agent.run.assert_called_once_with("q")
    agent.run_agentic.assert_not_called()


def test_run_one_eval_task_agentic_path_mock() -> None:
    agent = MagicMock()
    agent.run_agentic.return_value = MagicMock(answer="pytest 与 单元测试 清单")

    row = run_one_eval_task(
        agent,
        {"id": "T2", "category": "test", "query": "q2", "expected_keywords": ["pytest"]},
        cli_agentic=True,
        max_turns=3,
    )
    assert row["success"] is True
    assert row["agentic"] is True
    assert row["recovery_applied"] is False
    agent.run_agentic.assert_called_once_with("q2", max_turns=3)
    agent.run.assert_not_called()


def test_task_success_unchanged_for_eval() -> None:
    assert task_success("包含 pytest 关键字", ["pytest"]) is True
