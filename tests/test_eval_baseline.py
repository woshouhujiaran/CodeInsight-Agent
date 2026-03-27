from __future__ import annotations

from pathlib import Path

from scripts.run_eval import load_tasks, summarize, task_success


def test_load_tasks_has_minimum_size() -> None:
    tasks = load_tasks(Path("eval/tasks.json"))
    assert len(tasks) >= 20


def test_task_success_keyword_match() -> None:
    assert task_success("这里给出 pytest 测试建议", ["pytest", "覆盖"]) is True
    assert task_success("这里只讨论架构", ["测试"]) is False


def test_summarize_metrics() -> None:
    summary = summarize(
        [
            {"success": True, "duration_ms": 100, "recovery_applied": False},
            {"success": False, "duration_ms": 300, "recovery_applied": True},
        ]
    )
    assert summary["total"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["avg_duration_ms"] == 200.0
    assert summary["recovery_trigger_rate"] == 0.5
