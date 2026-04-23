from __future__ import annotations

from app.metrics.quality import summarize_agentic_turn, summarize_retrieval_cases, summarize_tool_trace


def test_summarize_retrieval_cases_includes_recall_and_mrr() -> None:
    tasks = [
        {"kind": "retrieval_expectation", "details": {"matched_rank": 1, "reciprocal_rank": 1.0}},
        {"kind": "retrieval_expectation", "details": {"matched_rank": 4, "reciprocal_rank": 0.25}},
        {"kind": "retrieval_expectation", "details": {"matched_rank": 0, "reciprocal_rank": 0.0}},
    ]
    metrics = summarize_retrieval_cases(tasks)
    assert metrics["retrieval_case_count"] == 3
    assert metrics["retrieval_hit_rate"] == 0.6667
    assert metrics["retrieval_mrr"] == 0.4167
    assert metrics["retrieval_recall_at_1"] == 0.3333
    assert metrics["retrieval_recall_at_3"] == 0.3333
    assert metrics["retrieval_recall_at_5"] == 0.6667


def test_summarize_agentic_turn_aggregates_tool_and_round_metrics() -> None:
    task_results = [
        {"task_id": "t1", "status": "done"},
        {"task_id": "t2", "status": "failed"},
        {"task_id": "t1", "status": "done"},
    ]
    tool_trace = [
        {"status": "ok", "attempts": 1, "timed_out": False},
        {"status": "error", "attempts": 2, "timed_out": True, "error_type": "transient"},
    ]
    metrics = summarize_agentic_turn(task_results=task_results, tool_trace=tool_trace)
    assert metrics["task_count"] == 3
    assert metrics["task_completed_count"] == 2
    assert metrics["tool_step_count"] == 2
    assert metrics["tool_success_rate"] == 0.5
    assert metrics["tool_timeout_rate"] == 0.5
    assert metrics["agent_rounds"] == 2


def test_summarize_tool_trace_empty() -> None:
    metrics = summarize_tool_trace([])
    assert metrics["tool_step_count"] == 0
    assert metrics["tool_success_rate"] == 0.0
