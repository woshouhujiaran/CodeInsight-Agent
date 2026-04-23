from __future__ import annotations

from collections import Counter
from typing import Any


def _round(value: float) -> float:
    return round(float(value), 4)


def summarize_retrieval_cases(tasks: list[dict[str, Any]]) -> dict[str, float | int]:
    retrieval_tasks = [task for task in tasks if str(task.get("kind") or "") == "retrieval_expectation"]
    case_count = len(retrieval_tasks)
    if case_count == 0:
        return {
            "retrieval_case_count": 0,
            "retrieval_hit_rate": 0.0,
            "retrieval_mrr": 0.0,
            "retrieval_recall_at_1": 0.0,
            "retrieval_recall_at_3": 0.0,
            "retrieval_recall_at_5": 0.0,
        }

    hit_count = 0
    reciprocal_rank_sum = 0.0
    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0
    for task in retrieval_tasks:
        details = task.get("details") if isinstance(task.get("details"), dict) else {}
        rank = int(details.get("matched_rank", 0) or 0)
        rr = float(details.get("reciprocal_rank", 0.0) or 0.0)
        if rank > 0:
            hit_count += 1
        reciprocal_rank_sum += rr
        if 0 < rank <= 1:
            recall_at_1 += 1
        if 0 < rank <= 3:
            recall_at_3 += 1
        if 0 < rank <= 5:
            recall_at_5 += 1
    return {
        "retrieval_case_count": case_count,
        "retrieval_hit_rate": _round(hit_count / case_count),
        "retrieval_mrr": _round(reciprocal_rank_sum / case_count),
        "retrieval_recall_at_1": _round(recall_at_1 / case_count),
        "retrieval_recall_at_3": _round(recall_at_3 / case_count),
        "retrieval_recall_at_5": _round(recall_at_5 / case_count),
    }


def summarize_tool_trace(tool_trace: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(tool_trace)
    if total == 0:
        return {
            "tool_step_count": 0,
            "tool_success_rate": 0.0,
            "tool_error_rate": 0.0,
            "tool_timeout_rate": 0.0,
            "tool_avg_retries": 0.0,
            "tool_error_transient_rate": 0.0,
        }
    ok_count = sum(1 for item in tool_trace if item.get("status") == "ok")
    err_count = sum(1 for item in tool_trace if item.get("status") == "error")
    timeout_count = sum(1 for item in tool_trace if bool(item.get("timed_out")))
    retries = [max(int(item.get("attempts", 1)) - 1, 0) for item in tool_trace]
    transient_error_count = sum(
        1
        for item in tool_trace
        if item.get("status") == "error" and str(item.get("error_type") or "") == "transient"
    )
    return {
        "tool_step_count": total,
        "tool_success_rate": _round(ok_count / total),
        "tool_error_rate": _round(err_count / total),
        "tool_timeout_rate": _round(timeout_count / total),
        "tool_avg_retries": _round(sum(retries) / total),
        "tool_error_transient_rate": _round(transient_error_count / total),
    }


def summarize_task_results(task_results: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(task_results)
    if total == 0:
        return {
            "task_count": 0,
            "task_completed_count": 0,
            "task_failed_count": 0,
            "task_completion_rate": 0.0,
        }
    completed = sum(1 for item in task_results if str(item.get("status")) == "done")
    failed = sum(1 for item in task_results if str(item.get("status")) == "failed")
    return {
        "task_count": total,
        "task_completed_count": completed,
        "task_failed_count": failed,
        "task_completion_rate": _round(completed / total),
    }


def summarize_agentic_turn(
    *,
    task_results: list[dict[str, Any]],
    tool_trace: list[dict[str, Any]],
) -> dict[str, float | int]:
    task_metrics = summarize_task_results(task_results)
    tool_metrics = summarize_tool_trace(tool_trace)
    task_id_counts = Counter(str(item.get("task_id") or "") for item in task_results if str(item.get("task_id") or ""))
    rounds = max(task_id_counts.values(), default=0)
    return {
        **task_metrics,
        **tool_metrics,
        "agent_rounds": int(rounds),
    }
