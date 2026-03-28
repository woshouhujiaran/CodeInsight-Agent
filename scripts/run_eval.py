from __future__ import annotations

"""
端到端评测默认走 agent.run（Planner 路径）。传入 --agentic 时对每条任务调用 run_agentic
（可用 tasks.json 单条字段 "agentic": true/false 覆盖 CLI 默认值）。
模型侧工具策略由 CodeAgent.run_agentic 的 system 提示（含 AGENTIC_TOOL_USE_POLICY）约束。
"""

import argparse
import json
from pathlib import Path
import time
from typing import Any

from app.agent.agent import CodeAgent
from app.main import build_agent, default_index_dir


def load_tasks(tasks_path: Path) -> list[dict[str, Any]]:
    with tasks_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("tasks file must be a JSON array")
    if len(data) < 20:
        raise ValueError("tasks must contain at least 20 items")
    return data


def task_success(answer: str, expected_keywords: list[str]) -> bool:
    text = (answer or "").lower()
    for kw in expected_keywords:
        if str(kw).lower() in text:
            return True
    return False


def task_use_agentic(task: dict[str, Any], *, cli_agentic: bool) -> bool:
    """Per-task 'agentic' key overrides CLI when present (bool)."""
    if "agentic" in task:
        return bool(task["agentic"])
    return cli_agentic


def run_one_eval_task(
    agent: CodeAgent,
    task: dict[str, Any],
    *,
    cli_agentic: bool,
    max_turns: int,
) -> dict[str, Any]:
    """Run a single eval task; mock-friendly (no API required in tests)."""
    task_id = str(task.get("id", ""))
    category = str(task.get("category", "unknown"))
    query = str(task.get("query", ""))
    expected_keywords = task.get("expected_keywords", [])
    if not isinstance(expected_keywords, list):
        expected_keywords = []
    use_agentic = task_use_agentic(task, cli_agentic=cli_agentic)
    started = time.perf_counter()
    if use_agentic:
        turn = agent.run_agentic(query, max_turns=max_turns)
        recovery_applied = False
    else:
        turn = agent.run(query)
        recovery_applied = bool(getattr(turn, "recovery_applied", False))
    duration_ms = int((time.perf_counter() - started) * 1000)
    ok = task_success(turn.answer, [str(x) for x in expected_keywords])
    return {
        "id": task_id,
        "category": category,
        "success": ok,
        "duration_ms": duration_ms,
        "recovery_applied": recovery_applied,
        "agentic": use_agentic,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"total": 0, "success_rate": 0.0, "avg_duration_ms": 0.0, "recovery_trigger_rate": 0.0}
    success_count = sum(1 for r in results if r.get("success"))
    avg_duration_ms = sum(float(r.get("duration_ms", 0)) for r in results) / total
    recovery_count = sum(1 for r in results if r.get("recovery_applied"))
    return {
        "total": total,
        "success_rate": round(success_count / total, 4),
        "avg_duration_ms": round(avg_duration_ms, 2),
        "recovery_trigger_rate": round(recovery_count / total, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CodeInsight-Agent E2E evaluation baseline")
    parser.add_argument("--tasks", default="eval/tasks.json", help="Task dataset path")
    parser.add_argument("--output", default="outputs/eval_result.json", help="Output JSON path")
    parser.add_argument("--codebase-dir", default="data/codebase", help="Codebase directory for RAG index")
    parser.add_argument("--top-k", type=int, default=5, help="Retriever top-k")
    parser.add_argument("--force-reindex", action="store_true", help="Force reindex before evaluation")
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="默认对每条任务使用 run_agentic（单条任务可用 JSON 字段 agentic: true/false 覆盖）",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=8,
        help="agentic 模式下每任务 max_turns（默认 8）",
    )
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    tasks = load_tasks(tasks_path)
    codebase_dir = Path(args.codebase_dir)
    codebase_dir.mkdir(parents=True, exist_ok=True)

    agent = build_agent(
        codebase_dir=str(codebase_dir),
        top_k=args.top_k,
        index_dir=default_index_dir(str(codebase_dir)),
        force_reindex=args.force_reindex,
    )

    max_turns = max(1, int(args.max_turns))
    run_results: list[dict[str, Any]] = []
    for task in tasks:
        row = run_one_eval_task(
            agent,
            task,
            cli_agentic=bool(args.agentic),
            max_turns=max_turns,
        )
        run_results.append(row)
        task_id = row["id"]
        category = row["category"]
        ok = row["success"]
        duration_ms = row["duration_ms"]
        recovery = row["recovery_applied"]
        ag = row["agentic"]
        print(
            f"[{task_id}] {category} success={ok} duration_ms={duration_ms} "
            f"recovery={recovery} agentic={ag}"
        )

    summary = summarize(run_results)
    payload = {
        "tasks_path": str(tasks_path),
        "codebase_dir": str(codebase_dir),
        "eval_options": {
            "cli_agentic": bool(args.agentic),
            "max_turns": max_turns,
        },
        "summary": summary,
        "results": run_results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved eval result to: {out_path}")


if __name__ == "__main__":
    main()
