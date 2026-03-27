from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

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

    run_results: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("id", ""))
        category = str(task.get("category", "unknown"))
        query = str(task.get("query", ""))
        expected_keywords = task.get("expected_keywords", [])
        if not isinstance(expected_keywords, list):
            expected_keywords = []
        started = time.perf_counter()
        turn = agent.run(query)
        duration_ms = int((time.perf_counter() - started) * 1000)
        ok = task_success(turn.answer, [str(x) for x in expected_keywords])
        run_results.append(
            {
                "id": task_id,
                "category": category,
                "success": ok,
                "duration_ms": duration_ms,
                "recovery_applied": bool(turn.recovery_applied),
            }
        )
        print(f"[{task_id}] {category} success={ok} duration_ms={duration_ms} recovery={turn.recovery_applied}")

    summary = summarize(run_results)
    payload = {
        "tasks_path": str(tasks_path),
        "codebase_dir": str(codebase_dir),
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
