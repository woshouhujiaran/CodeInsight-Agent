from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.web.service import WebAgentService
from scripts._common import (
    EVAL_RESULT_PATH,
    OUTPUTS_DIR,
    REPO_ROOT,
    build_index_for_workspace,
    build_retriever_for_workspace,
    ensure_outputs_dir,
    environment_summary,
    iso_timestamp,
    resolve_workspace_root,
    write_json,
)

TaskSpec = dict[str, Any]

DEFAULT_TASKS: list[TaskSpec] = [
    {"name": "workspace_root_exists", "kind": "path_exists", "path": ".", "path_type": "dir"},
    {"name": "rag_index_ready", "kind": "build_index", "workspace_root": ".", "force_reindex": False},
    {
        "name": "retrieval_session_store",
        "kind": "retrieval_expectation",
        "query": "session store persistence history",
        "expected_path_contains": "app/web/session_store.py",
        "top_k": 5,
    },
    {
        "name": "retrieval_run_eval",
        "kind": "retrieval_expectation",
        "query": "run eval summary script",
        "expected_path_contains": "scripts/run_eval.py",
        "top_k": 5,
    },
    {"name": "web_eval_readable", "kind": "web_eval_readable"},
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal offline evaluation suite.")
    parser.add_argument(
        "--workspace-root",
        default=".",
        help="Workspace root used by evaluation tasks. Defaults to the repository root.",
    )
    return parser


def load_task_specs(repo_root: Path) -> tuple[list[TaskSpec], Path | None]:
    tasks_path = repo_root / "eval" / "tasks.json"
    if not tasks_path.is_file():
        return list(DEFAULT_TASKS), None

    with tasks_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    tasks = payload.get("tasks") if isinstance(payload, dict) else payload
    if not isinstance(tasks, list):
        raise ValueError(f"Invalid eval task file: expected a list under {tasks_path}")
    return [dict(item) for item in tasks], tasks_path


def build_result_payload(
    *,
    tasks: list[dict[str, Any]],
    workspace_root: Path,
    tasks_path: Path | None,
    duration_seconds: float,
) -> dict[str, Any]:
    total_tasks = len(tasks)
    passed_tasks = sum(1 for task in tasks if task.get("status") == "passed")
    failed_tasks = total_tasks - passed_tasks
    pass_rate = round((passed_tasks / total_tasks) if total_tasks else 0.0, 4)
    avg_duration_ms = round(
        (sum(float(task.get("duration_seconds", 0.0)) for task in tasks) * 1000 / total_tasks) if total_tasks else 0.0,
        2,
    )
    retrieval_tasks = [task for task in tasks if str(task.get("kind") or "") == "retrieval_expectation"]
    retrieval_hits = sum(1 for task in retrieval_tasks if task.get("status") == "passed")
    retrieval_hit_rate = round((retrieval_hits / len(retrieval_tasks)) if retrieval_tasks else 0.0, 4)
    retrieval_mrr = round(
        (
            sum(float((task.get("details") or {}).get("reciprocal_rank", 0.0)) for task in retrieval_tasks)
            / len(retrieval_tasks)
        )
        if retrieval_tasks
        else 0.0,
        4,
    )
    summary = {
        "total_tasks": total_tasks,
        "passed_tasks": passed_tasks,
        "failed_tasks": failed_tasks,
        "pass_rate": pass_rate,
        "duration_seconds": round(duration_seconds, 3),
        "timestamp": iso_timestamp(),
        "success_rate": pass_rate,
        "task_completion_rate": pass_rate,
        "avg_duration_ms": avg_duration_ms,
        "recovery_trigger_rate": 0.0,
        "retrieval_case_count": len(retrieval_tasks),
        "retrieval_hit_rate": retrieval_hit_rate,
        "retrieval_mrr": retrieval_mrr,
    }
    return {
        "tasks_path": str(tasks_path.resolve()) if tasks_path is not None else None,
        "workspace_root": str(workspace_root),
        "summary": summary,
        "tasks": tasks,
        "environment": environment_summary(workspace_root=workspace_root),
    }


def run_task(
    spec: TaskSpec,
    *,
    workspace_root: Path,
    payload_factory: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    name = str(spec.get("name") or spec.get("kind") or "unnamed_task")
    started = time.perf_counter()
    error: str | None = None
    details: dict[str, Any] = {}

    try:
        kind = str(spec.get("kind") or "").strip()
        if kind == "path_exists":
            details = _run_path_exists_task(spec, workspace_root=workspace_root)
        elif kind == "build_index":
            details = _run_build_index_task(spec, workspace_root=workspace_root)
        elif kind == "retrieval_expectation":
            details = _run_retrieval_expectation_task(spec, workspace_root=workspace_root)
        elif kind == "web_eval_readable":
            details = _run_web_eval_readable_task(payload_factory=payload_factory)
        else:
            raise ValueError(f"unsupported task kind: {kind!r}")
        status = "passed"
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = str(exc)

    duration_seconds = round(time.perf_counter() - started, 3)
    return {
        "name": name,
        "kind": str(spec.get("kind") or ""),
        "status": status,
        "error": error,
        "duration": duration_seconds,
        "duration_seconds": duration_seconds,
        "details": details,
    }


def _run_path_exists_task(spec: TaskSpec, *, workspace_root: Path) -> dict[str, Any]:
    raw_path = str(spec.get("path") or ".")
    candidate = Path(raw_path)
    resolved = candidate.resolve() if candidate.is_absolute() else (workspace_root / candidate).resolve()
    path_type = str(spec.get("path_type") or "").strip().lower()
    if not resolved.exists():
        raise FileNotFoundError(f"path does not exist: {resolved}")
    if path_type == "dir" and not resolved.is_dir():
        raise NotADirectoryError(f"path is not a directory: {resolved}")
    if path_type == "file" and not resolved.is_file():
        raise FileNotFoundError(f"path is not a file: {resolved}")
    return {"path": str(resolved), "path_type": path_type or "any"}


def _run_build_index_task(spec: TaskSpec, *, workspace_root: Path) -> dict[str, Any]:
    target_root = spec.get("workspace_root")
    resolved_root = resolve_workspace_root(workspace_root if target_root in (None, "", ".") else target_root)
    result = build_index_for_workspace(
        resolved_root,
        force_reindex=bool(spec.get("force_reindex", False)),
    )
    return {
        "status": result.get("status"),
        "index_dir": result.get("index_dir"),
        "snapshot": result.get("snapshot"),
        "document_count": result.get("document_count"),
    }


def _run_retrieval_expectation_task(spec: TaskSpec, *, workspace_root: Path) -> dict[str, Any]:
    query = str(spec.get("query") or "").strip()
    expected = str(spec.get("expected_path_contains") or "").strip().replace("\\", "/").lower()
    if not query:
        raise ValueError("retrieval_expectation requires query")
    if not expected:
        raise ValueError("retrieval_expectation requires expected_path_contains")
    top_k = int(spec.get("top_k") or 5)
    force_reindex = bool(spec.get("force_reindex", False))
    target_root = spec.get("workspace_root")
    resolved_root = resolve_workspace_root(workspace_root if target_root in (None, "", ".") else target_root)
    retriever, meta = build_retriever_for_workspace(resolved_root, force_reindex=force_reindex)
    hits = retriever.retrieve(query=query, top_k=top_k)
    reciprocal_rank = 0.0
    matched_path = ""
    for index, hit in enumerate(hits, start=1):
        path = str(hit.get("file_path") or "").replace("\\", "/").lower()
        if expected in path:
            reciprocal_rank = round(1.0 / index, 4)
            matched_path = str(hit.get("file_path") or "")
            break
    if reciprocal_rank == 0.0:
        preview = [str(hit.get("file_path") or "") for hit in hits[:top_k]]
        raise AssertionError(f"expected `{expected}` in top-{top_k} hits, got {preview}")
    return {
        "query": query,
        "expected_path_contains": expected,
        "matched_path": matched_path,
        "top_k": top_k,
        "hit_count": len(hits),
        "reciprocal_rank": reciprocal_rank,
        "index_status": meta.get("status"),
        "index_dir": meta.get("index_dir"),
    }


def _run_web_eval_readable_task(*, payload_factory: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    payload = payload_factory()
    write_json(EVAL_RESULT_PATH, payload)
    service = WebAgentService(repo_root=REPO_ROOT, outputs_dir=OUTPUTS_DIR)
    latest = service.get_latest_eval_result()
    if latest.get("path") != str(EVAL_RESULT_PATH.resolve()):
        raise ValueError("web service did not return outputs/eval_result.json as the latest eval result")
    loaded = latest.get("payload")
    if not isinstance(loaded, dict):
        raise ValueError("web service returned an empty eval payload")
    if not isinstance(loaded.get("summary"), dict):
        raise ValueError("web service payload is missing summary")
    return {"path": latest["path"]}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_outputs_dir()

    try:
        workspace_root = resolve_workspace_root(args.workspace_root)
        task_specs, tasks_path = load_task_specs(REPO_ROOT)
    except Exception as exc:  # noqa: BLE001
        print(f"[run_eval] failed to initialize evaluation: {exc}", file=sys.stderr)
        return 1

    started = time.perf_counter()
    task_results: list[dict[str, Any]] = []

    def current_payload() -> dict[str, Any]:
        return build_result_payload(
            tasks=list(task_results),
            workspace_root=workspace_root,
            tasks_path=tasks_path,
            duration_seconds=time.perf_counter() - started,
        )

    for spec in task_specs:
        task_result = run_task(
            spec,
            workspace_root=workspace_root,
            payload_factory=current_payload,
        )
        task_results.append(task_result)
        print(f"[{task_result['status']}] {task_result['name']} ({task_result['duration_seconds']}s)")
        if task_result["error"]:
            print(f"  error: {task_result['error']}")

    payload = build_result_payload(
        tasks=task_results,
        workspace_root=workspace_root,
        tasks_path=tasks_path,
        duration_seconds=time.perf_counter() - started,
    )
    write_json(EVAL_RESULT_PATH, payload)
    print(f"Wrote eval result: {EVAL_RESULT_PATH.resolve()}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
