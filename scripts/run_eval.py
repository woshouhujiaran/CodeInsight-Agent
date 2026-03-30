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
    summary = {
        "total_tasks": total_tasks,
        "passed_tasks": passed_tasks,
        "failed_tasks": failed_tasks,
        "pass_rate": pass_rate,
        "duration_seconds": round(duration_seconds, 3),
        "timestamp": iso_timestamp(),
        "success_rate": pass_rate,
        "avg_duration_ms": avg_duration_ms,
        "recovery_trigger_rate": 0.0,
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
