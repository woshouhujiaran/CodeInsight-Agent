from __future__ import annotations

from pathlib import Path

from app.rag.embeddings import HashEmbedding
from app.web.service import WebAgentService
from scripts import _common
from scripts.clear_state import apply_cleanup, collect_cleanup_targets
from scripts.run_eval import build_result_payload, run_task


class _FakeIndex:
    def __init__(self, total: int) -> None:
        self.ntotal = total


class _FakeStore:
    def __init__(self, total: int) -> None:
        self.index = _FakeIndex(total)


def test_build_index_for_workspace_reuses_runtime_logic(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_text("x = 1\n", encoding="utf-8")

    calls: dict[str, object] = {}

    monkeypatch.setattr(_common, "create_embedding_backend", lambda: HashEmbedding(dim=16))

    def fake_load_or_build(
        codebase_dir: str,
        index_dir: Path,
        embedding: HashEmbedding,
        *,
        force_reindex: bool = False,
    ) -> tuple[_FakeStore, dict[str, str]]:
        calls["codebase_dir"] = codebase_dir
        calls["index_dir"] = str(index_dir)
        calls["force_reindex"] = force_reindex
        calls["embedding_dim"] = embedding.dim
        return _FakeStore(3), {"status": "built", "snapshot": "snap", "index_dir": str(index_dir)}

    monkeypatch.setattr(_common, "load_or_build_vector_store", fake_load_or_build)

    result = _common.build_index_for_workspace(workspace)

    assert calls["codebase_dir"] == str(workspace.resolve())
    assert calls["index_dir"] == str(_common.default_index_dir(str(workspace.resolve())).resolve())
    assert calls["force_reindex"] is False
    assert calls["embedding_dim"] == 16
    assert result["status"] == "built"
    assert result["snapshot"] == "snap"
    assert result["document_count"] == 3


def test_collect_cleanup_targets_respects_keep_eval_result(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    generated = outputs_dir / "generated_test_demo.py"
    generated.write_text("print('x')\n", encoding="utf-8")
    (outputs_dir / "eval_result.json").write_text("{}", encoding="utf-8")
    (outputs_dir / ".pytest_cache").mkdir()
    pycache_dir = tmp_path / "pkg" / "__pycache__"
    pycache_dir.mkdir(parents=True)
    (pycache_dir / "x.pyc").write_bytes(b"x")

    targets = collect_cleanup_targets(
        tmp_path,
        keep_eval_result=True,
        include_pytest_cache=True,
        include_pycache=True,
    )
    target_paths = {target.path for target in targets}

    assert generated.resolve() in target_paths
    assert (outputs_dir / ".pytest_cache").resolve() in target_paths
    assert pycache_dir.resolve() in target_paths
    assert (outputs_dir / "eval_result.json").resolve() not in target_paths

    summary = apply_cleanup(targets, dry_run=True)
    assert summary["deleted_files"] == 2
    assert summary["deleted_dirs"] == 2


def test_eval_payload_is_readable_by_web_service(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    payload = build_result_payload(
        tasks=[
            {"name": "workspace_root_exists", "kind": "path_exists", "status": "passed", "error": None, "duration_seconds": 0.1},
            {
                "name": "retrieval_session_store",
                "kind": "retrieval_expectation",
                "status": "failed",
                "error": "boom",
                "duration_seconds": 0.2,
                "details": {"reciprocal_rank": 0.0},
            },
        ],
        workspace_root=tmp_path,
        tasks_path=None,
        duration_seconds=0.5,
    )

    _common.write_json(outputs_dir / "eval_result.json", payload)
    service = WebAgentService(repo_root=tmp_path, outputs_dir=outputs_dir)
    latest = service.get_latest_eval_result()

    assert latest["path"] == str((outputs_dir / "eval_result.json").resolve())
    assert latest["payload"]["summary"]["total_tasks"] == 2
    assert latest["payload"]["summary"]["passed_tasks"] == 1
    assert latest["payload"]["summary"]["failed_tasks"] == 1
    assert latest["payload"]["summary"]["pass_rate"] == 0.5
    assert latest["payload"]["summary"]["retrieval_case_count"] == 1
    assert latest["payload"]["summary"]["retrieval_hit_rate"] == 0.0
    assert "python_version" in latest["payload"]["environment"]


def test_run_eval_retrieval_task_reports_reciprocal_rank(tmp_path: Path, monkeypatch) -> None:
    class _FakeRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, object]]:
            assert query == "session store"
            assert top_k == 3
            return [
                {"file_path": "app/web/service.py"},
                {"file_path": "app/web/session_store.py"},
            ]

    monkeypatch.setattr("scripts.run_eval.build_retriever_for_workspace", lambda *args, **kwargs: (_FakeRetriever(), {"status": "loaded", "index_dir": "x"}))

    result = run_task(
        {
            "name": "retrieval_session_store",
            "kind": "retrieval_expectation",
            "query": "session store",
            "expected_path_contains": "app/web/session_store.py",
            "top_k": 3,
        },
        workspace_root=tmp_path,
        payload_factory=lambda: {},
    )

    assert result["status"] == "passed"
    assert result["details"]["matched_path"] == "app/web/session_store.py"
    assert result["details"]["reciprocal_rank"] == 0.5
