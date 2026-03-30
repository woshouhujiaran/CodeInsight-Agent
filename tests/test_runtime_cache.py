from __future__ import annotations

from pathlib import Path

import app.runtime as runtime
from app.rag.embeddings import HashEmbedding


class _FakeStore:
    def search(self, query: str, top_k: int = 5) -> list[dict[str, str | float]]:
        return []


def test_create_agent_reuses_cached_runtime_resources(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_text("x = 1\n", encoding="utf-8")

    embedding_calls: list[int] = []
    vector_store_calls: list[tuple[str, str, bool]] = []

    def fake_create_embedding_backend() -> HashEmbedding:
        embedding_calls.append(1)
        return HashEmbedding(dim=16)

    def fake_load_or_build_vector_store(
        codebase_dir: str,
        index_dir: Path,
        embedding: HashEmbedding,
        *,
        force_reindex: bool = False,
    ) -> tuple[_FakeStore, dict[str, str]]:
        vector_store_calls.append((str(codebase_dir), str(index_dir), force_reindex))
        return _FakeStore(), {"status": "built", "index_dir": str(index_dir), "snapshot": "snap"}

    runtime.reset_runtime_caches()
    monkeypatch.setattr(runtime, "create_embedding_backend", fake_create_embedding_backend)
    monkeypatch.setattr(runtime, "load_or_build_vector_store", fake_load_or_build_vector_store)

    first = runtime.create_agent_from_env(str(workspace))
    second = runtime.create_agent_from_env(str(workspace))

    assert first.llm is second.llm
    assert len(embedding_calls) == 1
    assert len(vector_store_calls) == 1


def test_create_agent_invalidates_vector_store_cache_when_snapshot_changes(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")

    vector_store_calls: list[tuple[str, str, bool]] = []

    def fake_load_or_build_vector_store(
        codebase_dir: str,
        index_dir: Path,
        embedding: HashEmbedding,
        *,
        force_reindex: bool = False,
    ) -> tuple[_FakeStore, dict[str, str]]:
        vector_store_calls.append((str(codebase_dir), str(index_dir), force_reindex))
        return _FakeStore(), {"status": "built", "index_dir": str(index_dir), "snapshot": "snap"}

    runtime.reset_runtime_caches()
    monkeypatch.setattr(runtime, "create_embedding_backend", lambda: HashEmbedding(dim=16))
    monkeypatch.setattr(runtime, "load_or_build_vector_store", fake_load_or_build_vector_store)

    runtime.create_agent_from_env(str(workspace))
    target.write_text("x = 2\n", encoding="utf-8")
    runtime.create_agent_from_env(str(workspace))

    assert len(vector_store_calls) == 2
