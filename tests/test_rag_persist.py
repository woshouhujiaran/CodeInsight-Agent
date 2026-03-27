from __future__ import annotations

from pathlib import Path

from app.rag.embeddings import HashEmbedding
from app.rag.index_manifest import compute_codebase_snapshot
from app.rag.ingest import CodeIngestor
from app.rag.load_or_build import load_or_build_vector_store
from app.rag.vector_store import FaissVectorStore, read_index_meta, write_index_meta


def test_faiss_save_load_search(tmp_path: Path) -> None:
    code_dir = tmp_path / "codebase"
    code_dir.mkdir()
    (code_dir / "math.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )

    emb = HashEmbedding(dim=384)
    store = FaissVectorStore(embedding=emb)
    ingestor = CodeIngestor(store=store, chunk_size=50, chunk_overlap=5)
    ingestor.ingest_directory(str(code_dir))

    index_dir = tmp_path / "index"
    store.save(index_dir)
    snap = compute_codebase_snapshot(
        str(code_dir.resolve()),
        include_suffixes=(".py", ".md", ".txt", ".js", ".ts", ".tsx", ".java"),
        excluded_dirs=frozenset({".git", ".pytest_cache", "__pycache__", ".venv", "venv"}),
    )
    write_index_meta(
        index_dir,
        codebase_root=str(code_dir.resolve()),
        snapshot=snap,
        backend_id="hash",
        dim=384,
        model_name=None,
    )

    loaded = FaissVectorStore.load(index_dir, HashEmbedding(dim=384))
    hits = loaded.search("add two numbers", top_k=2)
    assert hits, "expected at least one retrieval hit"
    assert any("math.py" in h["file_path"] for h in hits)


def test_load_or_build_skips_reingest_when_snapshot_matches(tmp_path: Path) -> None:
    code_dir = tmp_path / "cb"
    code_dir.mkdir()
    (code_dir / "x.py").write_text("x = 1\n", encoding="utf-8")

    index_dir = tmp_path / "idx"
    emb = HashEmbedding(dim=384)

    store1, meta1 = load_or_build_vector_store(str(code_dir), index_dir, emb, force_reindex=False)
    assert meta1["status"] == "built"
    assert store1.index.ntotal > 0

    store2, meta2 = load_or_build_vector_store(str(code_dir), index_dir, emb, force_reindex=False)
    assert meta2["status"] == "loaded"
    assert store2.index.ntotal == store1.index.ntotal

    m = read_index_meta(index_dir)
    assert m is not None
    assert m["backend_id"] == "hash"


def test_force_reindex_rebuilds(tmp_path: Path) -> None:
    code_dir = tmp_path / "cb2"
    code_dir.mkdir()
    (code_dir / "a.py").write_text("a = 1\n", encoding="utf-8")

    index_dir = tmp_path / "idx2"
    emb = HashEmbedding(dim=384)
    load_or_build_vector_store(str(code_dir), index_dir, emb, force_reindex=False)

    (code_dir / "b.py").write_text("b = 2\n", encoding="utf-8")
    _, meta = load_or_build_vector_store(str(code_dir), index_dir, emb, force_reindex=False)
    # Snapshot changed -> should rebuild automatically
    assert meta["status"] == "built"

    load_or_build_vector_store(str(code_dir), index_dir, emb, force_reindex=True)
    m = read_index_meta(index_dir)
    assert m is not None
