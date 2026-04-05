from __future__ import annotations

from pathlib import Path
from typing import Any

from app.rag.embeddings import EmbeddingBackend
from app.rag.index_manifest import compute_codebase_snapshot
from app.rag.ingest import CodeIngestor
from app.rag.vector_store import (
    FaissVectorStore,
    embedding_model_label,
    read_index_meta,
    write_index_meta,
)
from app.utils.logger import get_logger

logger = get_logger("codeinsight.rag.load_or_build")

DEFAULT_INCLUDE_SUFFIXES: tuple[str, ...] = (
    ".py",
    ".md",
    ".txt",
    ".js",
    ".ts",
    ".tsx",
    ".java",
)
DEFAULT_EXCLUDED_DIRS: tuple[str, ...] = (
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".venv",
    "venv",
    "outputs",
    "data",
    "build",
    "dist",
    "coverage",
)


def compute_vector_store_snapshot(codebase_dir: str) -> str:
    return compute_codebase_snapshot(
        str(Path(codebase_dir).resolve()),
        include_suffixes=DEFAULT_INCLUDE_SUFFIXES,
        excluded_dirs=frozenset(DEFAULT_EXCLUDED_DIRS),
    )


def load_or_build_vector_store(
    codebase_dir: str,
    index_dir: Path,
    embedding: EmbeddingBackend,
    *,
    force_reindex: bool = False,
    snapshot: str | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    include_suffixes: tuple[str, ...] = DEFAULT_INCLUDE_SUFFIXES,
    excluded_dirs: tuple[str, ...] = DEFAULT_EXCLUDED_DIRS,
) -> tuple[FaissVectorStore, dict[str, Any]]:
    """
    Load persisted FAISS index when meta snapshot matches the codebase; otherwise ingest and save.
    """
    root = Path(codebase_dir).resolve()
    index_dir = Path(index_dir)
    snapshot = snapshot or compute_codebase_snapshot(
        str(root),
        include_suffixes=include_suffixes,
        excluded_dirs=frozenset(excluded_dirs),
    )
    model_label = embedding_model_label(embedding)
    meta = read_index_meta(index_dir)

    def meta_compatible() -> bool:
        if not meta:
            return False
        if meta.get("snapshot") != snapshot:
            return False
        if meta.get("backend_id") != embedding.backend_id:
            return False
        if int(meta.get("dim", -1)) != embedding.dim:
            return False
        m_saved = meta.get("model_name") or ""
        m_cur = model_label or ""
        if m_saved != m_cur:
            return False
        return True

    if not force_reindex and meta_compatible():
        try:
            store = FaissVectorStore.load(index_dir, embedding)
            logger.info("Loaded persisted RAG index from %s", index_dir)
            return store, {
                "status": "loaded",
                "index_dir": str(index_dir),
                "snapshot": snapshot,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load persisted index, rebuilding: %s", exc)

    store = FaissVectorStore(embedding=embedding)
    ingestor = CodeIngestor(
        store=store,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    ingest_stats = ingestor.ingest_directory(str(root), include_suffixes=include_suffixes)
    store.save(index_dir)
    write_index_meta(
        index_dir,
        codebase_root=str(root),
        snapshot=snapshot,
        backend_id=embedding.backend_id,
        dim=embedding.dim,
        model_name=model_label,
    )
    logger.info("Built and saved RAG index under %s", index_dir)
    return store, {
        "status": "built",
        "index_dir": str(index_dir),
        "ingest": ingest_stats,
        "snapshot": snapshot,
    }
