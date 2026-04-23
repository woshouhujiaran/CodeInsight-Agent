from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.rag.embeddings import EmbeddingBackend
from app.rag.index_manifest import (
    compute_codebase_snapshot,
    compute_file_manifest,
    diff_file_manifests,
    load_file_manifest,
    write_file_manifest,
)
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


def _store_config_from_env() -> dict[str, Any]:
    return {
        "index_strategy": str(os.getenv("RAG_FAISS_INDEX_STRATEGY", "auto")).strip().lower(),
        "nlist": int(os.getenv("RAG_FAISS_NLIST", "128")),
        "m_pq": int(os.getenv("RAG_FAISS_M_PQ", "16")),
        "nprobe": int(os.getenv("RAG_FAISS_NPROBE", "16")),
        "ivf_min_points": int(os.getenv("RAG_FAISS_IVF_MIN_POINTS", "2000")),
    }


def _chunk_config_from_env() -> dict[str, Any]:
    return {
        "chunk_strategy": str(os.getenv("RAG_CHUNK_STRATEGY", "fixed")).strip().lower(),
    }


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
    root = Path(codebase_dir).resolve()
    index_dir = Path(index_dir)
    snapshot = snapshot or compute_codebase_snapshot(
        str(root),
        include_suffixes=include_suffixes,
        excluded_dirs=frozenset(excluded_dirs),
    )
    model_label = embedding_model_label(embedding)
    meta = read_index_meta(index_dir)
    store_cfg = _store_config_from_env()
    chunk_cfg = _chunk_config_from_env()

    def meta_compatible() -> bool:
        if not meta:
            return False
        if meta.get("backend_id") != embedding.backend_id:
            return False
        if int(meta.get("dim", -1)) != embedding.dim:
            return False
        m_saved = meta.get("model_name") or ""
        m_cur = model_label or ""
        if m_saved != m_cur:
            return False
        saved_store = meta.get("store") or {}
        for key, value in store_cfg.items():
            if str(saved_store.get(key, "")) != str(value):
                return False
        return True

    current_manifest = compute_file_manifest(
        str(root),
        include_suffixes=include_suffixes,
        excluded_dirs=frozenset(excluded_dirs),
    )
    previous_manifest = load_file_manifest(index_dir)
    changed_files, deleted_files = diff_file_manifests(previous_manifest, current_manifest)

    can_try_incremental = (
        (not force_reindex)
        and meta_compatible()
        and (index_dir / "index.faiss").exists()
        and (index_dir / "documents.json").exists()
        and previous_manifest is not None
    )
    if can_try_incremental:
        try:
            store = FaissVectorStore.load(index_dir, embedding, **store_cfg)
            if not changed_files and not deleted_files:
                logger.info("Loaded persisted RAG index from %s (no file changes)", index_dir)
                return store, {
                    "status": "loaded",
                    "index_dir": str(index_dir),
                    "snapshot": snapshot,
                    "incremental": {"changed_files": 0, "deleted_files": 0},
                }

            abs_changed = [root / rel for rel in changed_files]
            if deleted_files:
                store.delete_by_file_paths(str(root / rel) for rel in deleted_files)
            if abs_changed:
                store.delete_by_file_paths(str(path) for path in abs_changed)
                ingestor = CodeIngestor(
                    store=store,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    chunk_strategy=chunk_cfg["chunk_strategy"],
                )
                ingest_stats = ingestor.ingest_paths(abs_changed)
            else:
                ingest_stats = {"files_read": 0, "chunks_created": 0, "chunks_indexed": 0}

            store.save(index_dir)
            write_index_meta(
                index_dir,
                codebase_root=str(root),
                snapshot=snapshot,
                backend_id=embedding.backend_id,
                dim=embedding.dim,
                model_name=model_label,
                store_metadata=store.metadata(),
            )
            write_file_manifest(index_dir, current_manifest)
            logger.info(
                "Incremental RAG update finished changed=%d deleted=%d", len(changed_files), len(deleted_files)
            )
            return store, {
                "status": "incremental",
                "index_dir": str(index_dir),
                "snapshot": snapshot,
                "ingest": ingest_stats,
                "incremental": {
                    "changed_files": len(changed_files),
                    "deleted_files": len(deleted_files),
                },
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Incremental update failed, rebuilding full index: %s", exc)

    store = FaissVectorStore(embedding=embedding, **store_cfg)
    ingestor = CodeIngestor(
        store=store,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_strategy=chunk_cfg["chunk_strategy"],
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
        store_metadata=store.metadata(),
    )
    write_file_manifest(index_dir, current_manifest)
    logger.info("Built and saved RAG index under %s", index_dir)
    return store, {
        "status": "built",
        "index_dir": str(index_dir),
        "ingest": ingest_stats,
        "snapshot": snapshot,
    }
