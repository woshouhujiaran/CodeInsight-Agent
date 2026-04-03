from __future__ import annotations

import hashlib
import os
from pathlib import Path
from threading import RLock
from typing import Any

from app.agent.agent import CodeAgent
from app.agent.executor import Executor
from app.agent.memory import ConversationMemory
from app.agent.planner import Planner
from app.agent.tool_registry import ToolRegistry
from app.llm.llm import LLMClient
from app.rag.embeddings import EmbeddingBackend, create_embedding_backend
from app.rag.load_or_build import compute_vector_store_snapshot, load_or_build_vector_store
from app.rag.retriever import CodeRetriever
from app.rag.vector_store import embedding_model_label
from app.tools.analyze_tool import AnalyzeTool
from app.tools.filesystem_tools import GrepTool, ListDirTool, ReadFileTool
from app.tools.optimize_tool import OptimizeTool
from app.tools.run_command_tool import RunCommandTool
from app.tools.search_tool import SearchTool
from app.tools.test_tool import TestTool
from app.tools.write_tools import ApplyPatchTool, WriteFileTool
from app.utils.logger import get_logger

_CACHE_LOCK = RLock()
_LLM_CACHE: dict[tuple[str, str], LLMClient] = {}
_EMBEDDING_CACHE: dict[tuple[str, ...], EmbeddingBackend] = {}
_VECTOR_STORE_CACHE: dict[tuple[str, str, tuple[str, ...], str], tuple[Any, dict[str, Any]]] = {}


def default_index_dir(codebase_dir: str) -> Path:
    """Stable per-codebase path under data/index/ to avoid mixing indexes."""
    digest = hashlib.sha256(str(Path(codebase_dir).resolve()).encode("utf-8")).hexdigest()[:16]
    return Path("data/index") / digest


def reset_runtime_caches() -> None:
    with _CACHE_LOCK:
        _LLM_CACHE.clear()
        _EMBEDDING_CACHE.clear()
        _VECTOR_STORE_CACHE.clear()


def _llm_cache_key_from_env() -> tuple[str, str]:
    llm_provider = os.getenv("LLM_PROVIDER", "deepseek")
    llm_model = os.getenv("LLM_MODEL", "deepseek-chat")
    return llm_provider, llm_model


def create_llm_from_env() -> LLMClient:
    key = _llm_cache_key_from_env()
    with _CACHE_LOCK:
        cached = _LLM_CACHE.get(key)
        if cached is not None:
            return cached

    llm = LLMClient(model=key[1], provider=key[0])
    with _CACHE_LOCK:
        _LLM_CACHE[key] = llm
    return llm


def _embedding_cache_key_from_env() -> tuple[str, ...]:
    backend = os.getenv("EMBEDDING_BACKEND", "sentence_transformers").strip().lower()
    if backend == "hash":
        return (backend, os.getenv("EMBEDDING_DIM", "384").strip())
    if backend == "openai":
        return (
            backend,
            os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip(),
            os.getenv("OPENAI_BASE_URL", "").strip(),
            os.getenv("OPENAI_API_KEY", "").strip(),
        )
    return (backend, os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip())


def _get_cached_embedding_backend() -> EmbeddingBackend:
    key = _embedding_cache_key_from_env()
    with _CACHE_LOCK:
        cached = _EMBEDDING_CACHE.get(key)
        if cached is not None:
            return cached

    embedding = create_embedding_backend()
    with _CACHE_LOCK:
        _EMBEDDING_CACHE[key] = embedding
    return embedding


def _embedding_runtime_signature(embedding: EmbeddingBackend) -> tuple[str, ...]:
    return (
        embedding.backend_id,
        str(embedding.dim),
        str(embedding_model_label(embedding) or ""),
    )


def _vector_store_cache_key(
    workspace_root: str,
    index_dir: Path,
    embedding: EmbeddingBackend,
    snapshot: str,
) -> tuple[str, str, tuple[str, ...], str]:
    return (
        str(Path(workspace_root).resolve()),
        str(Path(index_dir).resolve()),
        _embedding_runtime_signature(embedding),
        snapshot,
    )


def _get_cached_vector_store(
    workspace_root: str,
    *,
    embedding: EmbeddingBackend,
    index_dir: Path,
    force_reindex: bool,
) -> tuple[Any, dict[str, Any]]:
    resolved_root = str(Path(workspace_root).resolve())
    resolved_index_dir = Path(index_dir).resolve()
    snapshot = compute_vector_store_snapshot(resolved_root)
    key = _vector_store_cache_key(resolved_root, resolved_index_dir, embedding, snapshot)

    with _CACHE_LOCK:
        cached = _VECTOR_STORE_CACHE.get(key)
        if cached is not None and not force_reindex:
            store, meta = cached
            cached_meta = dict(meta)
            cached_meta["status"] = "cached"
            return store, cached_meta

    store, rag_meta = load_or_build_vector_store(
        resolved_root,
        resolved_index_dir,
        embedding,
        force_reindex=force_reindex,
        snapshot=snapshot,
    )
    cache_meta = dict(rag_meta)
    cache_meta["snapshot"] = snapshot
    cache_meta["index_dir"] = str(resolved_index_dir)
    scope = key[:3]
    with _CACHE_LOCK:
        stale_keys = [item for item in _VECTOR_STORE_CACHE if item[:3] == scope and item != key]
        for stale_key in stale_keys:
            _VECTOR_STORE_CACHE.pop(stale_key, None)
        _VECTOR_STORE_CACHE[key] = (store, cache_meta)
    return store, rag_meta


def create_agent_from_env(
    workspace_root: str,
    *,
    memory: ConversationMemory | None = None,
    top_k: int = 5,
    force_reindex: bool = False,
    allow_write: bool = False,
    allow_shell: bool = False,
    test_command: str = "",
    index_dir: Path | None = None,
) -> CodeAgent:
    logger = get_logger("codeinsight.runtime")
    llm = create_llm_from_env()

    embedding = _get_cached_embedding_backend()
    idx_path = Path(index_dir or default_index_dir(workspace_root)).resolve()
    store, rag_meta = _get_cached_vector_store(
        workspace_root,
        embedding=embedding,
        index_dir=idx_path,
        force_reindex=force_reindex,
    )
    logger.info("RAG: %s", rag_meta)

    resolved_root = Path(workspace_root).resolve()
    retriever = CodeRetriever(store=store)
    registry = ToolRegistry()
    registry.register(SearchTool(retriever=retriever, top_k=top_k))
    registry.register(AnalyzeTool(llm=llm))
    registry.register(OptimizeTool(llm=llm))
    registry.register(TestTool(llm=llm))
    registry.register(ReadFileTool(workspace_root=resolved_root))
    registry.register(ListDirTool(workspace_root=resolved_root))
    registry.register(GrepTool(workspace_root=resolved_root))

    if allow_write:
        registry.register(ApplyPatchTool(workspace_root=resolved_root))
        registry.register(WriteFileTool(workspace_root=resolved_root))
        logger.info("allow_write=True: registered apply_patch_tool and write_file_tool")

    if allow_shell:
        allowed_commands = [test_command] if str(test_command or "").strip() else []
        registry.register(
            RunCommandTool(
                workspace_root=resolved_root,
                allowed_commands=allowed_commands,
            )
        )
        logger.info("allow_shell=True: registered run_command_tool")

    planner = Planner(llm=llm, write_tools_enabled=allow_write)
    executor = Executor(registry=registry)
    return CodeAgent(
        planner=planner,
        executor=executor,
        llm=llm,
        memory=memory,
        workspace_root=str(resolved_root),
    )
