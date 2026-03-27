from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from app.rag.embeddings import EmbeddingBackend
from app.utils.logger import get_logger

try:
    import faiss  # type: ignore
except Exception:  # noqa: BLE001
    faiss = None


@dataclass
class CodeDocument:
    file_path: str
    content: str
    chunk_id: str = ""


class FaissVectorStore:
    """
    FAISS vector store for code snippets.

    Embeddings are provided by an EmbeddingBackend (hash / sentence-transformers / OpenAI API).
    Use save()/load() for persistent indexes to avoid full re-ingest on each startup.
    """

    def __init__(self, embedding: EmbeddingBackend, logger_name: str = "codeinsight.rag.faiss") -> None:
        if faiss is None:
            raise ImportError("faiss is not installed. Please install faiss-cpu first.")
        self.embedding = embedding
        self.dim = embedding.dim
        self.logger = get_logger(logger_name)
        self.index = faiss.IndexFlatIP(self.dim)
        self.documents: list[CodeDocument] = []

    def add_documents(self, documents: Iterable[CodeDocument]) -> int:
        docs = list(documents)
        if not docs:
            return 0

        texts = [d.content for d in docs]
        vectors = self.embedding.embed_texts(texts).astype("float32")
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.documents.extend(docs)
        self.logger.info("Added %d documents to FAISS index.", len(docs))
        return len(docs)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, str | float]]:
        if not query.strip() or self.index.ntotal == 0:
            return []

        q = self.embedding.embed_query(query).reshape(1, -1).astype("float32")
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, min(top_k, self.index.ntotal))

        results: list[dict[str, str | float]] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            results.append(
                {
                    "file_path": doc.file_path,
                    "content": doc.content,
                    "chunk_id": doc.chunk_id,
                    "score": float(score),
                }
            )
        return results

    def save(self, index_dir: Path) -> None:
        """Persist FAISS index + documents + metadata sidecar."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "index.faiss"))
        docs_path = index_dir / "documents.json"
        with docs_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(d) for d in self.documents], f, ensure_ascii=False)

    @classmethod
    def load(cls, index_dir: Path, embedding: EmbeddingBackend) -> FaissVectorStore:
        """Load index from disk; `embedding` must match the backend used when building."""
        index_dir = Path(index_dir)
        if faiss is None:
            raise ImportError("faiss is not installed. Please install faiss-cpu first.")

        index_path = index_dir / "index.faiss"
        docs_path = index_dir / "documents.json"
        if not index_path.exists() or not docs_path.exists():
            raise FileNotFoundError(f"Missing index files under {index_dir}")

        faiss_index = faiss.read_index(str(index_path))
        if faiss_index.d != embedding.dim:
            raise ValueError(
                f"Index dimension {faiss_index.d} does not match embedding.dim={embedding.dim}"
            )

        with docs_path.open(encoding="utf-8") as f:
            raw_docs = json.load(f)
        documents = [
            CodeDocument(
                file_path=str(d["file_path"]),
                content=str(d["content"]),
                chunk_id=str(d.get("chunk_id", "") or ""),
            )
            for d in raw_docs
        ]

        store = cls(embedding=embedding)
        store.index = faiss_index
        store.documents = documents
        return store

    def build_from_directory(
        self,
        target_dir: str,
        include_suffixes: tuple[str, ...] = (".py", ".md", ".txt"),
    ) -> int:
        base = Path(target_dir)
        if not base.exists():
            self.logger.warning("Target directory not found: %s", target_dir)
            return 0

        docs: list[CodeDocument] = []
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if include_suffixes and path.suffix.lower() not in include_suffixes:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            docs.append(CodeDocument(file_path=str(path), content=content))

        return self.add_documents(docs)


def write_index_meta(
    index_dir: Path,
    *,
    codebase_root: str,
    snapshot: str,
    backend_id: str,
    dim: int,
    model_name: str | None = None,
) -> None:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "version": 1,
        "backend_id": backend_id,
        "dim": dim,
        "codebase_root": codebase_root,
        "snapshot": snapshot,
    }
    if model_name is not None:
        meta["model_name"] = model_name
    meta_path = index_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def read_index_meta(index_dir: Path) -> dict[str, Any] | None:
    meta_path = Path(index_dir) / "meta.json"
    if not meta_path.exists():
        return None
    with meta_path.open(encoding="utf-8") as f:
        return json.load(f)


def embedding_model_label(embedding: EmbeddingBackend) -> str | None:
    if hasattr(embedding, "model_name"):
        return str(getattr(embedding, "model_name"))
    if hasattr(embedding, "model"):
        return str(getattr(embedding, "model"))
    return None
