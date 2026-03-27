from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np

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

    Notes:
    - Uses lightweight deterministic embedding (hash-based) by default.
    - You can replace `_embed_text` with real embedding model output later.
    """

    def __init__(self, dim: int = 384, logger_name: str = "codeinsight.rag.faiss") -> None:
        if faiss is None:
            raise ImportError("faiss is not installed. Please install faiss-cpu first.")
        self.dim = dim
        self.logger = get_logger(logger_name)
        self.index = faiss.IndexFlatIP(dim)
        self.documents: list[CodeDocument] = []

    def add_documents(self, documents: Iterable[CodeDocument]) -> int:
        docs = list(documents)
        if not docs:
            return 0

        vectors = np.vstack([self._embed_text(item.content) for item in docs]).astype("float32")
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.documents.extend(docs)
        self.logger.info("Added %d documents to FAISS index.", len(docs))
        return len(docs)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, str | float]]:
        if not query.strip() or self.index.ntotal == 0:
            return []

        q = self._embed_text(query).reshape(1, -1).astype("float32")
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

    def _embed_text(self, text: str) -> np.ndarray:
        # Deterministic pseudo-embedding for bootstrap.
        # Replace with real embeddings for production retrieval quality.
        vec = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return vec

        tokens = text.split()
        if not tokens:
            tokens = [text]

        for token in tokens:
            idx, val = self._token_projection(token, self.dim)
            vec[idx] += val

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    @staticmethod
    @lru_cache(maxsize=50000)
    def _token_projection(token: str, dim: int) -> tuple[int, float]:
        """Cache token hash projection to reduce repeated SHA256 work."""
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        val = (int.from_bytes(digest[4:8], "little") % 1000) / 1000.0
        return idx, val
