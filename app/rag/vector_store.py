from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

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
    chunk_id: str


@runtime_checkable
class VectorStore(Protocol):
    """Unified vector store interface for local/remote backends."""

    def upsert_documents(self, documents: Iterable[CodeDocument]) -> int: ...

    def search(self, query: str, top_k: int = 5) -> list[dict[str, str | float]]: ...

    def save(self, index_dir: Path) -> None: ...

    def metadata(self) -> dict[str, Any]: ...


class FaissVectorStore:
    """
    FAISS vector store for code snippets.

    Index strategy:
    - `auto`: choose IndexFlatIP for small corpora, IVF+PQ for larger corpora.
    - `flat`: always IndexFlatIP
    - `ivfpq`: always IVF+PQ (with Flat fallback when faiss cannot train)
    """

    def __init__(
        self,
        embedding: EmbeddingBackend,
        logger_name: str = "codeinsight.rag.faiss",
        *,
        index_strategy: str = "auto",
        nlist: int = 128,
        m_pq: int = 16,
        nprobe: int = 16,
        ivf_min_points: int = 2000,
    ) -> None:
        if faiss is None:
            raise ImportError("faiss is not installed. Please install faiss-cpu first.")
        self.embedding = embedding
        self.dim = embedding.dim
        self.logger = get_logger(logger_name)
        self.index_strategy = str(index_strategy or "auto").strip().lower()
        self.nlist = max(1, int(nlist))
        self.m_pq = max(1, int(m_pq))
        self.nprobe = max(1, int(nprobe))
        self.ivf_min_points = max(256, int(ivf_min_points))

        self.index = faiss.IndexFlatIP(self.dim)
        self.documents: list[CodeDocument] = []
        self._doc_by_chunk_id: dict[str, CodeDocument] = {}
        self._index_type = "flat"

    def add_documents(self, documents: Iterable[CodeDocument]) -> int:
        """Backward-compatible append-only API."""
        docs = list(documents)
        if not docs:
            return 0
        self.documents.extend(docs)
        self._doc_by_chunk_id = {d.chunk_id: d for d in self.documents}
        self._rebuild_index()
        return len(docs)

    def upsert_documents(self, documents: Iterable[CodeDocument]) -> int:
        docs = list(documents)
        if not docs:
            return 0
        for doc in docs:
            chunk_id = str(doc.chunk_id or "").strip()
            if not chunk_id:
                # keep deterministic key when caller did not provide one
                chunk_id = f"{doc.file_path}::auto_{abs(hash(doc.content))}"
            self._doc_by_chunk_id[chunk_id] = CodeDocument(
                file_path=str(doc.file_path),
                content=str(doc.content),
                chunk_id=chunk_id,
            )
        self.documents = list(self._doc_by_chunk_id.values())
        self._rebuild_index()
        self.logger.info("Upserted %d documents. total=%d", len(docs), len(self.documents))
        return len(docs)

    def delete_by_file_paths(self, file_paths: Iterable[str]) -> int:
        targets = {str(p) for p in file_paths if str(p).strip()}
        if not targets:
            return 0
        before = len(self._doc_by_chunk_id)
        self._doc_by_chunk_id = {
            cid: doc for cid, doc in self._doc_by_chunk_id.items() if doc.file_path not in targets
        }
        self.documents = list(self._doc_by_chunk_id.values())
        if len(self.documents) == 0:
            self.index = faiss.IndexFlatIP(self.dim)
            self._index_type = "flat"
        else:
            self._rebuild_index()
        removed = before - len(self._doc_by_chunk_id)
        self.logger.info("Removed %d documents for %d files.", removed, len(targets))
        return removed

    def search(self, query: str, top_k: int = 5) -> list[dict[str, str | float]]:
        if not query.strip() or self.index.ntotal == 0:
            return []

        q = self.embedding.embed_query(query).reshape(1, -1).astype("float32")
        faiss.normalize_L2(q)
        self._apply_search_params()
        scores, indices = self.index.search(q, min(top_k, self.index.ntotal))

        results: list[dict[str, str | float]] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            dense_score = float(score)
            results.append(
                {
                    "file_path": doc.file_path,
                    "content": doc.content,
                    "chunk_id": doc.chunk_id,
                    "dense_score": dense_score,
                    "lexical_score": 0.0,
                    "rerank_score": dense_score,
                    "score": dense_score,
                    "why_matched": "embedding_score",
                    "source_backend": "faiss",
                }
            )
        return results

    def save(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "index.faiss"))
        docs_path = index_dir / "documents.json"
        with docs_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(d) for d in self.documents], f, ensure_ascii=False)

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "faiss",
            "dim": self.dim,
            "index_type": self._index_type,
            "index_strategy": self.index_strategy,
            "nlist": self.nlist,
            "m_pq": self.m_pq,
            "nprobe": self.nprobe,
            "ivf_min_points": self.ivf_min_points,
            "documents": len(self.documents),
        }

    @classmethod
    def load(
        cls,
        index_dir: Path,
        embedding: EmbeddingBackend,
        *,
        index_strategy: str = "auto",
        nlist: int = 128,
        m_pq: int = 16,
        nprobe: int = 16,
        ivf_min_points: int = 2000,
    ) -> FaissVectorStore:
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

        store = cls(
            embedding=embedding,
            index_strategy=index_strategy,
            nlist=nlist,
            m_pq=m_pq,
            nprobe=nprobe,
            ivf_min_points=ivf_min_points,
        )
        store.index = faiss_index
        store.documents = documents
        store._doc_by_chunk_id = {d.chunk_id: d for d in documents}
        if isinstance(faiss_index, faiss.IndexIVFPQ):
            store._index_type = "ivfpq"
        else:
            store._index_type = "flat"
        store._apply_search_params()
        return store

    def _rebuild_index(self) -> None:
        if not self.documents:
            self.index = faiss.IndexFlatIP(self.dim)
            self._index_type = "flat"
            return

        texts = [d.content for d in self.documents]
        vectors = self.embedding.embed_texts(texts).astype("float32")
        faiss.normalize_L2(vectors)
        self.index = self._create_index(vectors)
        self.index.add(vectors)
        self._apply_search_params()

    def _create_index(self, vectors: np.ndarray) -> Any:
        n_docs = int(vectors.shape[0])
        want_ivf = self.index_strategy == "ivfpq" or (
            self.index_strategy == "auto" and n_docs >= self.ivf_min_points
        )

        if not want_ivf:
            self._index_type = "flat"
            return faiss.IndexFlatIP(self.dim)

        try:
            # Use an IP quantizer over normalized vectors (cosine similarity via IP).
            quantizer = faiss.IndexFlatIP(self.dim)
            effective_nlist = min(self.nlist, max(1, n_docs // 8))
            index = faiss.IndexIVFPQ(
                quantizer,
                self.dim,
                effective_nlist,
                self.m_pq,
                8,
                faiss.METRIC_INNER_PRODUCT,
            )
            if not index.is_trained:
                index.train(vectors)
            self._index_type = "ivfpq"
            return index
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Falling back to flat index due to IVF/PQ build failure: %s", exc)
            self._index_type = "flat"
            return faiss.IndexFlatIP(self.dim)

    def _apply_search_params(self) -> None:
        if hasattr(self.index, "nprobe"):
            try:
                self.index.nprobe = self.nprobe
            except Exception:  # noqa: BLE001
                pass


def write_index_meta(
    index_dir: Path,
    *,
    codebase_root: str,
    snapshot: str,
    backend_id: str,
    dim: int,
    model_name: str | None = None,
    store_metadata: dict[str, Any] | None = None,
) -> None:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "version": 2,
        "backend_id": backend_id,
        "dim": dim,
        "codebase_root": codebase_root,
        "snapshot": snapshot,
    }
    if model_name is not None:
        meta["model_name"] = model_name
    if store_metadata:
        meta["store"] = dict(store_metadata)
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
