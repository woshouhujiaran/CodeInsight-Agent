from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable
from urllib import error, request

import numpy as np

from app.utils.logger import get_logger

logger = get_logger("codeinsight.rag.embeddings")


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Text -> dense vector for FAISS (inner product on L2-normalized vectors)."""

    backend_id: str
    dim: int

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return float32 array of shape (len(texts), dim)."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Return float32 vector of shape (dim,)."""
        ...


class HashEmbedding:
    """Deterministic pseudo-embedding (baseline / no extra deps)."""

    backend_id = "hash"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        rows = [self._embed_one(t) for t in texts]
        return np.vstack(rows).astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return vec

        tokens = text.split()
        if not tokens:
            tokens = [text]

        for token in tokens:
            idx, val = self._token_projection(token, self.dim)
            vec[idx] += val

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    @staticmethod
    @lru_cache(maxsize=50000)
    def _token_projection(token: str, dim: int) -> tuple[int, float]:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        val = (int.from_bytes(digest[4:8], "little") % 1000) / 1000.0
        return idx, val


class SentenceTransformersEmbedding:
    """Local embeddings via sentence-transformers (e.g. BGE / GTE families on HuggingFace)."""

    backend_id = "sentence_transformers"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        self._ensure_model()
        assert self._dim is not None
        return self._dim

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for EMBEDDING_BACKEND=sentence_transformers. "
                "Install with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading SentenceTransformer model=%s", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self._ensure_model()
        assert self._model is not None
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        emb = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(emb, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


# Known OpenAI embedding dimensions (avoid a dummy API call just to learn dim).
_OPENAI_EMBEDDING_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAICompatibleEmbedding:
    """OpenAI-compatible /embeddings HTTP API (OpenAI, Azure OpenAI base URL, etc.)."""

    backend_id = "openai"

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        self.model = model
        self.api_key = api_key.strip()
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._dim = _OPENAI_EMBEDDING_DIMS.get(model)
        if self._dim is None:
            logger.warning(
                "Unknown OpenAI embedding model=%s; defaulting dim=1536. "
                "Set OPENAI_EMBEDDING_DIM if retrieval looks wrong.",
                model,
            )
            self._dim = int(os.getenv("OPENAI_EMBEDDING_DIM", "1536"))

    @property
    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        # OpenAI allows batching; keep batches moderate.
        batch_size = 64
        chunks: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            chunks.append(self._embed_batch(batch))
        return np.vstack(chunks).astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        url = f"{self.base_url}/embeddings"
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        req = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="ignore")
            logger.error("Embeddings HTTP error: %s body=%s", exc.code, err_body)
            raise

        data = body.get("data") or []
        # Preserve API order
        by_idx = {int(item.get("index", i)): item for i, item in enumerate(data)}
        vectors: list[list[float]] = []
        for j in range(len(texts)):
            item = by_idx.get(j) or (data[j] if j < len(data) else {})
            emb = item.get("embedding")
            if not isinstance(emb, list):
                raise RuntimeError(f"Invalid embedding response at index {j}")
            vectors.append(emb)
        arr = np.asarray(vectors, dtype="float32")
        # Normalize for cosine / inner product
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        arr = arr / norms
        return arr


def create_embedding_backend() -> EmbeddingBackend:
    """
    Env:
      EMBEDDING_BACKEND = hash | sentence_transformers | openai  (default: sentence_transformers)
      EMBEDDING_MODEL   = HF model id for sentence_transformers (default: BAAI/bge-small-en-v1.5)
      OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_EMBEDDING_MODEL for openai backend
    """
    backend = os.getenv("EMBEDDING_BACKEND", "sentence_transformers").strip().lower()

    if backend == "hash":
        dim = int(os.getenv("EMBEDDING_DIM", "384"))
        logger.info("Using HashEmbedding dim=%d", dim)
        return HashEmbedding(dim=dim)

    if backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            logger.warning("OPENAI_API_KEY missing; falling back to HashEmbedding.")
            return HashEmbedding(dim=int(os.getenv("EMBEDDING_DIM", "384")))
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
        logger.info("Using OpenAI-compatible embeddings model=%s", model)
        return OpenAICompatibleEmbedding(model=model, api_key=api_key)

    if backend == "sentence_transformers":
        model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip()
        try:
            emb = SentenceTransformersEmbedding(model_name=model)
            _ = emb.dim  # load weights early; fail fast if deps missing
            logger.info("Using SentenceTransformersEmbedding model=%s dim=%d", model, emb.dim)
            return emb
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; falling back to HashEmbedding. "
                "pip install sentence-transformers"
            )
            return HashEmbedding(dim=int(os.getenv("EMBEDDING_DIM", "384")))

    logger.warning("Unknown EMBEDDING_BACKEND=%s; using hash.", backend)
    return HashEmbedding(dim=int(os.getenv("EMBEDDING_DIM", "384")))
