from __future__ import annotations

from app.rag.vector_store import FaissVectorStore
from app.utils.logger import get_logger


class CodeRetriever:
    """Retriever API: input query, return top-k code snippets."""

    def __init__(self, store: FaissVectorStore, logger_name: str = "codeinsight.rag.retriever") -> None:
        self.store = store
        self.logger = get_logger(logger_name)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, str | float]]:
        self.logger.info("Retrieving top-%d snippets for query.", top_k)
        results = self.store.search(query=query, top_k=top_k)
        self.logger.info("Retrieved %d snippet(s).", len(results))
        return results
