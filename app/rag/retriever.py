from __future__ import annotations

import re
import time

from app.rag.vector_store import FaissVectorStore
from app.utils.logger import get_logger, log_event


class CodeRetriever:
    """Retriever API: input query, return top-k code snippets."""

    def __init__(self, store: FaissVectorStore, logger_name: str = "codeinsight.rag.retriever") -> None:
        self.store = store
        self.logger = get_logger(logger_name)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, str | float]]:
        started = time.perf_counter()
        self.logger.info("Retrieving top-%d snippets for query.", top_k)
        rewritten_queries = self._rewrite_queries(query)
        per_query_k = max(top_k, min(8, top_k * 2))
        raw_hits: list[dict[str, str | float]] = []
        for q in rewritten_queries:
            raw_hits.extend(self.store.search(query=q, top_k=per_query_k))

        deduped = self._dedupe_hits(raw_hits)
        ranked = self._rerank_hits(query=query, hits=deduped)[:top_k]
        self.logger.info("Retrieved %d snippet(s).", len(ranked))
        log_event(
            self.logger,
            module="rag",
            action="retrieve",
            status="ok",
            duration_ms=int((time.perf_counter() - started) * 1000),
            top_k=top_k,
            expanded_queries=len(rewritten_queries),
            raw_hits=len(raw_hits),
            dedup_hits=len(deduped),
            hits=len(ranked),
        )
        return ranked

    def _rewrite_queries(self, query: str) -> list[str]:
        base = (query or "").strip()
        if not base:
            return [""]
        expanded = [
            base,
            f"{base} 入口 核心 调用 路径",
        ]
        tokens = [t for t in re.split(r"\s+", base) if t]
        if tokens:
            expanded.append(" ".join(tokens[:4]))
        out: list[str] = []
        for item in expanded:
            s = item.strip()
            if s and s not in out:
                out.append(s)
        return out

    def _dedupe_hits(self, hits: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
        dedup: dict[tuple[str, str], dict[str, str | float]] = {}
        for hit in hits:
            file_path = str(hit.get("file_path", ""))
            chunk_id = str(hit.get("chunk_id", ""))
            key = (file_path, chunk_id)
            prev = dedup.get(key)
            if prev is None or float(hit.get("score", 0.0)) > float(prev.get("score", 0.0)):
                dedup[key] = dict(hit)
        return list(dedup.values())

    def _rerank_hits(self, *, query: str, hits: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
        q = query.lower()
        q_tokens = [t for t in re.split(r"[^a-zA-Z0-9_]+", q) if len(t) >= 2]
        scored: list[tuple[float, dict[str, str | float]]] = []
        for item in hits:
            file_path = str(item.get("file_path", ""))
            file_low = file_path.lower()
            content = str(item.get("content", ""))
            score = float(item.get("score", 0.0))
            reasons: list[str] = []
            boost = 0.0

            filename_hit = any(token in file_low for token in q_tokens)
            if filename_hit:
                boost += 0.08
                reasons.append("filename_token_match")

            symbol_hit = any(re.search(rf"\b{re.escape(token)}\b", content, flags=re.IGNORECASE) for token in q_tokens)
            if symbol_hit:
                boost += 0.05
                reasons.append("symbol_match")

            if not reasons:
                reasons.append("embedding_score")

            final_score = score + boost
            enriched = dict(item)
            enriched["why_matched"] = ",".join(reasons)
            enriched["rerank_score"] = final_score
            scored.append((final_score, enriched))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored]
