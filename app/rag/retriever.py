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
        terms = self._query_terms(base)
        expanded = [
            base,
            f"{base} 入口 核心 调用 路径",
        ]
        if terms:
            expanded.append(" ".join(terms[:6]))
            expanded.append(" ".join(dict.fromkeys(term.replace("_", " ") for term in terms[:4])))
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
        q_tokens = self._query_terms(query)
        scored: list[tuple[float, dict[str, str | float]]] = []
        for item in hits:
            file_path = str(item.get("file_path", ""))
            file_low = file_path.lower()
            content = str(item.get("content", ""))
            content_low = content.lower()
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

            lexical_overlap = self._lexical_overlap_score(q_tokens, file_low=file_low, content_low=content_low)
            if lexical_overlap > 0:
                boost += min(0.12, lexical_overlap * 0.08)
                reasons.append("lexical_overlap")

            path_hint = self._path_hint_score(q_tokens, file_low=file_low)
            if path_hint > 0:
                boost += path_hint
                reasons.append("path_hint")

            if not reasons:
                reasons.append("embedding_score")

            final_score = score + boost
            enriched = dict(item)
            enriched["why_matched"] = ",".join(reasons)
            enriched["rerank_score"] = final_score
            enriched["lexical_score"] = round(float(lexical_overlap), 4)
            scored.append((final_score, enriched))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored]

    def _query_terms(self, query: str) -> list[str]:
        base = str(query or "").strip().lower()
        if not base:
            return []
        ascii_terms = [t for t in re.split(r"[^a-zA-Z0-9_]+", base) if len(t) >= 2]
        cjk_terms = [t for t in re.findall(r"[\u4e00-\u9fff]{2,}", base) if t]
        out: list[str] = []
        for term in [*ascii_terms, *cjk_terms]:
            if term not in out:
                out.append(term)
        return out

    def _lexical_overlap_score(self, query_terms: list[str], *, file_low: str, content_low: str) -> float:
        if not query_terms:
            return 0.0
        matched = sum(1 for term in query_terms if term in file_low or term in content_low)
        return matched / len(query_terms)

    def _path_hint_score(self, query_terms: list[str], *, file_low: str) -> float:
        if not query_terms:
            return 0.0
        path_terms = [part for part in re.split(r"[\\/._-]+", file_low) if part]
        if not path_terms:
            return 0.0
        matched = sum(1 for term in query_terms if any(term == path_term for path_term in path_terms))
        if matched == 0:
            return 0.0
        return min(0.06, matched * 0.02)
