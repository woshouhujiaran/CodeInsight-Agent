from __future__ import annotations

import math
import re
import time
import hashlib
from dataclasses import dataclass
from typing import Any

from app.rag.vector_store import FaissVectorStore
from app.utils.logger import get_logger, log_event


@dataclass
class RetrievalHit:
    file_path: str
    content: str
    chunk_id: str
    symbol_name: str | None
    start_line: int | None
    end_line: int | None
    chunk_kind: str
    chunk_version: str
    dense_score: float
    lexical_score: float
    rerank_score: float
    why_matched: str
    source_backend: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "content": self.content,
            "chunk_id": self.chunk_id,
            "symbol_name": self.symbol_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_kind": self.chunk_kind,
            "chunk_version": self.chunk_version,
            "dense_score": float(self.dense_score),
            "lexical_score": float(self.lexical_score),
            "rerank_score": float(self.rerank_score),
            "score": float(self.rerank_score),
            "why_matched": self.why_matched,
            "source_backend": self.source_backend,
        }


class _BM25Index:
    def __init__(self) -> None:
        self.docs: list[dict[str, str]] = []
        self.term_freqs: list[dict[str, int]] = []
        self.doc_freq: dict[str, int] = {}
        self.doc_lens: list[int] = []
        self.avg_doc_len = 0.0

    def build(self, docs: list[dict[str, str]]) -> None:
        self.docs = docs
        self.term_freqs = []
        self.doc_freq = {}
        self.doc_lens = []
        for doc in docs:
            tokens = self._tokenize(f"{doc['file_path']} {doc['symbol_name']} {doc['content']}")
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.term_freqs.append(tf)
            self.doc_lens.append(len(tokens))
            for token in tf:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
        self.avg_doc_len = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 0.0

    def search(self, query: str, top_k: int) -> list[tuple[float, dict[str, str]]]:
        q_terms = self._tokenize(query)
        if not q_terms or not self.docs:
            return []
        n_docs = len(self.docs)
        k1 = 1.2
        b = 0.75
        scored: list[tuple[float, dict[str, str]]] = []
        for i, doc in enumerate(self.docs):
            tf = self.term_freqs[i]
            dl = self.doc_lens[i] or 1
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                df = self.doc_freq.get(term, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                f = tf[term]
                denom = f + k1 * (1 - b + b * (dl / max(self.avg_doc_len, 1.0)))
                score += idf * (f * (k1 + 1)) / max(denom, 1e-6)
            if score > 0:
                scored.append((float(score), doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        base = str(text or "").strip().lower()
        if not base:
            return []
        ascii_terms = [t for t in re.split(r"[^a-zA-Z0-9_]+", base) if len(t) >= 2]
        cjk_terms = [t for t in re.findall(r"[\u4e00-\u9fff]{2,}", base) if t]
        return [*ascii_terms, *cjk_terms]


class CodeRetriever:
    """Retriever API: input query, return top-k structured code snippets."""

    def __init__(
        self,
        store: FaissVectorStore,
        logger_name: str = "codeinsight.rag.retriever",
        *,
        dense_weight: float = 0.65,
        lexical_weight: float = 0.35,
        retrieval_profile: str = "hybrid",
    ) -> None:
        self.store = store
        self.logger = get_logger(logger_name)
        self.dense_weight = max(0.0, float(dense_weight))
        self.lexical_weight = max(0.0, float(lexical_weight))
        self.retrieval_profile = self._normalize_retrieval_profile(retrieval_profile)
        self._bm25 = _BM25Index()
        self._bm25_signature = ""

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        started = time.perf_counter()
        self.logger.info("Retrieving top-%d snippets for query.", top_k)
        rewritten_queries = self._rewrite_queries(query)
        per_query_k = max(top_k, min(10, top_k * 2))

        if self.retrieval_profile in {"bm25", "hybrid"}:
            self._ensure_lexical_index()
        raw_hits: list[dict[str, Any]] = []
        for q in rewritten_queries:
            if self.retrieval_profile in {"dense", "hybrid"}:
                raw_hits.extend(self.store.search(query=q, top_k=per_query_k))
            if self.retrieval_profile in {"bm25", "hybrid"}:
                raw_hits.extend(self._lexical_search(query=q, top_k=per_query_k))

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

    def _ensure_lexical_index(self) -> None:
        docs = getattr(self.store, "documents", [])
        signature = self._bm25_docs_signature(docs)
        if self._bm25_signature == signature:
            return
        payload = [
            {
                "file_path": str(doc.file_path),
                "content": str(doc.content),
                "chunk_id": str(doc.chunk_id),
                "symbol_name": str(doc.symbol_name or ""),
                "start_line": str(doc.start_line or ""),
                "end_line": str(doc.end_line or ""),
                "chunk_kind": str(doc.chunk_kind),
                "chunk_version": str(doc.chunk_version),
            }
            for doc in docs
        ]
        self._bm25.build(payload)
        self._bm25_signature = signature

    def _lexical_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        rows = self._bm25.search(query=query, top_k=top_k)
        if not rows:
            return []
        max_score = max(score for score, _ in rows) or 1.0
        hits: list[dict[str, Any]] = []
        for score, doc in rows:
            lex_score = float(score / max_score)
            hits.append(
                {
                    "file_path": doc["file_path"],
                    "content": doc["content"],
                    "chunk_id": doc["chunk_id"],
                    "symbol_name": doc.get("symbol_name") or None,
                    "start_line": int(doc["start_line"]) if str(doc.get("start_line") or "").isdigit() else None,
                    "end_line": int(doc["end_line"]) if str(doc.get("end_line") or "").isdigit() else None,
                    "chunk_kind": doc.get("chunk_kind") or "text",
                    "chunk_version": doc.get("chunk_version") or "v3",
                    "dense_score": 0.0,
                    "lexical_score": lex_score,
                    "rerank_score": lex_score,
                    "score": lex_score,
                    "why_matched": "bm25_score",
                    "source_backend": "bm25",
                }
            )
        return hits

    def _rewrite_queries(self, query: str) -> list[str]:
        base = (query or "").strip()
        if not base:
            return [""]
        terms = self._query_terms(base)
        expanded = [base]
        if terms:
            expanded.append(" ".join(terms[:8]))
            expanded.append(" ".join(dict.fromkeys(term.replace("_", " ") for term in terms[:5])))
        out: list[str] = []
        for item in expanded:
            s = item.strip()
            if s and s not in out:
                out.append(s)
        return out

    def _dedupe_hits(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        dedup: dict[str, dict[str, Any]] = {}
        for hit in hits:
            row = dict(hit)
            if "dense_score" not in row:
                row["dense_score"] = float(row.get("score", 0.0))
            chunk_id = str(row.get("chunk_id", "")).strip()
            if not chunk_id:
                chunk_id = f"{row.get('file_path', '')}::anon"
            prev = dedup.get(chunk_id)
            if prev is None:
                dedup[chunk_id] = row
                continue
            merged = dict(prev)
            merged["dense_score"] = max(float(prev.get("dense_score", 0.0)), float(row.get("dense_score", 0.0)))
            merged["lexical_score"] = max(float(prev.get("lexical_score", 0.0)), float(row.get("lexical_score", 0.0)))
            merged["source_backend"] = "hybrid"
            dedup[chunk_id] = merged
        return list(dedup.values())

    def _rerank_hits(self, *, query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        q_tokens = self._query_terms(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in hits:
            file_path = str(item.get("file_path", ""))
            file_low = file_path.lower()
            content = str(item.get("content", ""))
            content_low = content.lower()
            symbol = str(item.get("symbol_name") or "")
            symbol_low = symbol.lower()
            dense_score = float(item.get("dense_score", 0.0))
            lexical_score = float(item.get("lexical_score", 0.0))
            reasons: list[str] = []
            boost = 0.0

            filename_hit = any(token in file_low for token in q_tokens)
            if filename_hit:
                boost += 0.06
                reasons.append("filename_token_match")

            lexical_overlap = self._lexical_overlap_score(q_tokens, file_low=file_low, content_low=content_low)
            if lexical_overlap > 0:
                boost += min(0.10, lexical_overlap * 0.07)
                reasons.append("lexical_overlap")

            symbol_overlap = self._symbol_overlap_score(q_tokens, symbol_low=symbol_low)
            if symbol_overlap > 0:
                boost += min(0.16, symbol_overlap * 0.12)
                reasons.append("symbol_match")

            path_hint = self._path_hint_score(q_tokens, file_low=file_low)
            if path_hint > 0:
                boost += path_hint
                reasons.append("path_hint")

            if dense_score > 0:
                reasons.append("dense")
            if lexical_score > 0:
                reasons.append("bm25")
            if not reasons:
                reasons.append("embedding_score")

            weighted = self.dense_weight * dense_score + self.lexical_weight * lexical_score
            final_score = weighted + boost
            effective_lexical_score = max(lexical_score, lexical_overlap)
            hit = RetrievalHit(
                file_path=file_path,
                content=content,
                chunk_id=str(item.get("chunk_id", "")),
                symbol_name=(str(item.get("symbol_name")) if item.get("symbol_name") is not None else None),
                start_line=(int(item.get("start_line")) if item.get("start_line") is not None else None),
                end_line=(int(item.get("end_line")) if item.get("end_line") is not None else None),
                chunk_kind=str(item.get("chunk_kind") or "text"),
                chunk_version=str(item.get("chunk_version") or "v3"),
                dense_score=dense_score,
                lexical_score=round(float(effective_lexical_score), 4),
                rerank_score=round(float(final_score), 6),
                why_matched=",".join(dict.fromkeys(reasons)),
                source_backend=str(item.get("source_backend") or "hybrid"),
            )
            scored.append((final_score, hit.to_dict()))

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

    def _symbol_overlap_score(self, query_terms: list[str], *, symbol_low: str) -> float:
        if not query_terms or not symbol_low:
            return 0.0
        matched = sum(1 for term in query_terms if term in symbol_low)
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

    def _normalize_retrieval_profile(self, retrieval_profile: str) -> str:
        mode = str(retrieval_profile or "hybrid").strip().lower()
        if mode not in {"dense", "bm25", "hybrid"}:
            return "hybrid"
        return mode

    def _bm25_docs_signature(self, docs: list[Any]) -> str:
        hasher = hashlib.sha1()
        for doc in docs:
            hasher.update(str(getattr(doc, "chunk_id", "")).encode("utf-8"))
            hasher.update(b"\x1f")
            hasher.update(str(getattr(doc, "file_path", "")).encode("utf-8"))
            hasher.update(b"\x1f")
            content_hash = str(getattr(doc, "content_hash", "")).strip()
            if content_hash:
                hasher.update(content_hash.encode("utf-8"))
            else:
                hasher.update(str(getattr(doc, "content", "")).encode("utf-8"))
            hasher.update(b"\x1e")
        return hasher.hexdigest()
