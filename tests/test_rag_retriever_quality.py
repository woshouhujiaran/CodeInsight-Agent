from __future__ import annotations

from types import SimpleNamespace

from app.rag.retriever import CodeRetriever


class DummyStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[dict[str, str | float]]:
        self.calls.append((query, top_k))
        return [
            {
                "file_path": "src/auth/login.py",
                "content": "def login(user, password):\n    return True\n",
                "chunk_id": "c1",
                "score": 0.60,
            },
            {
                "file_path": "src/auth/login.py",
                "content": "def login(user, password):\n    return True\n",
                "chunk_id": "c1",
                "score": 0.58,
            },
            {
                "file_path": "src/utils/text.py",
                "content": "def normalize(text):\n    return text.strip()\n",
                "chunk_id": "c2",
                "score": 0.62,
            },
        ]


class LexicalStore:
    def search(self, query: str, top_k: int = 5) -> list[dict[str, str | float]]:
        return [
            {
                "file_path": "app/web/session_store.py",
                "content": "class SessionStore:\n    def save_session(self, snapshot):\n        return snapshot\n",
                "chunk_id": "s1",
                "score": 0.52,
            },
            {
                "file_path": "app/web/service.py",
                "content": "class WebAgentService:\n    pass\n",
                "chunk_id": "s2",
                "score": 0.56,
            },
        ]


class BM25OnlyStore:
    def __init__(self) -> None:
        self.documents = [
            SimpleNamespace(
                file_path="app/auth/session_store.py",
                content="def persist_session(snapshot):\n    return snapshot\n",
                chunk_id="bm25-1",
                symbol_name="persist_session",
                start_line=1,
                end_line=2,
                chunk_kind="function",
                chunk_version="v3",
                content_hash="h1",
            )
        ]

    def search(self, query: str, top_k: int = 5) -> list[dict[str, str | float]]:
        return []


def test_retriever_dedup_and_why_matched() -> None:
    retriever = CodeRetriever(store=DummyStore())  # type: ignore[arg-type]
    hits = retriever.retrieve("login auth", top_k=5)
    keys = {(str(h["file_path"]), str(h["chunk_id"])) for h in hits}
    assert len(hits) == len(keys)
    assert hits
    assert all("why_matched" in h for h in hits)


def test_retriever_rerank_improves_token_relevance() -> None:
    retriever = CodeRetriever(store=DummyStore())  # type: ignore[arg-type]
    hits = retriever.retrieve("login", top_k=2)
    assert len(hits) == 2
    # login.py should rank above text.py after filename/symbol boost
    assert "login.py" in str(hits[0]["file_path"])


def test_retriever_rerank_uses_lexical_overlap_and_path_hints() -> None:
    retriever = CodeRetriever(store=LexicalStore())  # type: ignore[arg-type]
    hits = retriever.retrieve("session store persistence", top_k=2)

    assert hits[0]["file_path"] == "app/web/session_store.py"
    assert "lexical_overlap" in str(hits[0]["why_matched"])
    assert float(hits[0]["lexical_score"]) > 0


def test_retriever_returns_structured_metadata_fields() -> None:
    retriever = CodeRetriever(store=DummyStore())  # type: ignore[arg-type]
    hits = retriever.retrieve("login", top_k=1)
    assert hits
    row = hits[0]
    assert "symbol_name" in row
    assert "start_line" in row
    assert "end_line" in row
    assert "chunk_kind" in row
    assert row.get("chunk_version") == "v3"


def test_retriever_dense_profile_skips_bm25() -> None:
    retriever = CodeRetriever(store=BM25OnlyStore(), retrieval_profile="dense")  # type: ignore[arg-type]
    hits = retriever.retrieve("persist_session", top_k=3)
    assert hits == []


def test_retriever_bm25_profile_returns_lexical_hits() -> None:
    retriever = CodeRetriever(store=BM25OnlyStore(), retrieval_profile="bm25")  # type: ignore[arg-type]
    hits = retriever.retrieve("persist_session", top_k=3)
    assert hits
    assert hits[0]["file_path"] == "app/auth/session_store.py"
    assert "bm25" in str(hits[0]["why_matched"])


def test_retriever_rebuilds_bm25_index_when_doc_content_changes_with_same_count() -> None:
    store = BM25OnlyStore()
    retriever = CodeRetriever(store=store, retrieval_profile="bm25")  # type: ignore[arg-type]
    first_hits = retriever.retrieve("persist_session", top_k=3)
    assert first_hits

    store.documents[0] = SimpleNamespace(
        file_path="app/auth/session_store.py",
        content="def rotate_token(token):\n    return token\n",
        chunk_id="bm25-1",
        symbol_name="rotate_token",
        start_line=1,
        end_line=2,
        chunk_kind="function",
        chunk_version="v3",
        content_hash="h2",
    )
    refreshed_hits = retriever.retrieve("rotate_token", top_k=3)
    assert refreshed_hits
    assert "rotate_token" in str(refreshed_hits[0]["content"])
