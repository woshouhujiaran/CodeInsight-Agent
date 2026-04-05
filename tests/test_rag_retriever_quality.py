from __future__ import annotations

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
