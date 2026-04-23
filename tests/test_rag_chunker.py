from __future__ import annotations

from app.rag.chunker import TokenChunker


def test_python_ast_chunker_extracts_symbols_and_lines() -> None:
    text = """
class A:
    def m(self, x):
        return x + 1


def f(a, b):
    return a + b
""".strip()
    chunker = TokenChunker(chunk_size=120, overlap=20, strategy="structured_v3")
    chunks = chunker.split("m.py", text)

    assert chunks
    assert all(c.chunk_version == "v3" for c in chunks)
    assert all(c.file_path == "m.py" for c in chunks)
    assert all(c.start_line is not None and c.end_line is not None for c in chunks)
    symbols = {c.symbol_name for c in chunks}
    assert "A" in symbols
    assert "A.m" in symbols
    assert "f" in symbols


def test_non_python_chunker_generates_structured_metadata() -> None:
    text = """
export function loadUsers() {
  return []
}

export function saveUsers(users) {
  return users.length
}
""".strip()
    chunker = TokenChunker(chunk_size=30, overlap=5, strategy="structured_v3")
    chunks = chunker.split("x.ts", text)

    assert chunks
    assert all(c.chunk_kind == "text" for c in chunks)
    assert all(c.symbol_name is None for c in chunks)
    assert all(c.start_line is not None and c.end_line is not None for c in chunks)
    assert all(c.content_hash for c in chunks)


def test_chunker_rejects_legacy_strategies() -> None:
    try:
        TokenChunker(chunk_size=20, overlap=5, strategy="fixed")
    except ValueError as exc:
        assert "structured_v3" in str(exc)
    else:
        raise AssertionError("expected ValueError")
