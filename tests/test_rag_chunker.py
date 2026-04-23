from __future__ import annotations

from app.rag.chunker import TokenChunker


def test_fixed_chunker_overlap() -> None:
    text = " ".join(f"t{i}" for i in range(30))
    chunker = TokenChunker(chunk_size=10, overlap=2, strategy="fixed")
    chunks = chunker.split("x.py", text)
    assert len(chunks) >= 3
    first_tokens = chunks[0].content.split()
    second_tokens = chunks[1].content.split()
    assert first_tokens[-2:] == second_tokens[:2]


def test_semantic_chunker_preserves_boundaries() -> None:
    text = """
class A:
    pass

def f():
    return 1

def g():
    return 2
""".strip()
    chunker = TokenChunker(chunk_size=8, overlap=2, strategy="semantic")
    chunks = chunker.split("m.py", text)
    assert chunks
    joined = "\n".join(c.content for c in chunks)
    assert "class A" in joined
    assert "def f" in joined
    assert "def g" in joined
