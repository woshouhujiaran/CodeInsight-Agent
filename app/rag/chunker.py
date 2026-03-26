from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CodeChunk:
    file_path: str
    content: str
    chunk_id: str


class TokenChunker:
    """
    Split text into fixed-size token chunks.

    Token here is approximated by whitespace split, which is enough for
    initial bootstrap and can be replaced by a real tokenizer later.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, file_path: str, text: str) -> list[CodeChunk]:
        tokens = text.split()
        if not tokens:
            return []

        chunks: list[CodeChunk] = []
        step = self.chunk_size - self.overlap
        start = 0
        idx = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = " ".join(chunk_tokens)
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    content=chunk_text,
                    chunk_id=f"{file_path}::chunk_{idx}",
                )
            )
            idx += 1
            start += step

        return chunks
