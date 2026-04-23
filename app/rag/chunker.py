from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class CodeChunk:
    file_path: str
    content: str
    chunk_id: str


class TokenChunker:
    """
    Chunk strategies:
    - `fixed`: fixed-size token window with overlap.
    - `semantic`: group by coarse semantic boundaries (function/class/blank lines),
      with token budget + overlap.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50, *, strategy: str = "fixed") -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = str(strategy or "fixed").strip().lower()

    def split(self, file_path: str, text: str) -> list[CodeChunk]:
        if self.strategy == "semantic":
            return self._split_semantic(file_path=file_path, text=text)
        return self._split_fixed(file_path=file_path, text=text)

    def _split_fixed(self, file_path: str, text: str) -> list[CodeChunk]:
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

    def _split_semantic(self, file_path: str, text: str) -> list[CodeChunk]:
        blocks = self._semantic_blocks(text)
        if not blocks:
            return []

        chunks: list[CodeChunk] = []
        current_parts: list[str] = []
        current_tokens = 0
        idx = 0

        for block in blocks:
            block_tokens = self._count_tokens(block)
            if block_tokens > self.chunk_size:
                # fallback for oversized block: split internally with fixed windows.
                oversized = self._split_fixed(file_path, block)
                for item in oversized:
                    chunks.append(
                        CodeChunk(
                            file_path=file_path,
                            content=item.content,
                            chunk_id=f"{file_path}::chunk_{idx}",
                        )
                    )
                    idx += 1
                continue

            if current_tokens + block_tokens <= self.chunk_size:
                current_parts.append(block)
                current_tokens += block_tokens
                continue

            if current_parts:
                chunk_text = "\n\n".join(current_parts).strip()
                if chunk_text:
                    chunks.append(
                        CodeChunk(
                            file_path=file_path,
                            content=chunk_text,
                            chunk_id=f"{file_path}::chunk_{idx}",
                        )
                    )
                    idx += 1
                current_parts, current_tokens = self._semantic_overlap_tail(current_parts)

            current_parts.append(block)
            current_tokens += block_tokens

        if current_parts:
            chunk_text = "\n\n".join(current_parts).strip()
            if chunk_text:
                chunks.append(
                    CodeChunk(
                        file_path=file_path,
                        content=chunk_text,
                        chunk_id=f"{file_path}::chunk_{idx}",
                    )
                )
        return chunks

    def _semantic_blocks(self, text: str) -> list[str]:
        lines = text.splitlines()
        if not lines:
            return []
        blocks: list[str] = []
        current: list[str] = []
        boundary = re.compile(r"^\s*(def |class |async def |interface |function |export )")
        for line in lines:
            if boundary.search(line) and current:
                blocks.append("\n".join(current).strip())
                current = [line]
                continue
            current.append(line)
            if line.strip() == "":
                blocks.append("\n".join(current).strip())
                current = []
        if current:
            blocks.append("\n".join(current).strip())
        return [b for b in blocks if b]

    def _semantic_overlap_tail(self, blocks: list[str]) -> tuple[list[str], int]:
        if self.overlap <= 0 or not blocks:
            return [], 0
        tail: list[str] = []
        tokens = 0
        for block in reversed(blocks):
            block_tokens = self._count_tokens(block)
            if tokens + block_tokens > self.overlap and tail:
                break
            tail.insert(0, block)
            tokens += block_tokens
            if tokens >= self.overlap:
                break
        return tail, tokens

    def _count_tokens(self, text: str) -> int:
        return len(text.split())
