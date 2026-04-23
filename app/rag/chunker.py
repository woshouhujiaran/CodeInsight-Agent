from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _stable_chunk_id(
    *,
    file_path: str,
    symbol_name: str | None,
    start_line: int | None,
    end_line: int | None,
    content_hash: str,
    chunk_kind: str,
) -> str:
    symbol = (symbol_name or "_").strip() or "_"
    sl = str(start_line or 0)
    el = str(end_line or 0)
    payload = "|".join([file_path, symbol, sl, el, chunk_kind, content_hash])
    return f"{Path(file_path).as_posix()}::v3::{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:20]}"


@dataclass
class CodeChunk:
    file_path: str
    content: str
    chunk_id: str
    symbol_name: str | None
    start_line: int | None
    end_line: int | None
    chunk_kind: str
    content_hash: str
    chunk_version: str = "v3"


@dataclass
class _RawChunk:
    file_path: str
    content: str
    symbol_name: str | None
    start_line: int | None
    end_line: int | None
    chunk_kind: str


class TokenChunker:
    """
    Structured chunker v3.

    - Python files: AST-based symbol chunks (class/function/method).
    - Non-Python files: enhanced semantic text chunks.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50, *, strategy: str = "structured_v3") -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        normalized = str(strategy or "structured_v3").strip().lower()
        if normalized != "structured_v3":
            raise ValueError("TokenChunker only supports strategy=structured_v3")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = normalized

    def split(self, file_path: str, text: str) -> list[CodeChunk]:
        path = str(file_path)
        if Path(path).suffix.lower() == ".py":
            raw = self._split_python_ast(file_path=path, text=text)
            if raw:
                return self._finalize_chunks(raw)
        raw = self._split_enhanced_text(file_path=path, text=text)
        return self._finalize_chunks(raw)

    def _split_python_ast(self, *, file_path: str, text: str) -> list[_RawChunk]:
        try:
            root = ast.parse(text)
        except SyntaxError:
            return []

        lines = text.splitlines()
        chunks: list[_RawChunk] = []

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.class_stack: list[str] = []

            def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
                start = getattr(node, "lineno", None)
                end = getattr(node, "end_lineno", None)
                if isinstance(start, int) and isinstance(end, int) and end >= start:
                    body = "\n".join(lines[start - 1 : end]).strip()
                    if body:
                        chunks.append(
                            _RawChunk(
                                file_path=file_path,
                                content=body,
                                symbol_name=node.name,
                                start_line=start,
                                end_line=end,
                                chunk_kind="class",
                            )
                        )
                self.class_stack.append(node.name)
                self.generic_visit(node)
                self.class_stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
                self._append_function(node=node, async_func=False)
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
                self._append_function(node=node, async_func=True)
                self.generic_visit(node)

            def _append_function(self, *, node: ast.AST, async_func: bool) -> None:
                start = getattr(node, "lineno", None)
                end = getattr(node, "end_lineno", None)
                name = getattr(node, "name", None)
                if not isinstance(start, int) or not isinstance(end, int) or end < start or not isinstance(name, str):
                    return
                body = "\n".join(lines[start - 1 : end]).strip()
                if not body:
                    return
                if self.class_stack:
                    symbol_name = f"{self.class_stack[-1]}.{name}"
                    kind = "method"
                else:
                    symbol_name = name
                    kind = "async_function" if async_func else "function"
                chunks.append(
                    _RawChunk(
                        file_path=file_path,
                        content=body,
                        symbol_name=symbol_name,
                        start_line=start,
                        end_line=end,
                        chunk_kind=kind,
                    )
                )

        Visitor().visit(root)
        return self._dedupe_raw_chunks(chunks)

    def _split_enhanced_text(self, *, file_path: str, text: str) -> list[_RawChunk]:
        blocks = self._semantic_blocks_with_lines(text)
        if not blocks:
            return []

        merged: list[tuple[int, int, str]] = []
        min_tokens = max(12, self.chunk_size // 12)
        max_tokens = self.chunk_size
        cur_start = 0
        cur_end = 0
        cur_parts: list[str] = []
        cur_tokens = 0

        for start, end, body in blocks:
            tok = self._count_tokens(body)
            if tok == 0:
                continue
            if not cur_parts:
                cur_start, cur_end = start, end
                cur_parts = [body]
                cur_tokens = tok
                continue
            if cur_tokens < min_tokens or (cur_tokens + tok) <= max_tokens:
                cur_parts.append(body)
                cur_end = end
                cur_tokens += tok
            else:
                merged.append((cur_start, cur_end, "\n\n".join(cur_parts).strip()))
                cur_start, cur_end = start, end
                cur_parts = [body]
                cur_tokens = tok

        if cur_parts:
            merged.append((cur_start, cur_end, "\n\n".join(cur_parts).strip()))

        out: list[_RawChunk] = []
        for start, end, body in merged:
            for piece_start, piece_end, piece_text in self._split_oversized_by_lines(
                start_line=start,
                end_line=end,
                text=body,
            ):
                if piece_text.strip():
                    out.append(
                        _RawChunk(
                            file_path=file_path,
                            content=piece_text,
                            symbol_name=None,
                            start_line=piece_start,
                            end_line=piece_end,
                            chunk_kind="text",
                        )
                    )
        return self._dedupe_raw_chunks(out)

    def _semantic_blocks_with_lines(self, text: str) -> list[tuple[int, int, str]]:
        lines = text.splitlines()
        if not lines:
            return []
        blocks: list[tuple[int, int, str]] = []
        boundary = re.compile(r"^\s*(class |def |async def |interface |function |export |const |let |var )")

        start = 1
        current: list[str] = []
        for idx, line in enumerate(lines, start=1):
            if boundary.search(line) and current:
                blocks.append((start, idx - 1, "\n".join(current).strip()))
                current = [line]
                start = idx
                continue

            current.append(line)
            if not line.strip() and current:
                blocks.append((start, idx, "\n".join(current).strip()))
                current = []
                start = idx + 1

        if current:
            blocks.append((start, len(lines), "\n".join(current).strip()))
        return [(s, e, b) for s, e, b in blocks if b]

    def _split_oversized_by_lines(self, *, start_line: int, end_line: int, text: str) -> list[tuple[int, int, str]]:
        if self._count_tokens(text) <= self.chunk_size:
            return [(start_line, end_line, text)]

        lines = text.splitlines()
        if not lines:
            return []

        parts: list[tuple[int, int, str]] = []
        i = 0
        while i < len(lines):
            j = i
            tokens = 0
            while j < len(lines):
                next_tokens = self._count_tokens(lines[j])
                if j > i and tokens + next_tokens > self.chunk_size:
                    break
                tokens += max(1, next_tokens)
                j += 1
            if j == i:
                j += 1
            part_text = "\n".join(lines[i:j]).strip()
            part_start = start_line + i
            part_end = start_line + j - 1
            parts.append((part_start, part_end, part_text))
            if j >= len(lines):
                break
            if self.overlap > 0:
                back = 0
                k = j - 1
                while k > i and back < self.overlap:
                    back += max(1, self._count_tokens(lines[k]))
                    k -= 1
                i = max(i + 1, k + 1)
            else:
                i = j
        return parts

    def _dedupe_raw_chunks(self, chunks: list[_RawChunk]) -> list[_RawChunk]:
        out: list[_RawChunk] = []
        seen: set[tuple[str, str, int, int, str]] = set()
        for chunk in chunks:
            content = chunk.content.strip()
            if not content:
                continue
            key = (
                chunk.file_path,
                chunk.symbol_name or "",
                int(chunk.start_line or 0),
                int(chunk.end_line or 0),
                _sha1_text(content),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(
                _RawChunk(
                    file_path=chunk.file_path,
                    content=content,
                    symbol_name=chunk.symbol_name,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    chunk_kind=chunk.chunk_kind,
                )
            )
        return out

    def _finalize_chunks(self, chunks: list[_RawChunk]) -> list[CodeChunk]:
        out: list[CodeChunk] = []
        for raw in chunks:
            content = raw.content.strip()
            if not content:
                continue
            content_hash = _sha1_text(content)
            chunk_id = _stable_chunk_id(
                file_path=raw.file_path,
                symbol_name=raw.symbol_name,
                start_line=raw.start_line,
                end_line=raw.end_line,
                content_hash=content_hash,
                chunk_kind=raw.chunk_kind,
            )
            out.append(
                CodeChunk(
                    file_path=raw.file_path,
                    content=content,
                    chunk_id=chunk_id,
                    symbol_name=raw.symbol_name,
                    start_line=raw.start_line,
                    end_line=raw.end_line,
                    chunk_kind=raw.chunk_kind,
                    content_hash=content_hash,
                    chunk_version="v3",
                )
            )
        return out

    def _count_tokens(self, text: str) -> int:
        return len(str(text or "").split())
