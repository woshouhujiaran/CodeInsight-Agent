from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import Any

DEFAULT_CONTEXT_MAX_CHARS = 16_000
MIN_CONTEXT_MAX_CHARS = 2_000
MAX_CONTEXT_MAX_CHARS = 120_000
MAX_TOOL_OUTPUT_CHARS = 2_000
MAX_SEARCH_SNIPPET_CHARS = 1_600

_TOOL_PRIORITY: dict[str, int] = {
    "search_tool": 120,
    "code_search": 120,
    "read_file_tool": 100,
    "open_file": 100,
    "find_symbol": 98,
    "grep_tool": 95,
    "list_dir_tool": 80,
    "run_code": 58,
    "run_tests": 56,
    "analyze_tool": 70,
    "optimize_tool": 60,
    "test_tool": 55,
}


@dataclass
class _ContextPiece:
    text: str
    priority: float
    dedupe_key: str
    order: int


def build_context(
    *,
    user_query: str,
    primary_results: list[dict[str, Any]],
    recovery_results: list[dict[str, Any]] | None,
    max_chars: int | None = None,
) -> str:
    budget = _resolve_budget(max_chars)
    blocks: list[str] = [f"User Query:\n{user_query}"]
    used_chars = len(blocks[0])
    seen_keys: set[str] = set()

    used_chars = _append_round(
        blocks=blocks,
        used_chars=used_chars,
        budget=budget,
        title="=== Round 1 (initial plan) tool results ===",
        tool_results=primary_results,
        seen_keys=seen_keys,
    )

    if recovery_results:
        used_chars = _append_round(
            blocks=blocks,
            used_chars=used_chars,
            budget=budget,
            title="=== Round 2 (recovery replan) tool results ===",
            tool_results=recovery_results,
            seen_keys=seen_keys,
        )

    text = "\n\n".join(blocks).strip()
    if len(text) <= budget:
        return text
    tail = "\n\n[context_builder truncated by budget]"
    return text[: max(0, budget - len(tail))] + tail


def _append_round(
    *,
    blocks: list[str],
    used_chars: int,
    budget: int,
    title: str,
    tool_results: list[dict[str, Any]],
    seen_keys: set[str],
) -> int:
    pieces = _collect_pieces(tool_results)
    if not pieces:
        return used_chars

    title_delta = len(title) + 2
    if used_chars + title_delta > budget:
        return used_chars
    blocks.append(title)
    used_chars += title_delta

    omitted = 0
    for piece in pieces:
        if piece.dedupe_key in seen_keys:
            omitted += 1
            continue
        delta = len(piece.text) + 2
        if used_chars + delta > budget:
            omitted += 1
            continue
        blocks.append(piece.text)
        used_chars += delta
        seen_keys.add(piece.dedupe_key)

    if omitted > 0:
        note = f"[context_builder] omitted {omitted} item(s) due to dedupe/budget."
        delta = len(note) + 2
        if used_chars + delta <= budget:
            blocks.append(note)
            used_chars += delta
    return used_chars


def _collect_pieces(tool_results: list[dict[str, Any]]) -> list[_ContextPiece]:
    depths = _compute_dependency_depth(tool_results)
    pieces: list[_ContextPiece] = []
    next_order = 0
    for idx, row in enumerate(tool_results, start=1):
        tool = str(row.get("tool", "unknown_tool"))
        step_id = str(row.get("step_id", ""))
        status = str(row.get("status", "unknown"))
        deps = row.get("deps") if isinstance(row.get("deps"), list) else []
        dep_text = ",".join(str(item) for item in deps) if deps else "-"
        depth = depths.get(step_id, 0)
        base_priority = float(_TOOL_PRIORITY.get(tool, 50))
        if status == "ok":
            base_priority += 20.0
        base_priority += float(depth) * 3.0

        if tool in {"search_tool", "code_search"}:
            parsed = _parse_search_hits(row.get("output"))
            if parsed:
                for hit_idx, hit in enumerate(parsed, start=1):
                    file_path = str(hit.get("file_path") or "")
                    content = str(hit.get("content") or "")
                    content = _clip_text(content, MAX_SEARCH_SNIPPET_CHARS)
                    start = hit.get("start_line")
                    end = hit.get("end_line")
                    score = _safe_float(hit.get("rerank_score"), fallback=_safe_float(hit.get("score"), fallback=0.0))
                    chunk_id = str(hit.get("chunk_id") or "").strip()
                    range_text = _format_range(start, end)
                    head = (
                        f"[{idx}.{hit_idx}] step_id={step_id} Tool: {tool}\n"
                        f"Status: {status} deps={dep_text} dep_depth={depth}\n"
                        f"file={file_path}{range_text} score={score:.4f}"
                    )
                    text = f"{head}\nSnippet:\n{content}"
                    key = chunk_id or _hash_text(f"{file_path}|{start}|{end}|{content}")
                    pieces.append(
                        _ContextPiece(
                            text=text,
                            priority=base_priority + score,
                            dedupe_key=f"search:{key}",
                            order=next_order,
                        )
                    )
                    next_order += 1
                continue

        output = _clip_text(str(row.get("output", "")), MAX_TOOL_OUTPUT_CHARS)
        head = (
            f"[{idx}] step_id={step_id} Tool: {tool}\n"
            f"Status: {status} deps={dep_text} dep_depth={depth}"
        )
        success_criteria = str(row.get("success_criteria", "") or "").strip()
        if success_criteria:
            head += f"\nSuccess criteria: {success_criteria}"
        body = f"{head}\nOutput:\n{output}"
        dedupe_key = f"{tool}:{_hash_text(output.strip())}"
        pieces.append(
            _ContextPiece(
                text=body,
                priority=base_priority,
                dedupe_key=dedupe_key,
                order=next_order,
            )
        )
        next_order += 1

    pieces.sort(key=lambda item: (-item.priority, item.order))
    return pieces


def _compute_dependency_depth(tool_results: list[dict[str, Any]]) -> dict[str, int]:
    deps_by_step: dict[str, list[str]] = {}
    for row in tool_results:
        step_id = str(row.get("step_id", "")).strip()
        if not step_id:
            continue
        deps = row.get("deps")
        if not isinstance(deps, list):
            deps = []
        deps_by_step[step_id] = [str(item).strip() for item in deps if str(item).strip()]

    depth_cache: dict[str, int] = {}

    def dfs(step_id: str, visiting: set[str]) -> int:
        if step_id in depth_cache:
            return depth_cache[step_id]
        if step_id in visiting:
            return 0
        visiting.add(step_id)
        depth = 0
        for dep in deps_by_step.get(step_id, []):
            depth = max(depth, 1 + dfs(dep, visiting))
        visiting.remove(step_id)
        depth_cache[step_id] = depth
        return depth

    for key in deps_by_step:
        dfs(key, set())
    return depth_cache


def _parse_search_hits(output: Any) -> list[dict[str, Any]]:
    if isinstance(output, list):
        rows = output
    elif isinstance(output, str):
        raw = output.strip()
        if not raw:
            return []
        try:
            rows = json.loads(raw)
        except json.JSONDecodeError:
            return []
    else:
        return []
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            out.append(row)
    return out


def _resolve_budget(value: int | None) -> int:
    if isinstance(value, int) and value > 0:
        return max(MIN_CONTEXT_MAX_CHARS, min(value, MAX_CONTEXT_MAX_CHARS))
    env_value = os.getenv("CONTEXT_MAX_CHARS", "").strip()
    if env_value.isdigit():
        parsed = int(env_value)
        if parsed > 0:
            return max(MIN_CONTEXT_MAX_CHARS, min(parsed, MAX_CONTEXT_MAX_CHARS))
    return DEFAULT_CONTEXT_MAX_CHARS


def _format_range(start_line: Any, end_line: Any) -> str:
    if isinstance(start_line, int) and isinstance(end_line, int):
        return f":{start_line}-{end_line}"
    if isinstance(start_line, int):
        return f":{start_line}"
    return ""


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _clip_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 14)] + "\n...[truncated]"


def _safe_float(value: Any, *, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)
