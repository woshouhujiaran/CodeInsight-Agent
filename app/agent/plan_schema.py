from __future__ import annotations

import copy
from typing import Any

try:
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None  # type: ignore[assignment]

READONLY_PLAN_TOOL_ENUM = [
    "search_tool",
    "analyze_tool",
    "optimize_tool",
    "test_tool",
    "read_file_tool",
    "list_dir_tool",
    "grep_tool",
]
WRITE_PLAN_TOOLS = ["apply_patch_tool", "write_file_tool"]
FULL_PLAN_TOOL_ENUM = READONLY_PLAN_TOOL_ENUM + WRITE_PLAN_TOOLS

ALLOWED_TOOLS = frozenset(FULL_PLAN_TOOL_ENUM)

# JSON Schema Draft 2020-12 (subset; validated with jsonschema library).
PLAN_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "CodeInsight Planner Plan",
    "type": "array",
    "minItems": 1,
    "maxItems": 6,
    "items": {
        "type": "object",
        "required": ["id", "deps", "tool", "args", "success_criteria"],
        "additionalProperties": False,
        "properties": {
            "id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 64,
                "pattern": r"^[a-zA-Z0-9_-]+$",
            },
            "deps": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
            "tool": {
                "type": "string",
                "enum": list(READONLY_PLAN_TOOL_ENUM),
            },
            "args": {"type": "object"},
            "success_criteria": {"type": "string", "minLength": 1, "maxLength": 2000},
            "max_retries": {"type": "integer", "minimum": 0, "maximum": 2},
        },
    },
}


def validate_plan_json_schema(plan: list[Any], *, write_tools_enabled: bool = False) -> None:
    """Raise jsonschema.ValidationError if invalid."""
    if jsonschema is None:
        raise RuntimeError("jsonschema is required. pip install jsonschema")
    schema = copy.deepcopy(PLAN_JSON_SCHEMA)
    enum = FULL_PLAN_TOOL_ENUM if write_tools_enabled else READONLY_PLAN_TOOL_ENUM
    schema["items"]["properties"]["tool"]["enum"] = list(enum)
    jsonschema.validate(instance=plan, schema=schema)


def validate_tool_args(tool: str, args: Any) -> str | None:
    """
    Per-tool args checks beyond JSON Schema.
    Returns an error message, or None if OK.
    """
    if not isinstance(args, dict):
        return "args must be an object"

    if tool == "search_tool":
        q = args.get("query")
        inp = args.get("input")
        if isinstance(q, str) and q.strip():
            return None
        if isinstance(inp, str) and inp.strip():
            return None
        return "search_tool.args 需要非空字符串字段 query 或 input"

    if tool in ("analyze_tool", "optimize_tool"):
        inp = args.get("input")
        if isinstance(inp, str) and inp.strip():
            return None
        return f"{tool}.args 需要非空字符串字段 input"

    if tool == "test_tool":
        inp = args.get("input")
        if isinstance(inp, str) and inp.strip():
            return None
        return "test_tool.args 需要非空字符串字段 input（可含 [ORIGINAL_CODE] / [OPTIMIZED_CODE] 段落）"

    if tool == "read_file_tool":
        p = args.get("path")
        if not isinstance(p, str) or not p.strip():
            return "read_file_tool.args 需要非空字符串字段 path"
        sl = args.get("start_line")
        el = args.get("end_line")
        if sl is not None and (not isinstance(sl, int) or sl < 1):
            return "read_file_tool.args 的 start_line 须为 >=1 的整数"
        if el is not None and (not isinstance(el, int) or el < 1):
            return "read_file_tool.args 的 end_line 须为 >=1 的整数"
        if sl is not None and el is not None and el < sl:
            return "read_file_tool.args 的 end_line 不能小于 start_line"
        mc = args.get("max_chars")
        if mc is not None and (not isinstance(mc, int) or mc < 1):
            return "read_file_tool.args 的 max_chars 须为 >=1 的整数"
        return None

    if tool == "list_dir_tool":
        p = args.get("path")
        if not isinstance(p, str) or not p.strip():
            return "list_dir_tool.args 需要非空字符串字段 path"
        depth = args.get("depth")
        if depth is not None and (not isinstance(depth, int) or depth < 1 or depth > 8):
            return "list_dir_tool.args 的 depth 须为 1~8 的整数"
        me = args.get("max_entries")
        if me is not None and (not isinstance(me, int) or me < 1 or me > 5000):
            return "list_dir_tool.args 的 max_entries 须为 1~5000 的整数"
        return None

    if tool == "grep_tool":
        pat = args.get("pattern")
        path = args.get("path")
        if not isinstance(pat, str) or not pat.strip():
            return "grep_tool.args 需要非空字符串字段 pattern"
        if not isinstance(path, str) or not path.strip():
            return "grep_tool.args 需要非空字符串字段 path"
        mm = args.get("max_matches")
        if mm is not None and (not isinstance(mm, int) or mm < 1 or mm > 5000):
            return "grep_tool.args 的 max_matches 须为 1~5000 的整数"
        return None

    if tool == "write_file_tool":
        p = args.get("path")
        if not isinstance(p, str) or not p.strip():
            return "write_file_tool.args 需要非空 path"
        if "content" not in args or not isinstance(args.get("content"), str):
            return "write_file_tool.args 需要字符串 content"
        co = args.get("create_only")
        if co is not None and not isinstance(co, bool):
            return "write_file_tool.args 的 create_only 须为布尔值"
        ech = args.get("expected_content_hash")
        if ech is not None and (not isinstance(ech, str) or not ech.strip()):
            return "write_file_tool.args 的 expected_content_hash 若提供须为非空字符串"
        return None

    if tool == "apply_patch_tool":
        patch = args.get("patch")
        if not isinstance(patch, str) or not patch.strip():
            return "apply_patch_tool.args 需要非空字符串 patch（unified diff）"
        st = args.get("strip")
        if st is not None and (not isinstance(st, int) or st not in (0, 1)):
            return "apply_patch_tool.args 的 strip 仅能为 0 或 1"
        return None

    return f"unknown tool {tool}"


def args_to_tool_input(tool: str, args: dict[str, Any]) -> str:
    """Flatten structured args to the single string expected by BaseTool.run."""
    if tool == "search_tool":
        q = args.get("query")
        if isinstance(q, str) and q.strip():
            return q.strip()
        inp = args.get("input")
        if isinstance(inp, str):
            return inp.strip()
        return ""
    inp = args.get("input")
    if isinstance(inp, str):
        return inp.strip()
    return ""


def validate_step_graph(steps: list[dict[str, Any]]) -> str | None:
    """Unique ids, deps refer to existing ids, DAG (no cycles). Returns error or None."""
    ids = [s.get("id") for s in steps]
    if len(set(ids)) != len(ids):
        return "duplicate step id"

    id_set = set(ids)
    for s in steps:
        sid = s.get("id")
        for d in s.get("deps") or []:
            if d not in id_set:
                return f"step {sid!r} depends on unknown id {d!r}"

    try:
        topological_sort_steps(steps)
    except ValueError as exc:
        return str(exc)

    return None


def topological_sort_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Order steps so every dependency runs before dependents.
    `deps` lists prerequisite step ids.
    """
    step_by_id = {s["id"]: s for s in steps}
    if len(step_by_id) != len(steps):
        raise ValueError("duplicate step id")

    remaining: dict[str, set[str]] = {}
    for s in steps:
        sid = s["id"]
        deps = s.get("deps") or []
        for d in deps:
            if d not in step_by_id:
                raise ValueError(f"step {sid!r} depends on unknown id {d!r}")
        remaining[sid] = set(deps)

    index = {s["id"]: i for i, s in enumerate(steps)}

    def sort_ready(lst: list[dict[str, Any]]) -> None:
        lst.sort(key=lambda x: index[x["id"]])

    ready = [s for s in steps if not remaining[s["id"]]]
    sort_ready(ready)

    result: list[dict[str, Any]] = []
    completed: set[str] = set()

    while ready:
        s = ready.pop(0)
        sid = s["id"]
        if sid in completed:
            continue
        completed.add(sid)
        result.append(s)

        for other in steps:
            oid = other["id"]
            if oid in completed:
                continue
            if sid in remaining[oid]:
                remaining[oid].discard(sid)
                if not remaining[oid] and not any(r["id"] == oid for r in ready):
                    ready.append(other)
        sort_ready(ready)

    if len(result) != len(steps):
        raise ValueError("cycle in plan dependencies")

    return result


def _is_legacy_step(step: dict[str, Any]) -> bool:
    """Legacy steps have tool but no structured id."""
    return "id" not in step and "tool" in step


def coerce_legacy_plan(raw: list[Any], user_query: str) -> list[dict[str, Any]] | None:
    """
    Convert legacy [{tool, input}, ...] to structured steps. Returns None if not legacy.
    """
    if not raw or not isinstance(raw, list):
        return None
    if not all(isinstance(s, dict) for s in raw):
        return None
    if not raw:
        return None
    if not all(_is_legacy_step(s) for s in raw):
        return None

    out: list[dict[str, Any]] = []
    for i, step in enumerate(raw):
        step = step  # dict
        if "tool" not in step:
            return None
        if "input" not in step and "args" not in step:
            return None

        tool = step.get("tool")
        if tool not in ALLOWED_TOOLS:
            return None

        if isinstance(step.get("args"), dict):
            args = dict(step["args"])
        else:
            inp = step.get("input", user_query)
            if not isinstance(inp, str) or not inp.strip():
                inp = user_query
            args = {"input": inp.strip()}

        sid = f"step_{i + 1}"
        deps = [f"step_{i}"] if i > 0 else []
        mr = step.get("max_retries", 1)
        if not isinstance(mr, int):
            mr = 1
        mr = max(0, min(2, mr))

        out.append(
            {
                "id": sid,
                "deps": deps,
                "tool": tool,
                "args": args,
                "success_criteria": step.get("success_criteria") or "完成该工具调用并产出可用输出",
                "max_retries": mr,
            }
        )

    return out
