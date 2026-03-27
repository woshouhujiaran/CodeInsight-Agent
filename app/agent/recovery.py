from __future__ import annotations

import json
from typing import Any


def search_tool_output_is_empty(output: str) -> bool:
    """search_tool returns JSON array of hits; empty list means no retrieval."""
    text = (output or "").strip()
    if not text:
        return True
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(data, list) and len(data) == 0


def should_recovery_replan(tool_results: list[dict[str, Any]]) -> bool:
    """
    Whether to run a single recovery replan pass.

    Triggers:
    - search_tool: status error, or ok but empty hit list
    """
    for r in tool_results:
        if r.get("tool") != "search_tool":
            continue
        if r.get("status") == "error":
            return True
        out = r.get("output", "")
        if isinstance(out, str) and search_tool_output_is_empty(out):
            return True
    return False
