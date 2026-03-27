from __future__ import annotations

import json
import re
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
    return evaluate_recovery(tool_results)["triggered"]


def evaluate_recovery(tool_results: list[dict[str, Any]]) -> dict[str, str | bool]:
    """
    Return recovery decision with traceable reason + strategy.

    reason:
      - search_error
      - empty_search_hits
      - analyze_low_information
      - none
    strategy:
      - broad_search
      - split_search
      - analyze_with_assumption
      - none
    """
    for r in tool_results:
        if r.get("tool") != "search_tool":
            continue
        if r.get("status") == "error":
            return {"triggered": True, "reason": "search_error", "strategy": "broad_search"}
        out = r.get("output", "")
        if isinstance(out, str) and search_tool_output_is_empty(out):
            return {"triggered": True, "reason": "empty_search_hits", "strategy": "split_search"}

    for r in tool_results:
        if r.get("tool") != "analyze_tool":
            continue
        if r.get("status") != "ok":
            continue
        out = str(r.get("output", "") or "").strip()
        if _is_low_information_analyze(out):
            return {
                "triggered": True,
                "reason": "analyze_low_information",
                "strategy": "analyze_with_assumption",
            }

    return {"triggered": False, "reason": "none", "strategy": "none"}


def apply_recovery_strategy(
    plan: list[dict[str, Any]],
    *,
    strategy: str,
    user_query: str,
) -> list[dict[str, Any]]:
    if not plan:
        return plan
    patched: list[dict[str, Any]] = [dict(step) for step in plan]

    if strategy == "split_search":
        tokens = [t for t in re.split(r"\s+", user_query.strip()) if t]
        parts = [t for t in tokens[:3] if len(t) >= 1]
        for step in patched:
            if step.get("tool") == "search_tool":
                args = dict(step.get("args") or {})
                base = str(args.get("query") or user_query).strip()
                if len(parts) >= 2:
                    args["query"] = f"{base} {' '.join(parts)}"
                else:
                    args["query"] = f"{base} 核心 入口"
                step["args"] = args
                break
        return patched

    if strategy == "analyze_with_assumption":
        for step in patched:
            if step.get("tool") == "analyze_tool":
                args = dict(step.get("args") or {})
                original = str(args.get("input") or user_query).strip()
                args["input"] = (
                    f"上下文可能不足，请基于合理假设先分析：{original}\n"
                    "请显式列出假设、风险与后续验证步骤。"
                )
                step["args"] = args
                break
        return patched

    return patched


def _is_low_information_analyze(text: str) -> bool:
    if not text:
        return True
    low_info_markers = ("信息不足", "无上下文", "无法判断", "缺少代码", "无法分析")
    if any(token in text for token in low_info_markers):
        return True
    return False
