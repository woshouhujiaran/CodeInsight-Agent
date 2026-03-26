from __future__ import annotations

import json
from typing import Any

from app.llm.llm import LLMClient
from app.llm.prompt import PLANNER_SYSTEM_PROMPT, build_planner_user_prompt
from app.utils.logger import get_logger


ALLOWED_TOOLS = {"search_tool", "analyze_tool", "optimize_tool", "test_tool"}


class Planner:
    """LLM-driven planner that outputs structured JSON tool plan."""

    def __init__(self, llm: LLMClient, logger_name: str = "codeinsight.planner") -> None:
        self.llm = llm
        self.logger = get_logger(logger_name)

    def make_plan(self, user_query: str, history: list[dict[str, str]]) -> list[dict[str, Any]]:
        history_text = self._history_to_text(history)
        user_prompt = build_planner_user_prompt(user_query=user_query, history_text=history_text)
        raw = self.llm.generate_text(prompt=user_prompt, system_prompt=PLANNER_SYSTEM_PROMPT)
        self.logger.debug("Raw planner output: %s", raw)

        return self._parse_and_validate_plan(raw=raw, user_query=user_query)

    def _history_to_text(self, history: list[dict[str, str]]) -> str:
        if not history:
            return "(empty)"
        lines: list[str] = []
        for item in history[-10:]:
            role = item.get("role", "unknown")
            content = item.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _parse_and_validate_plan(self, raw: str, user_query: str) -> list[dict[str, Any]]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.warning("Planner returned invalid JSON. Fallback applied.")
            return self._fallback_plan(user_query)

        if not isinstance(data, list) or not data:
            self.logger.warning("Planner returned empty/non-list plan. Fallback applied.")
            return self._fallback_plan(user_query)

        normalized: list[dict[str, Any]] = []
        for step in data[:6]:
            if not isinstance(step, dict):
                continue
            tool = step.get("tool")
            tool_input = step.get("input", "")

            if tool not in ALLOWED_TOOLS:
                continue
            if not isinstance(tool_input, str) or not tool_input.strip():
                tool_input = user_query

            normalized.append({"tool": tool, "input": tool_input.strip()})

        if not normalized:
            self.logger.warning("Planner plan filtered to empty. Fallback applied.")
            return self._fallback_plan(user_query)

        return normalized

    def _fallback_plan(self, user_query: str) -> list[dict[str, Any]]:
        return [
            {"tool": "search_tool", "input": user_query},
            {"tool": "analyze_tool", "input": user_query},
        ]
