from __future__ import annotations

import json
from typing import Any

from app.agent.plan_schema import (
    coerce_legacy_plan,
    validate_plan_json_schema,
    validate_step_graph,
    validate_tool_args,
)
from app.llm.llm import LLMClient
from app.llm.prompt import (
    PLANNER_SYSTEM_PROMPT,
    RECOVERY_PLANNER_SYSTEM_PROMPT,
    build_planner_user_prompt,
    build_recovery_planner_user_prompt,
)
from app.utils.logger import get_logger


class Planner:
    """LLM-driven planner that outputs structured JSON tool plan (JSON Schema validated)."""

    def __init__(self, llm: LLMClient, logger_name: str = "codeinsight.planner") -> None:
        self.llm = llm
        self.logger = get_logger(logger_name)

    def make_plan(self, user_query: str, history: list[dict[str, str]]) -> list[dict[str, Any]]:
        history_text = self._history_to_text(history)
        user_prompt = build_planner_user_prompt(user_query=user_query, history_text=history_text)
        raw = self.llm.generate_text(prompt=user_prompt, system_prompt=PLANNER_SYSTEM_PROMPT)
        self.logger.debug("Raw planner output: %s", raw)

        return self._parse_and_validate_plan(raw=raw, user_query=user_query, recovery=False)

    def make_recovery_plan(
        self,
        user_query: str,
        history: list[dict[str, str]],
        previous_plan: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Second planning pass after failed/empty retrieval (at most once per agent turn)."""
        history_text = self._history_to_text(history)
        user_prompt = build_recovery_planner_user_prompt(
            user_query=user_query,
            history_text=history_text,
            previous_plan_json=json.dumps(previous_plan, ensure_ascii=False),
            tool_results_summary=self._summarize_tool_results(tool_results),
        )
        raw = self.llm.generate_text(prompt=user_prompt, system_prompt=RECOVERY_PLANNER_SYSTEM_PROMPT)
        self.logger.debug("Recovery planner raw output: %s", raw)

        return self._parse_and_validate_plan(raw=raw, user_query=user_query, recovery=True)

    def _summarize_tool_results(self, tool_results: list[dict[str, Any]], max_chars: int = 4000) -> str:
        slim: list[dict[str, Any]] = []
        for r in tool_results:
            out = r.get("output", "")
            preview = str(out)[:2000] if isinstance(out, str) else str(out)[:2000]
            slim.append(
                {
                    "step_id": r.get("step_id"),
                    "tool": r.get("tool"),
                    "status": r.get("status"),
                    "attempts": r.get("attempts"),
                    "output_preview": preview,
                }
            )
        text = json.dumps(slim, ensure_ascii=False, indent=2)
        if len(text) > max_chars:
            return text[: max_chars - 24] + "\n... [truncated]"
        return text

    def _history_to_text(self, history: list[dict[str, str]]) -> str:
        if not history:
            return "(empty)"
        lines: list[str] = []
        for item in history[-10:]:
            role = item.get("role", "unknown")
            content = item.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _parse_and_validate_plan(
        self, raw: str, user_query: str, *, recovery: bool
    ) -> list[dict[str, Any]]:
        fallback = self._recovery_fallback_plan if recovery else self._fallback_plan

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.warning("Planner returned invalid JSON. Fallback applied.")
            return fallback(user_query)

        if not isinstance(data, list) or not data:
            self.logger.warning("Planner returned empty/non-list plan. Fallback applied.")
            return fallback(user_query)

        plan = self._try_validate_structured_plan(data)
        if plan is None:
            legacy = coerce_legacy_plan(data, user_query)
            if legacy is not None:
                plan = self._try_validate_structured_plan(legacy)

        if plan is None:
            self.logger.warning("Planner plan failed schema/graph validation. Fallback applied.")
            return fallback(user_query)

        return plan[:6]

    def _try_validate_structured_plan(self, data: list[Any]) -> list[dict[str, Any]] | None:
        if not isinstance(data, list) or not data:
            return None
        try:
            validate_plan_json_schema(data)
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("Plan JSON Schema validation failed: %s", exc)
            return None

        steps: list[dict[str, Any]] = []
        for step in data:
            if not isinstance(step, dict):
                return None
            steps.append(step)

        for step in steps:
            tool = step.get("tool")
            args = step.get("args")
            err = validate_tool_args(str(tool), args)
            if err:
                self.logger.warning("Tool args invalid for %s: %s", tool, err)
                return None

        graph_err = validate_step_graph(steps)
        if graph_err:
            self.logger.warning("Plan graph invalid: %s", graph_err)
            return None

        return steps

    def _fallback_plan(self, user_query: str) -> list[dict[str, Any]]:
        return [
            {
                "id": "s1",
                "deps": [],
                "tool": "search_tool",
                "args": {"query": user_query},
                "success_criteria": "检索到与用户问题相关的代码片段（非空、可引用）",
                "max_retries": 1,
            },
            {
                "id": "s2",
                "deps": ["s1"],
                "tool": "analyze_tool",
                "args": {"input": user_query},
                "success_criteria": "基于检索结果给出技术分析、风险与可执行建议",
                "max_retries": 0,
            },
        ]

    def _recovery_fallback_plan(self, user_query: str) -> list[dict[str, Any]]:
        """Deterministic broader search + analyze when recovery LLM output is unusable."""
        broad_query = f"{user_query} 模块 入口 核心 代码结构"
        return [
            {
                "id": "r1",
                "deps": [],
                "tool": "search_tool",
                "args": {"query": broad_query},
                "success_criteria": "检索到非空代码片段或更相关的文件片段",
                "max_retries": 1,
            },
            {
                "id": "r2",
                "deps": ["r1"],
                "tool": "analyze_tool",
                "args": {
                    "input": (
                        f"首轮检索可能为空或不足。请结合用户问题做分析：{user_query}\n"
                        "若仍无代码上下文，请明确列出假设与下一步建议。"
                    )
                },
                "success_criteria": "在有限上下文下仍给出可执行结论或排查路径",
                "max_retries": 0,
            },
        ]
