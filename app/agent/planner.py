from __future__ import annotations

import json
import re
from typing import Any

from app.agent.plan_schema import (
    coerce_legacy_plan,
    validate_plan_json_schema,
    validate_step_graph,
    validate_tool_args,
)
from app.agent.task_board import TaskBoard
from app.llm.llm import LLMClient
from app.llm.prompt import (
    build_planner_system_prompt,
    build_planner_user_prompt,
    build_recovery_planner_system_prompt,
    build_recovery_planner_user_prompt,
    build_task_board_system_prompt,
    build_task_board_user_prompt,
)
from app.utils.logger import get_logger


class Planner:
    """LLM-driven planner that outputs structured JSON tool plan (JSON Schema validated)."""

    def __init__(
        self,
        llm: LLMClient,
        logger_name: str = "codeinsight.planner",
        *,
        write_tools_enabled: bool = False,
    ) -> None:
        self.llm = llm
        self.logger = get_logger(logger_name)
        self.last_plan_score: dict[str, Any] | None = None
        self._write_tools_enabled = write_tools_enabled

    def make_plan(self, user_query: str, history: list[dict[str, str]]) -> list[dict[str, Any]]:
        history_text = self._history_to_text(history)
        user_prompt = build_planner_user_prompt(user_query=user_query, history_text=history_text)
        raw = self.llm.generate_text(
            prompt=user_prompt,
            system_prompt=build_planner_system_prompt(self._write_tools_enabled),
        )
        self.logger.debug("Raw planner output: %s", raw)

        return self._parse_and_validate_plan(raw=raw, user_query=user_query, recovery=False)

    def make_task_board(self, user_query: str, history: list[dict[str, str]]) -> list[dict[str, Any]]:
        history_text = self._history_to_text(history)
        user_prompt = build_task_board_user_prompt(user_query=user_query, history_text=history_text)
        raw = self.llm.generate_text(
            prompt=user_prompt,
            system_prompt=build_task_board_system_prompt(),
        )
        self.logger.debug("Raw task board output: %s", raw)
        return self._parse_and_validate_task_board(raw=raw)

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
        raw = self.llm.generate_text(
            prompt=user_prompt,
            system_prompt=build_recovery_planner_system_prompt(self._write_tools_enabled),
        )
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
        plan = plan[:6]

        score = self._score_plan_semantics(plan=plan, user_query=user_query)
        self.last_plan_score = score
        self.logger.info(
            "Plan score overall=%.2f completeness=%.2f dependency=%.2f tool_relevance=%.2f",
            score["overall"],
            score["completeness"],
            score["dependency"],
            score["tool_relevance"],
        )
        if score["fallback_reason"]:
            self.logger.warning("Plan semantic fallback applied: %s", score["fallback_reason"])
            if score["fallback_reason"] == "missing_test_tool_for_test_intent" and not recovery:
                plan = self._test_intent_fallback_plan(user_query)
            else:
                plan = fallback(user_query)
            fallback_score = self._score_plan_semantics(plan=plan, user_query=user_query)
            self.last_plan_score = fallback_score
        return plan

    def _parse_and_validate_task_board(self, raw: str) -> list[dict[str, Any]]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.warning("Task board returned invalid JSON. Fallback applied.")
            return self._fallback_task_board()

        if not isinstance(data, list) or not data:
            self.logger.warning("Task board returned empty/non-list payload. Fallback applied.")
            return self._fallback_task_board()

        try:
            board = TaskBoard.from_dicts(data)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Task board validation failed: %s", exc)
            return self._fallback_task_board()
        return board.to_dicts()

    def _try_validate_structured_plan(self, data: list[Any]) -> list[dict[str, Any]] | None:
        if not isinstance(data, list) or not data:
            return None
        try:
            validate_plan_json_schema(data, write_tools_enabled=self._write_tools_enabled)
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

    def _test_intent_fallback_plan(self, user_query: str) -> list[dict[str, Any]]:
        return [
            {
                "id": "t1",
                "deps": [],
                "tool": "search_tool",
                "args": {"query": user_query},
                "success_criteria": "检索到与测试目标相关代码片段",
                "max_retries": 1,
            },
            {
                "id": "t2",
                "deps": ["t1"],
                "tool": "test_tool",
                "args": {"input": user_query},
                "success_criteria": "生成可执行 pytest 用例并给出覆盖重点",
                "max_retries": 0,
            },
        ]

    def _fallback_task_board(self) -> list[dict[str, Any]]:
        return TaskBoard.from_dicts(
            [
                {
                    "id": "t1",
                    "title": "定位相关代码",
                    "description": "检索与用户目标最相关的模块、入口和文件。",
                    "depends_on": [],
                    "status": "pending",
                    "acceptance": "能指出需要阅读或修改的关键文件。",
                },
                {
                    "id": "t2",
                    "title": "分析实现方案",
                    "description": "确认改动边界、依赖关系和潜在风险。",
                    "depends_on": ["t1"],
                    "status": "pending",
                    "acceptance": "形成可执行的实现路径或手动修改方案。",
                },
                {
                    "id": "t3",
                    "title": "执行修改或生成补丁",
                    "description": "在权限允许时修改代码，否则输出结构化 diff 建议。",
                    "depends_on": ["t2"],
                    "status": "pending",
                    "acceptance": "代码已修改，或给出可复制补丁与编辑步骤。",
                },
                {
                    "id": "t4",
                    "title": "验证与总结",
                    "description": "检查结果并总结验证状态、风险和下一步。",
                    "depends_on": ["t3"],
                    "status": "pending",
                    "acceptance": "产出清晰的验证结论与剩余事项。",
                },
            ]
        ).to_dicts()

    def _score_plan_semantics(self, *, plan: list[dict[str, Any]], user_query: str) -> dict[str, Any]:
        tools = [str(step.get("tool", "")) for step in plan]
        has_search = "search_tool" in tools
        has_fs = any(t in tools for t in ("read_file_tool", "list_dir_tool", "grep_tool"))
        has_context = has_search or has_fs
        has_analyze = "analyze_tool" in tools
        has_test = "test_tool" in tools

        completeness = 1.0 if (has_context and has_analyze) else 0.6 if (has_context or has_analyze) else 0.3
        dependency = 1.0
        if len(plan) >= 2:
            for i, step in enumerate(plan[1:], start=1):
                deps = step.get("deps") if isinstance(step.get("deps"), list) else []
                if not deps:
                    dependency = 0.7
                    break
                prev_id = str(plan[i - 1].get("id", ""))
                if prev_id and prev_id not in deps:
                    dependency = min(dependency, 0.8)

        tool_relevance = 1.0
        asks_test = self._query_requires_test(user_query)
        if asks_test and not has_test:
            tool_relevance = 0.35
        elif asks_test and has_test:
            tool_relevance = 1.0
        elif not asks_test and has_test and len(plan) == 1:
            tool_relevance = 0.75

        overall = round(0.35 * completeness + 0.25 * dependency + 0.40 * tool_relevance, 4)
        fallback_reason = ""
        if overall < 0.65:
            fallback_reason = "low_plan_score"
        if asks_test and not has_test:
            fallback_reason = "missing_test_tool_for_test_intent"
        return {
            "overall": overall,
            "completeness": round(completeness, 4),
            "dependency": round(dependency, 4),
            "tool_relevance": round(tool_relevance, 4),
            "fallback_reason": fallback_reason,
        }

    def _query_requires_test(self, user_query: str) -> bool:
        q = user_query.lower()
        keywords = ("测试", "test", "pytest", "单元测试", "回归测试")
        if any(k in q for k in keywords):
            return True
        tokens = re.split(r"\s+", q)
        return "ut" in tokens
