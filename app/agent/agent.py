from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent.executor import Executor
from app.agent.memory import ConversationMemory
from app.agent.planner import Planner
from app.agent.recovery import should_recovery_replan
from app.llm.llm import LLMClient
from app.utils.logger import get_logger


@dataclass
class AgentTurnResult:
    """Structured response for one agent turn."""

    answer: str
    plan: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    context: str
    recovery_applied: bool = False


class CodeAgent:
    """
    Main orchestrator for the code agent loop.

    Flow:
    1) accept user query
    2) ask planner for a tool plan
    3) execute tools via executor
    4) build context from tool outputs
    5) ask LLM for final answer
    6) update memory for multi-turn usage
    """

    def __init__(
        self,
        planner: Planner,
        executor: Executor,
        llm: LLMClient,
        memory: ConversationMemory | None = None,
        logger_name: str = "codeinsight.agent",
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.llm = llm
        self.memory = memory or ConversationMemory()
        self.logger = get_logger(logger_name)

    def run(self, user_query: str) -> AgentTurnResult:
        """Run one complete agent turn for a user query."""
        self.logger.info("Agent received query: %s", user_query)

        history = self.memory.get_messages()
        self.logger.debug("Loaded history messages: %d", len(history))

        plan = self.planner.make_plan(user_query=user_query, history=history)
        self.logger.info("Planner generated %d step(s).", len(plan))

        tool_results = self.executor.execute_plan(plan)
        self.logger.info("Executor completed %d tool call(s).", len(tool_results))

        recovery_applied = False
        recovery_results: list[dict[str, Any]] = []
        recovery_plan: list[dict[str, Any]] = []

        if should_recovery_replan(tool_results):
            self.logger.info("Recovery replan triggered (empty search or search error).")
            recovery_plan = self.planner.make_recovery_plan(
                user_query=user_query,
                history=history,
                previous_plan=plan,
                tool_results=tool_results,
            )
            recovery_results = self.executor.execute_plan(recovery_plan)
            recovery_applied = True
            self.logger.info("Recovery executor completed %d tool call(s).", len(recovery_results))

        combined_plan = plan + recovery_plan if recovery_applied else plan
        combined_results = self._merge_tool_results(
            primary=tool_results,
            recovery=recovery_results,
            recovery_applied=recovery_applied,
        )

        context = self._build_context(
            user_query=user_query,
            primary_results=tool_results,
            recovery_results=recovery_results if recovery_applied else None,
        )
        final_answer = self.llm.generate_answer(
            user_query=user_query,
            context=context,
            history=history,
        )

        self.memory.add_user_message(user_query)
        self.memory.add_assistant_message(final_answer)
        self.memory.add_turn_metadata(
            plan=combined_plan,
            tool_results=combined_results,
            recovery_applied=recovery_applied,
        )
        self.logger.info("Memory updated. Total messages: %d", len(self.memory.get_messages()))

        return AgentTurnResult(
            answer=final_answer,
            plan=combined_plan,
            tool_results=combined_results,
            context=context,
            recovery_applied=recovery_applied,
        )

    def _merge_tool_results(
        self,
        *,
        primary: list[dict[str, Any]],
        recovery: list[dict[str, Any]],
        recovery_applied: bool,
    ) -> list[dict[str, Any]]:
        if not recovery_applied:
            return list(primary)
        merged: list[dict[str, Any]] = []
        for r in primary:
            x = dict(r)
            x["replan_round"] = 1
            merged.append(x)
        for r in recovery:
            x = dict(r)
            x["replan_round"] = 2
            merged.append(x)
        return merged

    def _build_context(
        self,
        user_query: str,
        primary_results: list[dict[str, Any]],
        recovery_results: list[dict[str, Any]] | None,
    ) -> str:
        sections: list[str] = [f"User Query:\n{user_query}"]

        def append_block(title: str, tool_results: list[dict[str, Any]]) -> None:
            sections.append(title)
            for idx, result in enumerate(tool_results, start=1):
                tool_name = result.get("tool", "unknown_tool")
                step_id = result.get("step_id", "")
                status = result.get("status", "unknown")
                output = result.get("output", "")
                crit = result.get("success_criteria", "")
                head = f"[{idx}] step_id={step_id} Tool: {tool_name}\nStatus: {status}"
                if crit:
                    head += f"\nSuccess criteria: {crit}"
                sections.append(f"{head}\nOutput:\n{output}")

        append_block("=== Round 1 (initial plan) tool results ===", primary_results)
        if recovery_results:
            append_block("=== Round 2 (recovery replan) tool results ===", recovery_results)

        context = "\n\n".join(sections).strip()
        self.logger.debug("Built context with %d chars.", len(context))
        return context
