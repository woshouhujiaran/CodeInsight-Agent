from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import statistics
import time
from typing import Any
from uuid import uuid4

from app.agent.executor import Executor
from app.agent.memory import ConversationMemory
from app.agent.planner import Planner
from app.agent.recovery import apply_recovery_strategy, evaluate_recovery
from app.llm.llm import LLMClient
from app.utils.logger import get_logger, log_event, set_trace_id


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
        self._recent_turn_metrics: deque[dict[str, float]] = deque(maxlen=20)

    def run(self, user_query: str) -> AgentTurnResult:
        """Run one complete agent turn for a user query."""
        trace_id = uuid4().hex[:12]
        set_trace_id(trace_id)
        turn_started = time.perf_counter()
        self.logger.info("Agent received query: %s", user_query)
        log_event(self.logger, module="agent", action="run_start", status="ok", user_query_len=len(user_query))

        history = self.memory.get_messages()
        self.logger.debug("Loaded history messages: %d", len(history))

        plan = self.planner.make_plan(user_query=user_query, history=history)
        self.logger.info("Planner generated %d step(s).", len(plan))

        tool_results = self.executor.execute_plan(plan)
        self.logger.info("Executor completed %d tool call(s).", len(tool_results))

        recovery_applied = False
        recovery_results: list[dict[str, Any]] = []
        recovery_plan: list[dict[str, Any]] = []
        recovery_reason = "none"
        recovery_strategy = "none"
        recovery_attempted = False

        recovery_decision = evaluate_recovery(tool_results)
        recovery_reason = str(recovery_decision.get("reason", "none"))
        recovery_strategy = str(recovery_decision.get("strategy", "none"))
        if bool(recovery_decision.get("triggered")) and not recovery_attempted:
            recovery_attempted = True
            self.logger.info(
                "Recovery replan triggered reason=%s strategy=%s",
                recovery_reason,
                recovery_strategy,
            )
            raw_recovery_plan = self.planner.make_recovery_plan(
                user_query=user_query,
                history=history,
                previous_plan=plan,
                tool_results=tool_results,
            )
            recovery_plan = apply_recovery_strategy(
                raw_recovery_plan,
                strategy=recovery_strategy,
                user_query=user_query,
            )
            recovery_results = self.executor.execute_plan(recovery_plan)
            recovery_applied = True
            for item in recovery_results:
                item["recovery_trigger_reason"] = recovery_reason
                item["recovery_strategy"] = recovery_strategy
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
            trace_id=trace_id,
        )
        self.logger.info("Memory updated. Total messages: %d", len(self.memory.get_messages()))
        turn_duration_ms = int((time.perf_counter() - turn_started) * 1000)
        self._record_turn_metrics(
            tool_results=combined_results,
            duration_ms=turn_duration_ms,
        )
        summary = self.get_recent_metrics_summary()
        log_event(
            self.logger,
            module="agent",
            action="run_finish",
            status="ok",
            duration_ms=turn_duration_ms,
            recovery_applied=recovery_applied,
            recovery_reason=recovery_reason,
            recovery_strategy=recovery_strategy,
            steps=len(combined_results),
            metrics_window=summary["window"],
            success_rate=summary["success_rate"],
            avg_duration_ms=summary["avg_duration_ms"],
            avg_retries=summary["avg_retries"],
        )

        return AgentTurnResult(
            answer=final_answer,
            plan=combined_plan,
            tool_results=combined_results,
            context=context,
            recovery_applied=recovery_applied,
        )

    def _record_turn_metrics(self, *, tool_results: list[dict[str, Any]], duration_ms: int) -> None:
        if not tool_results:
            success_rate = 1.0
            avg_retries = 0.0
        else:
            ok_count = sum(1 for item in tool_results if item.get("status") == "ok")
            success_rate = ok_count / len(tool_results)
            retries = [max(int(item.get("attempts", 1)) - 1, 0) for item in tool_results]
            avg_retries = statistics.fmean(retries) if retries else 0.0
        self._recent_turn_metrics.append(
            {
                "success_rate": float(success_rate),
                "duration_ms": float(duration_ms),
                "avg_retries": float(avg_retries),
            }
        )

    def get_recent_metrics_summary(self) -> dict[str, float]:
        if not self._recent_turn_metrics:
            return {
                "window": 0,
                "success_rate": 0.0,
                "avg_duration_ms": 0.0,
                "avg_retries": 0.0,
            }
        window = len(self._recent_turn_metrics)
        success_rate = statistics.fmean(item["success_rate"] for item in self._recent_turn_metrics)
        avg_duration = statistics.fmean(item["duration_ms"] for item in self._recent_turn_metrics)
        avg_retries = statistics.fmean(item["avg_retries"] for item in self._recent_turn_metrics)
        return {
            "window": float(window),
            "success_rate": round(float(success_rate), 4),
            "avg_duration_ms": round(float(avg_duration), 2),
            "avg_retries": round(float(avg_retries), 4),
        }

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
