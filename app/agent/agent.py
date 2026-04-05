from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import json
import statistics
import time
from typing import Any
from uuid import uuid4

from app.agent.executor import Executor
from app.agent.memory import ConversationMemory
from app.agent.planner import Planner
from app.agent.recovery import apply_recovery_strategy, evaluate_recovery
from app.agent.tool_specs import compact_tool_specs_for_prompt
from app.llm.llm import AGENTIC_JSON_SYSTEM_SUFFIX, AGENTIC_TOOL_USE_POLICY, LLMClient
from app.utils.logger import get_logger, log_event, set_trace_id


@dataclass
class AgentTurnResult:
    """Structured response for one agent turn."""

    answer: str
    plan: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    context: str
    recovery_applied: bool = False


@dataclass
class AgenticTurnResult:
    """Structured response for one agentic multi-tool turn."""

    answer: str
    messages: list[dict[str, str]]
    tool_trace: list[dict[str, Any]]
    trace_id: str = ""


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
        *,
        workspace_root: str | None = None,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.llm = llm
        self.memory = memory or ConversationMemory()
        self.logger = get_logger(logger_name)
        self._recent_turn_metrics: deque[dict[str, float]] = deque(maxlen=20)
        self._workspace_root = workspace_root

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

    def run_agentic(
        self,
        user_query: str,
        *,
        max_turns: int = 8,
        workspace_root: str | None = None,
        persist_memory: bool = True,
        cancel_event: Any | None = None,
    ) -> AgenticTurnResult:
        """
        Multi-turn tool loop: each LLM step returns JSON either final answer or tool_calls.

        Does not use Planner; tools run via Executor.execute_agentic_calls.
        """
        trace_id = uuid4().hex[:12]
        set_trace_id(trace_id)
        turn_started = time.perf_counter()
        root = workspace_root if workspace_root is not None else (self._workspace_root or ".")
        self.logger.info("Agentic run query=%s max_turns=%s root=%s", user_query, max_turns, root)
        log_event(
            self.logger,
            module="agent",
            action="run_agentic_start",
            status="ok",
            user_query_len=len(user_query),
            max_turns=max_turns,
        )

        tool_specs_json = compact_tool_specs_for_prompt(self.executor.registry.list_specs())
        write_tools_available = any(
            self.executor.registry.get_tool(name) is not None
            for name in ("apply_patch_tool", "write_file_tool")
        )
        shell_tool_available = self.executor.registry.get_tool("run_command_tool") is not None
        extra_rules: list[str] = []
        if not write_tools_available:
            extra_rules.append(
                "当前未开放写文件工具。若用户要求修改代码，最终回答必须给出可复制的 unified diff 或分步编辑说明，不得声称已写入磁盘。"
            )
        if not shell_tool_available:
            extra_rules.append(
                "当前未开放 shell/命令执行工具。不得声称已经运行测试、构建或其他命令；只能基于现有上下文给出建议。"
            )
        extra_rules_text = "\n".join(extra_rules)
        system_prompt = (
            f"工作区根目录（说明用途；调用工具时路径需与此一致）: {root}\n"
            "安全：不要执行任意 shell、不要访问工作区外路径；只使用下列工具。\n\n"
            + AGENTIC_TOOL_USE_POLICY
            + (f"{extra_rules_text}\n\n" if extra_rules_text else "")
            + f"可用工具（JSON 数组，每项含 name、description、parameters JSON Schema；调用时 arguments 必须满足 parameters）：\n"
            f"{tool_specs_json}\n\n"
            + AGENTIC_JSON_SYSTEM_SUFFIX
        )

        transcript: list[dict[str, str]] = []
        history = self.memory.get_messages()
        for item in history[-10:]:
            role = item.get("role", "")
            content = item.get("content", "")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                transcript.append({"role": role, "content": content})
        transcript.append({"role": "user", "content": user_query})

        tool_trace: list[dict[str, Any]] = []
        answer = ""
        max_turns = max(1, int(max_turns))

        for _ in range(max_turns):
            if cancel_event is not None and cancel_event.is_set():
                answer = answer or "请求已取消。"
                break
            decision = self.llm.generate_agentic_json_turn(transcript, system_prompt=system_prompt)
            transcript.append({"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)})

            if decision.get("type") == "final":
                answer = str(decision.get("content", ""))
                break

            if decision.get("type") == "tool_calls":
                calls = decision.get("calls") or []
                if not isinstance(calls, list):
                    transcript.append(
                        {
                            "role": "user",
                            "content": "工具调用格式错误：calls 必须是列表。请输出 final 或修正 tool_calls。",
                        }
                    )
                    continue
                batch = self.executor.execute_agentic_calls(calls, cancel_event=cancel_event)
                tool_trace.extend(batch)
                transcript.append({"role": "user", "content": self._format_agentic_tool_feedback(batch)})
                continue

            answer = "模型返回了未知的 type 字段。"
            break
        else:
            answer = answer or self._synthesize_agentic_answer(
                user_query=user_query,
                tool_trace=tool_trace,
                history=history,
            )

        if persist_memory and not (cancel_event is not None and cancel_event.is_set()):
            self.memory.add_user_message(user_query)
            self.memory.add_assistant_message(answer)
            self.memory.add_turn_metadata(
                plan=[],
                tool_results=tool_trace,
                recovery_applied=False,
                trace_id=trace_id,
                extra={"agentic": True},
            )

        turn_duration_ms = int((time.perf_counter() - turn_started) * 1000)
        self._record_turn_metrics(tool_results=tool_trace, duration_ms=turn_duration_ms)
        log_event(
            self.logger,
            module="agent",
            action="run_agentic_finish",
            status="ok",
            duration_ms=turn_duration_ms,
            tool_steps=len(tool_trace),
            trace_id=trace_id,
        )

        return AgenticTurnResult(
            answer=answer,
            messages=list(transcript),
            tool_trace=tool_trace,
            trace_id=trace_id,
        )

    def _synthesize_agentic_answer(
        self,
        *,
        user_query: str,
        tool_trace: list[dict[str, Any]],
        history: list[dict[str, str]],
    ) -> str:
        if not tool_trace:
            return "已达到最大对话轮次仍未给出最终回答（type=final）。"
        context = self._build_agentic_tool_context(user_query=user_query, tool_trace=tool_trace)
        answer = str(
            self.llm.generate_answer(
                user_query=user_query,
                context=context,
                history=history,
            )
            or ""
        ).strip()
        return answer or "已达到最大对话轮次，但已基于现有工具结果生成总结。"

    def _format_agentic_tool_feedback(self, results: list[dict[str, Any]]) -> str:
        lines: list[str] = ["以下为工具执行结果（每行一个 JSON 对象）："]
        for r in results:
            payload = {
                "tool": r.get("tool"),
                "step_id": r.get("step_id"),
                "status": r.get("status"),
                "output": r.get("output"),
            }
            lines.append(json.dumps(payload, ensure_ascii=False))
        return "\n".join(lines)

    def _build_agentic_tool_context(self, *, user_query: str, tool_trace: list[dict[str, Any]]) -> str:
        lines: list[str] = [f"User Query:\n{user_query}", "", "=== Agentic tool results ==="]
        for idx, result in enumerate(tool_trace, start=1):
            lines.append(
                f"[{idx}] tool={result.get('tool', 'unknown_tool')} status={result.get('status', 'unknown')}\n"
                f"Output:\n{result.get('output', '')}"
            )
        return "\n\n".join(lines).strip()

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
