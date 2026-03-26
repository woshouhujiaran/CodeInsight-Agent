from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent.executor import Executor
from app.agent.memory import ConversationMemory
from app.agent.planner import Planner
from app.llm.llm import LLMClient
from app.utils.logger import get_logger


@dataclass
class AgentTurnResult:
    """Structured response for one agent turn."""

    answer: str
    plan: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    context: str


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

        context = self._build_context(user_query=user_query, tool_results=tool_results)
        final_answer = self.llm.generate_answer(
            user_query=user_query,
            context=context,
            history=history,
        )

        self.memory.add_user_message(user_query)
        self.memory.add_assistant_message(final_answer)
        self.memory.add_turn_metadata(plan=plan, tool_results=tool_results)
        self.logger.info("Memory updated. Total messages: %d", len(self.memory.get_messages()))

        return AgentTurnResult(
            answer=final_answer,
            plan=plan,
            tool_results=tool_results,
            context=context,
        )

    def _build_context(self, user_query: str, tool_results: list[dict[str, Any]]) -> str:
        sections: list[str] = [f"User Query:\n{user_query}", "Tool Results:"]

        for idx, result in enumerate(tool_results, start=1):
            tool_name = result.get("tool", "unknown_tool")
            status = result.get("status", "unknown")
            output = result.get("output", "")
            sections.append(
                f"[{idx}] Tool: {tool_name}\nStatus: {status}\nOutput:\n{output}"
            )

        context = "\n\n".join(sections).strip()
        self.logger.debug("Built context with %d chars.", len(context))
        return context
