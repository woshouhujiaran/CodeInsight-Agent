from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.agent.agent import CodeAgent
from app.agent.executor import Executor
from app.agent.memory import ConversationMemory
from app.agent.tool_registry import ToolRegistry
from app.llm.llm import LLMClient
from app.tools.filesystem_tools import ListDirTool


class StaticPlanner:
    def make_task_board(self, user_query: str, history: list[dict[str, str]]) -> list[dict[str, Any]]:
        return [
            {
                "id": "t1",
                "title": "Collect workspace evidence",
                "description": "Use a tool to inspect workspace files.",
                "acceptance": "At least one tool call succeeds with concrete file paths.",
                "depends_on": [],
                "status": "pending",
            },
            {
                "id": "t2",
                "title": "Summarize evidence",
                "description": "Give a concise user-facing summary.",
                "acceptance": "Summary references observed paths.",
                "depends_on": ["t1"],
                "status": "pending",
            },
        ]


class DemoLLM(LLMClient):
    def __init__(self) -> None:
        super().__init__(provider="none", model="demo")
        self._turn = 0

    def generate_agentic_json_turn(
        self,
        messages: list[dict[str, str]],
        *,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        self._turn += 1
        if self._turn % 2 == 1:
            return {
                "type": "tool_calls",
                "calls": [
                    {
                        "name": "list_dir_tool",
                        "arguments": {"path": ".", "depth": 1, "max_entries": 12},
                    }
                ],
            }
        return {"type": "final", "content": f"task-{(self._turn + 1) // 2} done"}

    def generate_answer(
        self,
        user_query: str,
        context: str,
        history: list[dict[str, str]],
    ) -> str:
        tool_steps = context.count("tool=")
        return f"Demo complete: {tool_steps} tool step(s) executed for query: {user_query}"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    llm = DemoLLM()
    planner = StaticPlanner()
    registry = ToolRegistry()
    registry.register(ListDirTool(workspace_root=root))
    executor = Executor(registry=registry)
    memory = ConversationMemory()
    agent = CodeAgent(
        planner=planner,  # type: ignore[arg-type]
        executor=executor,
        llm=llm,
        memory=memory,
        workspace_root=str(root),
    )

    result = agent.run_minimal_agent_loop(
        "List top-level workspace files and summarize them in one sentence.",
        max_tasks=2,
        max_turns_per_task=4,
        workspace_root=str(root),
    )
    turn_metadata = memory.get_turn_metadata()
    latest_turn = turn_metadata[-1] if turn_metadata else {}

    payload = {
        "answer": result.answer,
        "tool_steps": len(result.tool_trace),
        "tools": [row.get("tool") for row in result.tool_trace],
        "trace_id": result.trace_id,
        "reasoning_steps_count": len(latest_turn.get("reasoning_steps") or []),
        "repro_manifest": latest_turn.get("repro_manifest") or {},
        "first_input_args": (result.tool_trace[0].get("input_args") if result.tool_trace else {}),
        "first_output_preview": str(result.tool_trace[0].get("output", ""))[:200] if result.tool_trace else "",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
