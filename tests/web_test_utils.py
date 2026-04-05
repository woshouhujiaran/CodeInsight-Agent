from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agent.agent import AgenticTurnResult


def build_turn(answer: str, tool_trace: list[dict[str, Any]] | None = None) -> AgenticTurnResult:
    return AgenticTurnResult(
        answer=answer,
        messages=[],
        tool_trace=list(tool_trace or []),
        trace_id="test-trace",
    )


class FakePlanner:
    def make_task_board(self, user_query: str, history: list[dict[str, str]]) -> list[dict[str, Any]]:
        return [
            {
                "id": "t1",
                "title": "定位入口",
                "description": "找到相关文件和入口。",
                "depends_on": [],
                "status": "pending",
                "acceptance": "能够指出关键入口。",
            },
            {
                "id": "t2",
                "title": "执行修改",
                "description": "修改代码或给出补丁。",
                "depends_on": ["t1"],
                "status": "pending",
                "acceptance": "改动已落地或给出补丁。",
            },
            {
                "id": "t3",
                "title": "验证结果",
                "description": "验证改动结果并总结。",
                "depends_on": ["t2"],
                "status": "pending",
                "acceptance": "给出验证结论。",
            },
        ]


@dataclass
class FakeAgent:
    memory: Any
    turns: list[AgenticTurnResult]
    recorded_prompts: list[str] = field(default_factory=list)
    recorded_max_turns: list[int] = field(default_factory=list)
    planner: FakePlanner = field(default_factory=FakePlanner)

    def run_agentic(
        self,
        user_query: str,
        *,
        max_turns: int = 8,
        workspace_root: str | None = None,
        persist_memory: bool = True,
        cancel_event: Any | None = None,
    ) -> AgenticTurnResult:
        self.recorded_prompts.append(user_query)
        self.recorded_max_turns.append(max_turns)
        if not self.turns:
            return build_turn("没有更多假数据。")
        return self.turns.pop(0)


@dataclass
class FakeAgentFactory:
    turns: list[AgenticTurnResult]
    created_agents: list[FakeAgent] = field(default_factory=list)
    memories: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self,
        workspace_root: str,
        *,
        memory: Any = None,
        top_k: int = 5,
        force_reindex: bool = False,
        allow_write: bool = False,
        allow_shell: bool = False,
        test_command: str = "",
        index_dir: Any = None,
    ) -> FakeAgent:
        snapshot = memory.to_snapshot() if memory is not None else {"messages": [], "turn_metadata": []}
        self.memories.append(snapshot)
        self.calls.append(
            {
                "workspace_root": workspace_root,
                "top_k": top_k,
                "force_reindex": force_reindex,
                "allow_write": allow_write,
                "allow_shell": allow_shell,
                "test_command": test_command,
                "index_dir": index_dir,
            }
        )
        agent = FakeAgent(memory=memory, turns=list(self.turns))
        self.created_agents.append(agent)
        return agent


@dataclass
class FakeLLM:
    answer: str
    """按顺序消耗的答复；用于模式仲裁 + QA 等多轮 generate_text。"""
    call_answers: list[str] | None = None
    calls: list[dict[str, str | None]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._answer_queue: list[str] = list(self.call_answers or [])

    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        if self._answer_queue:
            return self._answer_queue.pop(0)
        return self.answer
