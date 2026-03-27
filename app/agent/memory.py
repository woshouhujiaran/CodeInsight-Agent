from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationMemory:
    """Store multi-turn messages and task metadata."""

    messages: list[dict[str, str]] = field(default_factory=list)
    turn_metadata: list[dict[str, Any]] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def add_turn_metadata(
        self,
        plan: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        *,
        recovery_applied: bool = False,
        trace_id: str = "",
    ) -> None:
        self.turn_metadata.append(
            {
                "plan": plan,
                "tool_results": tool_results,
                "recovery_applied": recovery_applied,
                "trace_id": trace_id,
            }
        )

    def get_messages(self) -> list[dict[str, str]]:
        return self.messages.copy()

    def clear(self) -> None:
        self.messages.clear()
        self.turn_metadata.clear()
