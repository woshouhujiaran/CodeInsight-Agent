from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from app.contracts import normalize_messages, normalize_test_summary, normalize_tool_trace, normalize_turn_metadata


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
        extra: dict[str, Any] | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "plan": plan,
            "tool_results": normalize_tool_trace(tool_results),
            "recovery_applied": recovery_applied,
            "trace_id": trace_id,
        }
        if extra:
            row.update(extra)
            if "last_test_summary" in row:
                row["last_test_summary"] = normalize_test_summary(row.get("last_test_summary"))
        self.turn_metadata.append(normalize_turn_metadata([row])[0])

    def get_messages(self) -> list[dict[str, str]]:
        return copy.deepcopy(self.messages)

    def get_turn_metadata(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.turn_metadata)

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "messages": self.get_messages(),
            "turn_metadata": self.get_turn_metadata(),
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any] | None) -> ConversationMemory:
        if not isinstance(snapshot, dict):
            return cls()
        messages = normalize_messages(snapshot.get("messages"))
        turn_metadata = normalize_turn_metadata(snapshot.get("turn_metadata"))
        return cls(
            messages=copy.deepcopy(messages),
            turn_metadata=copy.deepcopy(turn_metadata),
        )

    def clear(self) -> None:
        self.messages.clear()
        self.turn_metadata.clear()
