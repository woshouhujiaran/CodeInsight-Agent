from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    id: str
    title: str
    description: str
    acceptance_criteria: str
    owner_role: str = "worker"
    evidence_required: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)

    def to_task_item_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "acceptance": self.acceptance_criteria,
            "owner_role": self.owner_role,
            "evidence_required": list(self.evidence_required),
            "depends_on": list(self.depends_on),
            "status": "pending",
        }
