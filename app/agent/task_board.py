from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agent.plan_schema import topological_sort_steps

TASK_STATUSES = frozenset({"pending", "in_progress", "done", "failed"})


@dataclass
class TaskItem:
    id: str
    title: str
    description: str
    acceptance: str
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"
    summary: str = ""

    def __post_init__(self) -> None:
        self.id = str(self.id).strip()
        self.title = str(self.title).strip()
        self.description = str(self.description).strip()
        self.acceptance = str(self.acceptance).strip()
        self.depends_on = [str(item).strip() for item in self.depends_on if str(item).strip()]
        self.status = str(self.status or "pending").strip()
        self.summary = str(self.summary or "").strip()

        if not self.id:
            raise ValueError("task id cannot be empty")
        if not self.title:
            raise ValueError(f"task `{self.id}` title cannot be empty")
        if not self.description:
            raise ValueError(f"task `{self.id}` description cannot be empty")
        if not self.acceptance:
            raise ValueError(f"task `{self.id}` acceptance cannot be empty")
        if self.status not in TASK_STATUSES:
            raise ValueError(f"task `{self.id}` has invalid status `{self.status}`")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "depends_on": list(self.depends_on),
            "status": self.status,
            "acceptance": self.acceptance,
            "summary": self.summary,
        }


@dataclass
class TaskBoard:
    tasks: list[TaskItem] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.tasks:
            return
        if len(self.tasks) < 3 or len(self.tasks) > 10:
            raise ValueError("task board must contain between 3 and 10 tasks")

        ids = [task.id for task in self.tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("task board contains duplicate task ids")

        steps = [{"id": task.id, "deps": list(task.depends_on)} for task in self.tasks]
        topological_sort_steps(steps)

        id_set = set(ids)
        for task in self.tasks:
            for dep in task.depends_on:
                if dep not in id_set:
                    raise ValueError(f"task `{task.id}` depends on unknown task `{dep}`")

    @classmethod
    def from_dicts(cls, rows: list[dict[str, Any]] | None) -> TaskBoard:
        if not rows:
            return cls(tasks=[])
        return cls(tasks=[TaskItem(**row) for row in rows])

    def to_dicts(self) -> list[dict[str, Any]]:
        return [task.to_dict() for task in self.tasks]

    def ordered_tasks(self) -> list[TaskItem]:
        if not self.tasks:
            return []
        order = topological_sort_steps(
            [{"id": task.id, "deps": list(task.depends_on)} for task in self.tasks]
        )
        by_id = {task.id: task for task in self.tasks}
        return [by_id[item["id"]] for item in order]

    def get(self, task_id: str) -> TaskItem:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise KeyError(task_id)

    def mark_in_progress(self, task_id: str) -> TaskItem:
        task = self.get(task_id)
        self._transition(task, "in_progress")
        return task

    def mark_done(self, task_id: str, *, summary: str = "") -> TaskItem:
        task = self.get(task_id)
        self._transition(task, "done")
        task.summary = str(summary or "").strip()
        return task

    def mark_failed(self, task_id: str, *, summary: str = "") -> TaskItem:
        task = self.get(task_id)
        self._transition(task, "failed")
        task.summary = str(summary or "").strip()
        return task

    def completed_summaries(self) -> list[str]:
        lines: list[str] = []
        for task in self.ordered_tasks():
            if task.status == "done" and task.summary:
                lines.append(f"{task.title}: {task.summary}")
        return lines

    def _transition(self, task: TaskItem, next_status: str) -> None:
        current = task.status
        if current == next_status:
            return
        allowed = {
            "pending": {"in_progress", "failed"},
            "in_progress": {"done", "failed"},
            "done": set(),
            "failed": set(),
        }
        if next_status not in allowed[current]:
            raise ValueError(f"invalid task transition {current} -> {next_status} for `{task.id}`")
        task.status = next_status
