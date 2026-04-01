from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, validator

try:  # pragma: no cover - Pydantic v1 fallback
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover
    ConfigDict = None


class _ExtraAllowModel(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="allow")
    else:  # pragma: no cover
        class Config:
            extra = "allow"


class MessageModel(_ExtraAllowModel):
    role: str
    content: str


DEFAULT_MAX_TURNS = 8
MIN_MAX_TURNS = 1
MAX_MAX_TURNS = 20


def normalize_max_turns(value: Any) -> int:
    if isinstance(value, bool):
        return DEFAULT_MAX_TURNS
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_TURNS
    if parsed < MIN_MAX_TURNS or parsed > MAX_MAX_TURNS:
        return DEFAULT_MAX_TURNS
    return parsed


class SessionSettingsModel(_ExtraAllowModel):
    allow_write: bool = False
    allow_shell: bool = False
    test_command: str = ""
    auto_run_tests: bool = False
    max_turns: int = Field(default=DEFAULT_MAX_TURNS, ge=MIN_MAX_TURNS, le=MAX_MAX_TURNS)

    @validator("max_turns", pre=True, always=True)
    def _normalize_max_turns(cls, value: Any) -> int:
        return normalize_max_turns(value)


class TestSummaryModel(_ExtraAllowModel):
    passed: bool
    failed: bool
    duration_ms: int = 0
    command: str = ""
    raw_tail: str = ""
    returncode: int = -1
    timed_out: bool = False
    cancelled: bool = False


class ToolTraceEntryModel(_ExtraAllowModel):
    step: int | None = None
    step_id: str = ""
    tool: str = ""
    status: str = ""
    output: str = ""
    tool_result: dict[str, Any] | None = None
    success_criteria: str = ""
    attempts: int = 0
    error_type: str = ""
    duration_ms: int = 0
    timed_out: bool = False
    deps: list[str] = Field(default_factory=list)
    replan_round: int | None = None


class TaskResultModel(_ExtraAllowModel):
    task_id: str
    title: str
    status: str
    summary: str
    answer: str = ""
    tool_trace: list[ToolTraceEntryModel] = Field(default_factory=list)


class TurnMetadataModel(_ExtraAllowModel):
    plan: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[ToolTraceEntryModel] = Field(default_factory=list)
    recovery_applied: bool = False
    trace_id: str = ""
    mode: str | None = None
    agentic: bool | None = None
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    task_results: list[TaskResultModel] = Field(default_factory=list)
    last_test_summary: TestSummaryModel | None = None


class SessionSnapshotModel(_ExtraAllowModel):
    schema_version: int
    session_id: str
    title: str
    created_at: str
    updated_at: str
    workspace_root: str = ""
    messages: list[MessageModel] = Field(default_factory=list)
    turn_metadata: list[TurnMetadataModel] = Field(default_factory=list)
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    last_test_summary: TestSummaryModel | None = None
    settings: SessionSettingsModel = Field(default_factory=SessionSettingsModel)


ServiceEventName = Literal[
    "mode",
    "session",
    "task_board",
    "task_update",
    "assistant_delta",
    "assistant_final",
    "test_summary",
    "error",
]


class ServiceEvent(TypedDict):
    event: ServiceEventName
    data: Any


def dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def normalize_messages(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized.append(
            dump_model(
                MessageModel(
                    role=str(row.get("role") or ""),
                    content=str(row.get("content") or ""),
                )
            )
        )
    return normalized


def normalize_tool_trace(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized.append(dump_model(ToolTraceEntryModel(**row)))
    return normalized


def normalize_task_results(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized.append(dump_model(TaskResultModel(**row)))
    return normalized


def normalize_test_summary(summary: Any) -> dict[str, Any] | None:
    if not isinstance(summary, dict):
        return None
    return dump_model(TestSummaryModel(**summary))


def normalize_turn_metadata(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        payload = dict(row)
        payload["tool_results"] = normalize_tool_trace(payload.get("tool_results"))
        payload["task_results"] = normalize_task_results(payload.get("task_results"))
        payload["last_test_summary"] = normalize_test_summary(payload.get("last_test_summary"))
        normalized.append(dump_model(TurnMetadataModel(**payload)))
    return normalized
