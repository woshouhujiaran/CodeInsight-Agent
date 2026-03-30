from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SessionSettingsModel(BaseModel):
    allow_write: bool = False
    allow_shell: bool = False
    test_command: str = ""
    auto_run_tests: bool = False
    max_turns: int = 8


class SessionCreateModel(BaseModel):
    workspace_root: str = Field(..., min_length=1)
    settings: SessionSettingsModel = Field(default_factory=SessionSettingsModel)


class SessionUpdateModel(BaseModel):
    workspace_root: str | None = None
    settings: SessionSettingsModel | None = None


class MessageCreateModel(BaseModel):
    content: str = Field(..., min_length=1)


class SessionSummaryModel(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str
    workspace_root: str


class SessionSnapshotModel(BaseModel):
    schema_version: int
    session_id: str
    title: str
    created_at: str
    updated_at: str
    workspace_root: str
    messages: list[dict[str, str]]
    turn_metadata: list[dict[str, Any]]
    tasks: list[dict[str, Any]]
    last_test_summary: dict[str, Any] | None = None
    settings: SessionSettingsModel


class ChatResponseModel(BaseModel):
    session: SessionSnapshotModel
    assistant: str
    task_results: list[dict[str, Any]]
    last_test_summary: dict[str, Any] | None = None


class EvalLatestResponseModel(BaseModel):
    path: str | None = None
    payload: dict[str, Any] | None = None
