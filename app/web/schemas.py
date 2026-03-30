from __future__ import annotations

from typing import Any

from app.contracts import (
    MessageModel,
    SessionSettingsModel,
    TaskResultModel,
    TestSummaryModel,
    TurnMetadataModel,
)
from pydantic import BaseModel, Field


class SessionCreateModel(BaseModel):
    workspace_root: str = ""
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
    messages: list[MessageModel]
    turn_metadata: list[TurnMetadataModel]
    tasks: list[dict[str, Any]]
    last_test_summary: TestSummaryModel | None = None
    settings: SessionSettingsModel


class ChatResponseModel(BaseModel):
    session: SessionSnapshotModel
    assistant: str
    task_results: list[TaskResultModel]
    last_test_summary: TestSummaryModel | None = None


class EvalLatestResponseModel(BaseModel):
    path: str | None = None
    payload: dict[str, Any] | None = None
