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
    title: str | None = None
    pinned: bool | None = None
    archived: bool | None = None


class MessageCreateModel(BaseModel):
    content: str = Field(..., min_length=1)


class WorkspaceTreeEntryModel(BaseModel):
    path: str
    name: str
    is_dir: bool
    size_bytes: int | None = None
    modified_ns: int | None = None


class WorkspaceTreeResponseModel(BaseModel):
    workspace_root: str
    path: str
    entries: list[WorkspaceTreeEntryModel]
    note: str | None = None


class WorkspaceFileResponseModel(BaseModel):
    workspace_root: str
    path: str
    content: str
    content_sha256: str
    truncated: bool = False
    returned_chars: int = 0


class WorkspaceFileUpdateModel(BaseModel):
    path: str = Field(..., min_length=1)
    content: str
    expected_content_hash: str | None = None


class WorkspaceFileWriteResponseModel(BaseModel):
    workspace_root: str
    path: str
    content_sha256: str
    message: str


class LocalPathPickerResponseModel(BaseModel):
    selected: bool
    path: str | None = None
    workspace_root: str = ""
    file_path: str | None = None
    relative_path: str | None = None


class SessionSummaryModel(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str
    workspace_root: str
    pinned: bool = False
    archived: bool = False


class SessionSnapshotModel(BaseModel):
    schema_version: int
    session_id: str
    title: str
    created_at: str
    updated_at: str
    workspace_root: str
    pinned: bool = False
    archived: bool = False
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
