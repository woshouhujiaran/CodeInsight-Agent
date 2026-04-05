from __future__ import annotations

from typing import Any

from app.contracts import (
    MessageModel,
    SessionSettingsModel,
    TaskResultModel,
    TestSummaryModel,
    TurnMetadataModel,
)
from pydantic import BaseModel, Field, field_validator


class SessionCreateModel(BaseModel):
    workspace_root: str = ""
    settings: SessionSettingsModel = Field(default_factory=SessionSettingsModel)

    @field_validator("workspace_root", mode="before")
    @classmethod
    def _normalize_workspace_root(cls, value: Any) -> str:
        return str(value or "").strip()


class SessionUpdateModel(BaseModel):
    workspace_root: str | None = None
    settings: SessionSettingsModel | None = None
    title: str | None = None
    pinned: bool | None = None
    archived: bool | None = None

    @field_validator("workspace_root", mode="before")
    @classmethod
    def _normalize_optional_workspace_root(cls, value: Any) -> str | None:
        if value is None:
            return None
        return str(value).strip()


class MessageCreateModel(BaseModel):
    content: str = Field(..., min_length=1)

    @field_validator("content", mode="before")
    @classmethod
    def _normalize_content(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("content cannot be empty")
        return text


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

    @field_validator("path", mode="before")
    @classmethod
    def _normalize_path(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("path cannot be empty")
        return text

    @field_validator("expected_content_hash", mode="before")
    @classmethod
    def _normalize_expected_hash(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        if not text:
            return None
        if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
            raise ValueError("expected_content_hash must be a 64-char hex sha256")
        return text


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
