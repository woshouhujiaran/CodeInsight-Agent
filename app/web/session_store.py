from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from uuid import uuid4

from app.contracts import (
    SessionSettingsModel,
    SessionSnapshotModel as SessionSnapshotRecord,
    dump_model,
    normalize_messages,
    normalize_max_turns,
    normalize_test_summary,
    normalize_turn_metadata,
)

SESSION_SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def derive_session_title(messages: list[dict[str, Any]]) -> str:
    for item in messages:
        if item.get("role") != "user":
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        return content[:40]
    return "新会话"


def default_session_settings() -> dict[str, Any]:
    return dump_model(SessionSettingsModel())


def normalize_session_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    merged = default_session_settings()
    if not isinstance(settings, dict):
        return merged
    if "allow_write" in settings:
        merged["allow_write"] = bool(settings.get("allow_write"))
    if "allow_shell" in settings:
        merged["allow_shell"] = bool(settings.get("allow_shell"))
    if "test_command" in settings:
        merged["test_command"] = str(settings.get("test_command") or "").strip()
    if "auto_run_tests" in settings:
        merged["auto_run_tests"] = bool(settings.get("auto_run_tests"))
    if "max_turns" in settings:
        merged["max_turns"] = normalize_max_turns(settings.get("max_turns"))
    return dump_model(SessionSettingsModel(**merged))


def coerce_session_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    created_at = str(snapshot.get("created_at") or utc_now_iso())
    updated_at = str(snapshot.get("updated_at") or created_at)
    messages = normalize_messages(snapshot.get("messages"))
    turn_metadata = normalize_turn_metadata(snapshot.get("turn_metadata"))
    tasks = snapshot.get("tasks")
    settings = normalize_session_settings(snapshot.get("settings"))
    normalized = dump_model(
        SessionSnapshotRecord(
            schema_version=int(snapshot.get("schema_version") or SESSION_SCHEMA_VERSION),
            session_id=str(snapshot.get("session_id") or uuid4().hex),
            title=str(snapshot.get("title") or derive_session_title(messages)),
            created_at=created_at,
            updated_at=updated_at,
            workspace_root=str(snapshot.get("workspace_root") or ""),
            messages=messages,
            turn_metadata=turn_metadata,
            tasks=tasks if isinstance(tasks, list) else [],
            last_test_summary=normalize_test_summary(snapshot.get("last_test_summary")),
            settings=settings,
        )
    )
    if not normalized["title"]:
        normalized["title"] = derive_session_title(normalized["messages"])
    return normalized


class SessionStore:
    def __init__(self, root_dir: str | Path = "data/sessions") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def session_path(self, session_id: str) -> Path:
        return self.root_dir / f"{session_id}.json"

    def create_session(
        self,
        *,
        workspace_root: str,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_now_iso()
        snapshot = coerce_session_snapshot(
            {
                "schema_version": SESSION_SCHEMA_VERSION,
                "session_id": uuid4().hex,
                "title": "新会话",
                "created_at": now,
                "updated_at": now,
                "workspace_root": workspace_root,
                "messages": [],
                "turn_metadata": [],
                "tasks": [],
                "last_test_summary": None,
                "settings": normalize_session_settings(settings),
            }
        )
        self.save_session(snapshot)
        return snapshot

    def save_session(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        normalized = coerce_session_snapshot(snapshot)
        normalized["updated_at"] = utc_now_iso()
        if not normalized["title"] or normalized["title"] == "新会话":
            normalized["title"] = derive_session_title(normalized["messages"])
        path = self.session_path(normalized["session_id"])
        self._atomic_write_json(path, normalized)
        return normalized

    def get_session(self, session_id: str) -> dict[str, Any]:
        path = self.session_path(session_id)
        if not path.is_file():
            raise KeyError(session_id)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return coerce_session_snapshot(payload)

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for path in self.root_dir.glob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                sessions.append(coerce_session_snapshot(payload))
            except (OSError, json.JSONDecodeError):
                continue
        sessions.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return sessions

    def update_session(
        self,
        session_id: str,
        *,
        workspace_root: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        snapshot = self.get_session(session_id)
        if workspace_root is not None:
            snapshot["workspace_root"] = workspace_root
        if settings is not None:
            merged = dict(snapshot.get("settings") or {})
            merged.update(settings)
            snapshot["settings"] = normalize_session_settings(merged)
        return self.save_session(snapshot)

    def delete_session(self, session_id: str) -> None:
        path = self.session_path(session_id)
        if not path.is_file():
            raise KeyError(session_id)
        path.unlink()

    def _atomic_write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".session_", dir=str(path.parent), text=False)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
            os.replace(tmp_name, path)
        except Exception:
            Path(tmp_name).unlink(missing_ok=True)
            raise
