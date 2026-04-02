from __future__ import annotations

import json
from pathlib import Path

from app.web.session_store import DEFAULT_SESSION_TITLE, SessionStore


def test_session_store_create_and_get(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    created = store.create_session(workspace_root=str(tmp_path), settings={"allow_write": True})

    loaded = store.get_session(created["session_id"])

    assert loaded["workspace_root"] == str(tmp_path)
    assert loaded["settings"]["allow_write"] is True
    assert loaded["title"] == DEFAULT_SESSION_TITLE
    assert loaded["pinned"] is False
    assert loaded["archived"] is False


def test_session_store_allows_empty_workspace_root(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    created = store.create_session(workspace_root="")

    loaded = store.get_session(created["session_id"])

    assert loaded["workspace_root"] == ""


def test_session_store_list_sessions_sorted_by_updated_at(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    first = store.create_session(workspace_root=str(tmp_path / "one"))
    second = store.create_session(workspace_root=str(tmp_path / "two"))

    sessions = store.list_sessions()

    assert [item["session_id"] for item in sessions][:2] == [second["session_id"], first["session_id"]]


def test_session_store_persists_messages_tasks_and_test_summary(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create_session(workspace_root=str(tmp_path))
    session["messages"] = [{"role": "user", "content": "实现健康检查接口"}]
    session["turn_metadata"] = [{"trace_id": "abc"}]
    session["tasks"] = [
        {
            "id": "t1",
            "title": "定位",
            "description": "定位入口",
            "depends_on": [],
            "status": "done",
            "acceptance": "找到入口",
            "summary": "已定位",
        },
        {
            "id": "t2",
            "title": "修改",
            "description": "修改代码",
            "depends_on": ["t1"],
            "status": "pending",
            "acceptance": "改动完成",
            "summary": "",
        },
        {
            "id": "t3",
            "title": "验证",
            "description": "运行测试",
            "depends_on": ["t2"],
            "status": "pending",
            "acceptance": "测试通过",
            "summary": "",
        },
    ]
    session["last_test_summary"] = {
        "passed": True,
        "failed": False,
        "duration_ms": 12,
        "command": "pytest",
        "raw_tail": "",
    }

    store.save_session(session)
    loaded = store.get_session(session["session_id"])

    assert loaded["title"] == "实现健康检查接口"
    assert loaded["messages"][0]["content"] == "实现健康检查接口"
    assert loaded["turn_metadata"][0]["trace_id"] == "abc"
    assert loaded["tasks"][0]["summary"] == "已定位"
    assert loaded["last_test_summary"]["passed"] is True


def test_session_store_update_session_supports_pin_archive_and_rename(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create_session(workspace_root=str(tmp_path))

    updated = store.update_session(
        session["session_id"],
        title="自定义标题",
        pinned=True,
        archived=True,
    )

    loaded = store.get_session(session["session_id"])

    assert updated["title"] == "自定义标题"
    assert loaded["title"] == "自定义标题"
    assert loaded["pinned"] is True
    assert loaded["archived"] is True
    assert loaded["title_overridden"] is True


def test_session_store_coerces_missing_pin_and_archive_fields(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session_path = store.root_dir / "legacy.json"
    session_path.parent.mkdir(parents=True, exist_ok=True)
    session_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_id": "legacy",
                "title": "",
                "created_at": "2024-01-01T00:00:00+00:00",
                "updated_at": "2024-01-01T00:00:00+00:00",
                "workspace_root": "",
                "messages": [],
                "turn_metadata": [],
                "tasks": [],
                "last_test_summary": None,
                "settings": {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    loaded = store.get_session("legacy")

    assert loaded["pinned"] is False
    assert loaded["archived"] is False
    assert loaded["title"] == DEFAULT_SESSION_TITLE


def test_session_store_delete_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    created = store.create_session(workspace_root=str(tmp_path))

    store.delete_session(created["session_id"])

    assert store.list_sessions() == []
