from __future__ import annotations

from pathlib import Path

from app.web.session_store import SessionStore


def test_session_store_create_and_get(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    created = store.create_session(workspace_root=str(tmp_path), settings={"allow_write": True})

    loaded = store.get_session(created["session_id"])

    assert loaded["workspace_root"] == str(tmp_path)
    assert loaded["settings"]["allow_write"] is True
    assert loaded["title"] == "新会话"


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
    session["last_test_summary"] = {"passed": True, "failed": False, "duration_ms": 12, "command": "pytest", "raw_tail": ""}

    store.save_session(session)
    loaded = store.get_session(session["session_id"])

    assert loaded["title"] == "实现健康检查接口"
    assert loaded["messages"][0]["content"] == "实现健康检查接口"
    assert loaded["turn_metadata"][0]["trace_id"] == "abc"
    assert loaded["tasks"][0]["summary"] == "已定位"
    assert loaded["last_test_summary"]["passed"] is True
