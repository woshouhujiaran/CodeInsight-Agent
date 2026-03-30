from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.web.main import create_app
from app.web.service import WebAgentService
from app.web.session_store import SessionStore
from tests.web_test_utils import FakeAgentFactory, build_turn


def _client(tmp_path: Path) -> tuple[TestClient, SessionStore]:
    store = SessionStore(tmp_path / "sessions")
    factory = FakeAgentFactory(
        turns=[
            build_turn("已定位文件。", [{"tool": "search_tool", "status": "ok"}]),
            build_turn("给出补丁建议。", [{"tool": "analyze_tool", "status": "ok"}]),
            build_turn("验证完成。", [{"tool": "analyze_tool", "status": "ok"}]),
        ]
    )
    service = WebAgentService(session_store=store, agent_factory=factory, repo_root=tmp_path)
    client = TestClient(create_app(service))
    return client, store


def test_web_api_session_crud(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client, _store = _client(tmp_path)

    created = client.post(
        "/sessions",
        json={"workspace_root": str(workspace), "settings": {"allow_write": True}},
    )
    assert created.status_code == 200
    session_id = created.json()["session_id"]

    listed = client.get("/sessions")
    assert listed.status_code == 200
    assert listed.json()[0]["session_id"] == session_id

    fetched = client.get(f"/sessions/{session_id}")
    assert fetched.status_code == 200
    assert fetched.json()["workspace_root"] == str(workspace)

    updated = client.patch(
        f"/sessions/{session_id}",
        json={
            "workspace_root": str(workspace),
            "settings": {
                "allow_write": False,
                "allow_shell": True,
                "auto_run_tests": False,
                "test_command": "python -m pytest -q",
                "max_turns": 9,
            },
        },
    )
    assert updated.status_code == 200
    assert updated.json()["settings"]["allow_shell"] is True
    assert updated.json()["settings"]["max_turns"] == 9


def test_web_api_post_message_non_stream(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client, _store = _client(tmp_path)
    created = client.post("/sessions", json={"workspace_root": str(workspace), "settings": {}})
    session_id = created.json()["session_id"]

    response = client.post(
        f"/sessions/{session_id}/messages",
        json={"content": "请给项目新增健康检查"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "任务执行结果" in body["assistant"]
    assert body["session"]["messages"][-2]["content"] == "请给项目新增健康检查"
    assert len(body["session"]["tasks"]) == 3


def test_web_api_sse_stream_emits_task_board_and_final(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client, _store = _client(tmp_path)
    created = client.post("/sessions", json={"workspace_root": str(workspace), "settings": {}})
    session_id = created.json()["session_id"]

    with client.stream(
        "POST",
        f"/sessions/{session_id}/messages?stream=1",
        json={"content": "请分析并修改"},
    ) as response:
        assert response.status_code == 200
        text = "".join(chunk for chunk in response.iter_text())

    assert "event: task_board" in text
    assert "event: assistant_final" in text
