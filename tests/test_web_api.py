from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.web.main import create_app
from app.web.chat_components import StreamCancelled
from app.web.service import WebAgentService
from app.web.session_store import SessionStore
from tests.web_test_utils import FakeAgentFactory, FakeLLM, FakePlanner, build_turn


def _client(
    tmp_path: Path,
    *,
    agent_factory: FakeAgentFactory | None = None,
    llm_factory: object | None = None,
) -> tuple[TestClient, SessionStore]:
    store = SessionStore(tmp_path / "sessions")
    factory = agent_factory or FakeAgentFactory(
        turns=[
            build_turn("已定位文件。", [{"tool": "search_tool", "status": "ok"}]),
            build_turn("给出补丁建议。", [{"tool": "analyze_tool", "status": "ok"}]),
            build_turn("验证完成。", [{"tool": "analyze_tool", "status": "ok"}]),
        ]
    )
    kwargs = {
        "session_store": store,
        "agent_factory": factory,
        "repo_root": tmp_path,
    }
    if llm_factory is not None:
        kwargs["llm_factory"] = llm_factory
    service = WebAgentService(**kwargs)
    client = TestClient(create_app(service))
    return client, store


class _CancellingAgent:
    def __init__(self) -> None:
        self.planner = FakePlanner()

    def run_agentic(
        self,
        user_query: str,
        *,
        max_turns: int = 8,
        workspace_root: str | None = None,
        persist_memory: bool = True,
        cancel_event: object | None = None,
    ):
        raise StreamCancelled("stream cancelled")


class _CancellingAgentFactory:
    def __call__(
        self,
        workspace_root: str,
        *,
        memory: object | None = None,
        top_k: int = 5,
        force_reindex: bool = False,
        allow_write: bool = False,
        allow_shell: bool = False,
        test_command: str = "",
        index_dir: object | None = None,
    ) -> _CancellingAgent:
        return _CancellingAgent()


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
            "title": "自定义会话",
            "pinned": True,
            "archived": True,
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
    assert updated.json()["title"] == "自定义会话"
    assert updated.json()["pinned"] is True
    assert updated.json()["archived"] is True
    assert updated.json()["settings"]["allow_shell"] is True
    assert updated.json()["settings"]["max_turns"] == 9

    listed_after_update = client.get("/sessions")
    assert listed_after_update.status_code == 200
    assert listed_after_update.json()[0]["pinned"] is True
    assert listed_after_update.json()[0]["archived"] is True

    deleted = client.delete(f"/sessions/{session_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True, "session_id": session_id}


def test_web_api_can_create_session_without_workspace(tmp_path: Path) -> None:
    client, _store = _client(tmp_path)

    created = client.post("/sessions", json={"settings": {}})

    assert created.status_code == 200
    assert created.json()["workspace_root"] == ""


def test_web_index_serves_external_assets(tmp_path: Path) -> None:
    client, _store = _client(tmp_path)

    html = client.get("/")
    assert html.status_code == 200
    assert '/static/web/index.css?v=' in html.text
    assert '/static/web/index.js?v=' in html.text

    stylesheet = client.get("/static/web/index.css")
    assert stylesheet.status_code == 200
    assert ".layout{" in stylesheet.text

    script = client.get("/static/web/index.js")
    assert script.status_code == 200
    assert "ACTIVE_SESSION_STORAGE_KEY" in script.text


def test_web_api_pick_folder_returns_workspace_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client, _store = _client(tmp_path)

    def fake_pick_local_path(selection: str) -> dict[str, object]:
        assert selection == "folder"
        return {
            "selected": True,
            "path": str(workspace),
            "workspace_root": str(workspace),
            "file_path": None,
            "relative_path": None,
        }

    client.app.state.service.pick_local_path = fake_pick_local_path

    response = client.post("/system/pick-folder")

    assert response.status_code == 200
    assert response.json()["workspace_root"] == str(workspace)
    assert response.json()["file_path"] is None


def test_web_api_pick_file_returns_parent_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "main.py"
    target.write_text("print('demo')\n", encoding="utf-8")
    client, _store = _client(tmp_path)

    def fake_pick_local_path(selection: str) -> dict[str, object]:
        assert selection == "file"
        return {
            "selected": True,
            "path": str(target),
            "workspace_root": str(workspace),
            "file_path": str(target),
            "relative_path": "main.py",
        }

    client.app.state.service.pick_local_path = fake_pick_local_path

    response = client.post("/system/pick-file")

    assert response.status_code == 200
    assert response.json()["workspace_root"] == str(workspace)
    assert response.json()["relative_path"] == "main.py"


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
    assert "这次任务已经按" in body["assistant"]
    assert body["session"]["messages"][-2]["content"] == "请给项目新增健康检查"
    assert len(body["session"]["tasks"]) == 3
    assert body["task_results"][0]["tool_success_count"] >= 0


def test_web_api_rejects_blank_message_payload(tmp_path: Path) -> None:
    client, _store = _client(tmp_path)
    created = client.post("/sessions", json={"settings": {}})
    session_id = created.json()["session_id"]

    response = client.post(
        f"/sessions/{session_id}/messages",
        json={"content": "   "},
    )

    assert response.status_code == 422


def test_web_api_qa_message_works_without_workspace(tmp_path: Path) -> None:
    fake_llm = FakeLLM(answer="这是 QA 回答。示例代码只会以内联片段给出，不会写入本地文件。")
    agent_factory = FakeAgentFactory(turns=[build_turn("不应被调用")])
    client, _store = _client(
        tmp_path,
        agent_factory=agent_factory,
        llm_factory=lambda: fake_llm,
    )
    created = client.post("/sessions", json={"settings": {}})
    session_id = created.json()["session_id"]

    response = client.post(
        f"/sessions/{session_id}/messages",
        json={"content": "解释一下并查集，并给一个示例函数"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant"] == fake_llm.answer
    assert body["session"]["workspace_root"] == ""
    assert body["session"]["tasks"] == []
    assert body["task_results"] == []
    assert agent_factory.created_agents == []


def test_web_api_workspace_tree_and_file_read(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    pkg_dir = workspace / "pkg"
    pkg_dir.mkdir(parents=True)
    (workspace / "README.md").write_text("# demo\n", encoding="utf-8")
    target = pkg_dir / "mod.py"
    target.write_text("value = 1\n", encoding="utf-8")

    client, _store = _client(tmp_path)
    created = client.post(
        "/sessions",
        json={"workspace_root": str(workspace), "settings": {"allow_write": True}},
    )
    session_id = created.json()["session_id"]

    tree = client.get(f"/sessions/{session_id}/workspace/tree")
    assert tree.status_code == 200
    entries = {item["path"]: item for item in tree.json()["entries"]}
    assert entries["pkg"]["is_dir"] is True
    assert entries["pkg/mod.py"]["is_dir"] is False
    assert entries["README.md"]["name"] == "README.md"
    assert entries["pkg/mod.py"]["size_bytes"] is None

    tree_with_metadata = client.get(
        f"/sessions/{session_id}/workspace/tree",
        params={"include_metadata": True},
    )
    assert tree_with_metadata.status_code == 200
    detailed_entries = {item["path"]: item for item in tree_with_metadata.json()["entries"]}
    assert detailed_entries["pkg/mod.py"]["size_bytes"] == target.stat().st_size
    assert detailed_entries["pkg/mod.py"]["modified_ns"] is not None

    file_response = client.get(
        f"/sessions/{session_id}/workspace/file",
        params={"path": "pkg/mod.py"},
    )
    assert file_response.status_code == 200
    body = file_response.json()
    assert body["path"] == "pkg/mod.py"
    assert body["content"] == "value = 1\n"
    assert len(body["content_sha256"]) == 64


def test_web_api_workspace_write_updates_disk(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "app.py"
    target.write_text("print('old')\n", encoding="utf-8")

    client, _store = _client(tmp_path)
    created = client.post(
        "/sessions",
        json={"workspace_root": str(workspace), "settings": {"allow_write": True}},
    )
    session_id = created.json()["session_id"]

    read_response = client.get(
        f"/sessions/{session_id}/workspace/file",
        params={"path": "app.py"},
    )
    original_hash = read_response.json()["content_sha256"]

    write_response = client.put(
        f"/sessions/{session_id}/workspace/file",
        json={
            "path": "app.py",
            "content": "print('new')\n",
            "expected_content_hash": original_hash,
        },
    )
    assert write_response.status_code == 200
    assert write_response.json()["path"] == "app.py"
    assert target.read_text(encoding="utf-8") == "print('new')\n"


def test_web_api_workspace_write_requires_allow_write(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("print('old')\n", encoding="utf-8")

    client, _store = _client(tmp_path)
    created = client.post(
        "/sessions",
        json={"workspace_root": str(workspace), "settings": {"allow_write": False}},
    )
    session_id = created.json()["session_id"]

    response = client.put(
        f"/sessions/{session_id}/workspace/file",
        json={"path": "app.py", "content": "print('new')\n"},
    )

    assert response.status_code == 403
    assert "allow_write" in response.json()["detail"]


def test_web_api_workspace_write_rejects_invalid_expected_hash(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("print('old')\n", encoding="utf-8")

    client, _store = _client(tmp_path)
    created = client.post(
        "/sessions",
        json={"workspace_root": str(workspace), "settings": {"allow_write": True}},
    )
    session_id = created.json()["session_id"]

    response = client.put(
        f"/sessions/{session_id}/workspace/file",
        json={"path": "app.py", "content": "print('new')\n", "expected_content_hash": "not-a-sha"},
    )

    assert response.status_code == 422


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

    assert "event: stream_profile" in text
    assert '"kind": "phased"' in text
    assert "event: task_board" in text
    assert "event: assistant_final" in text


def test_web_api_qa_stream_does_not_emit_task_board(tmp_path: Path) -> None:
    fake_llm = FakeLLM(answer="这是 QA 模式流式回答。")
    client, _store = _client(tmp_path, llm_factory=lambda: fake_llm)
    created = client.post("/sessions", json={"settings": {}})
    session_id = created.json()["session_id"]

    with client.stream(
        "POST",
        f"/sessions/{session_id}/messages?stream=1",
        json={"content": "解释一下快速排序原理"},
    ) as response:
        assert response.status_code == 200
        text = "".join(chunk for chunk in response.iter_text())

    assert "event: assistant_final" in text
    assert "event: task_board" not in text


def test_web_api_sse_stream_emits_cancel_error(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client, _store = _client(tmp_path, agent_factory=_CancellingAgentFactory())
    created = client.post("/sessions", json={"workspace_root": str(workspace), "settings": {}})
    session_id = created.json()["session_id"]

    with client.stream(
        "POST",
        f"/sessions/{session_id}/messages?stream=1",
        json={"content": "\u8bf7\u5206\u6790\u5e76\u4fee\u590d"},
    ) as response:
        assert response.status_code == 200
        text = "".join(chunk for chunk in response.iter_text())

    assert "event: error" in text
    assert "\u5df2\u53d6\u6d88" in text
    assert "event: assistant_final" not in text


def test_web_api_sequential_qa_stream_turns_preserve_message_order(tmp_path: Path) -> None:
    fake_llm = FakeLLM(
        answer="unused",
        call_answers=["第一轮回答。", "第二轮回答。"],
    )
    client, _store = _client(tmp_path, llm_factory=lambda: fake_llm)
    created = client.post("/sessions", json={"settings": {}})
    session_id = created.json()["session_id"]

    with client.stream(
        "POST",
        f"/sessions/{session_id}/messages?stream=1",
        json={"content": "第一个问题"},
    ) as response:
        assert response.status_code == 200
        text = "".join(chunk for chunk in response.iter_text())
    assert "event: assistant_final" in text

    with client.stream(
        "POST",
        f"/sessions/{session_id}/messages?stream=1",
        json={"content": "第二个问题"},
    ) as response:
        assert response.status_code == 200
        text = "".join(chunk for chunk in response.iter_text())
    assert "event: assistant_final" in text

    session = client.get(f"/sessions/{session_id}")
    assert session.status_code == 200
    messages = session.json()["messages"]
    assert [item["role"] for item in messages] == ["user", "assistant", "user", "assistant"]
    assert [item["content"] for item in messages] == [
        "第一个问题",
        "第一轮回答。",
        "第二个问题",
        "第二轮回答。",
    ]
