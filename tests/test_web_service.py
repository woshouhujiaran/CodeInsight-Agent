from __future__ import annotations

from pathlib import Path
import sys

from app.web.service import WebAgentService
from app.web.session_store import SessionStore
from tests.web_test_utils import FakeAgentFactory, build_turn


def test_web_service_restores_memory_snapshot(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create_session(workspace_root=str(tmp_path))
    session["messages"] = [
        {"role": "user", "content": "旧问题"},
        {"role": "assistant", "content": "旧回答"},
    ]
    session["turn_metadata"] = [{"trace_id": "old-trace"}]
    session = store.save_session(session)

    factory = FakeAgentFactory(
        turns=[
            build_turn("已定位入口。"),
            build_turn("给出补丁建议。"),
            build_turn("验证完成。"),
        ]
    )
    service = WebAgentService(session_store=store, agent_factory=factory, repo_root=tmp_path)

    result = service.chat(session["session_id"], "新增一个接口")

    assert factory.memories[0]["messages"][0]["content"] == "旧问题"
    assert result["session"]["messages"][-2]["content"] == "新增一个接口"
    assert "任务执行结果" in result["session"]["messages"][-1]["content"]
    assert len(result["task_results"]) == 3


def test_web_service_returns_diff_guidance_when_write_disabled(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create_session(
        workspace_root=str(tmp_path),
        settings={"allow_write": False},
    )
    diff_text = "--- a/app.py\n+++ b/app.py\n@@\n-print('old')\n+print('new')"
    factory = FakeAgentFactory(
        turns=[
            build_turn("已定位相关文件。"),
            build_turn("建议补丁如下：\n" + diff_text),
            build_turn("验证阶段保持该补丁建议。"),
        ]
    )
    service = WebAgentService(session_store=store, agent_factory=factory, repo_root=tmp_path)

    result = service.chat(session["session_id"], "修改打印内容")

    assert "diff" in result["assistant"] or "--- a/app.py" in result["assistant"]
    assert "写权限：关闭" in factory.created_agents[0].recorded_prompts[0]


def test_web_service_auto_runs_tests_only_after_successful_write(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    store = SessionStore(tmp_path / "sessions")
    session = store.create_session(
        workspace_root=str(workspace),
        settings={
            "allow_write": True,
            "allow_shell": True,
            "auto_run_tests": True,
            "test_command": f'"{sys.executable}" -m pytest -q',
        },
    )
    factory = FakeAgentFactory(
        turns=[
            build_turn("已定位文件。", [{"tool": "search_tool", "status": "ok"}]),
            build_turn("已写入修改。", [{"tool": "apply_patch_tool", "status": "ok"}]),
            build_turn("验证摘要。", [{"tool": "analyze_tool", "status": "ok"}]),
        ]
    )
    service = WebAgentService(session_store=store, agent_factory=factory, repo_root=tmp_path)

    result = service.chat(session["session_id"], "新增健康检查")

    assert result["last_test_summary"] is not None
    assert result["last_test_summary"]["passed"] is True
    saved = store.get_session(session["session_id"])
    assert saved["last_test_summary"]["passed"] is True


def test_web_service_skips_auto_tests_without_successful_write(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = SessionStore(tmp_path / "sessions")
    session = store.create_session(
        workspace_root=str(workspace),
        settings={
            "allow_write": True,
            "allow_shell": True,
            "auto_run_tests": True,
            "test_command": f'"{sys.executable}" -m pytest -q',
        },
    )
    factory = FakeAgentFactory(
        turns=[
            build_turn("已定位文件。", [{"tool": "search_tool", "status": "ok"}]),
            build_turn("给出分析建议。", [{"tool": "analyze_tool", "status": "ok"}]),
            build_turn("验证摘要。", [{"tool": "analyze_tool", "status": "ok"}]),
        ]
    )
    service = WebAgentService(session_store=store, agent_factory=factory, repo_root=tmp_path)

    result = service.chat(session["session_id"], "仅分析，不写入")

    assert result["last_test_summary"] is None


def test_web_service_stream_emits_session_with_current_user_message(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create_session(workspace_root=str(tmp_path))
    factory = FakeAgentFactory(
        turns=[
            build_turn("已定位入口。"),
            build_turn("给出补丁建议。"),
            build_turn("验证完成。"),
        ]
    )
    service = WebAgentService(session_store=store, agent_factory=factory, repo_root=tmp_path)

    events = list(service.stream_chat(session["session_id"], "当前新消息"))

    first_session = next(item for item in events if item["event"] == "session")
    assert first_session["data"]["messages"][-1]["content"] == "当前新消息"
    assert any(item["event"] == "assistant_delta" for item in events)


def test_web_service_can_delete_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create_session(workspace_root=str(tmp_path))
    service = WebAgentService(session_store=store, repo_root=tmp_path)

    result = service.delete_session(session["session_id"])

    assert result == {"deleted": True, "session_id": session["session_id"]}
    assert store.list_sessions() == []
