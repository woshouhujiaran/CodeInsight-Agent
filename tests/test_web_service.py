from __future__ import annotations

from pathlib import Path
import sys
from threading import Event
import time

import pytest

from app.web.service import WebAgentService
from app.web.session_store import SessionStore
from tests.web_test_utils import FakeAgentFactory, FakeLLM, FakePlanner, build_turn


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


def test_web_service_qa_mode_allows_empty_workspace_and_skips_agent_factory(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create_session(workspace_root="")
    fake_llm = FakeLLM(
        answer="二分查找每次把区间减半，时间复杂度是 O(log n)。下面是示例代码，不会自动写入本地文件。"
    )
    factory = FakeAgentFactory(turns=[build_turn("不应被调用")])
    service = WebAgentService(
        session_store=store,
        agent_factory=factory,
        llm_factory=lambda: fake_llm,
        repo_root=tmp_path,
    )

    result = service.chat(session["session_id"], "解释一下二分查找，并给一个 Python 示例")

    assert result["assistant"] == fake_llm.answer
    assert result["task_results"] == []
    assert result["session"]["workspace_root"] == ""
    assert result["session"]["tasks"] == []
    assert result["session"]["turn_metadata"][-1]["mode"] == "qa"
    assert result["session"]["turn_metadata"][-1]["tool_results"] == []
    assert factory.created_agents == []
    assert fake_llm.calls
    assert "不会自动写入任何本地文件" in fake_llm.calls[0]["prompt"]


def test_web_service_agentic_mode_requires_workspace(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create_session(workspace_root="")
    factory = FakeAgentFactory(turns=[build_turn("不应被调用")])
    service = WebAgentService(session_store=store, agent_factory=factory, repo_root=tmp_path)

    with pytest.raises(ValueError, match="workspace_root"):
        service.chat(session["session_id"], "在这个项目里修复登录 bug")

    assert factory.created_agents == []


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


class _SlowCancellableAgent:
    def __init__(self) -> None:
        self.planner = FakePlanner()
        self.started = Event()
        self.cancelled = Event()

    def run_agentic(
        self,
        user_query: str,
        *,
        max_turns: int = 8,
        workspace_root: str | None = None,
        persist_memory: bool = True,
        cancel_event: Event | None = None,
    ):
        self.started.set()
        while cancel_event is None or not cancel_event.is_set():
            time.sleep(0.05)
        self.cancelled.set()
        return build_turn("cancelled")


class _SlowAgentFactory:
    def __init__(self) -> None:
        self.agent = _SlowCancellableAgent()

    def __call__(
        self,
        workspace_root: str,
        *,
        memory: object | None = None,
        top_k: int = 5,
        force_reindex: bool = False,
        allow_write: bool = False,
        allow_shell: bool = False,
        index_dir: object | None = None,
    ) -> _SlowCancellableAgent:
        return self.agent


def test_web_service_stream_close_cancels_background_turn(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create_session(workspace_root=str(tmp_path))
    factory = _SlowAgentFactory()
    service = WebAgentService(session_store=store, agent_factory=factory, repo_root=tmp_path)

    events = service.stream_chat(session["session_id"], "在这个项目里修复登录 bug")
    assert next(events)["event"] == "mode"
    assert next(events)["event"] == "session"
    assert next(events)["event"] == "task_board"
    assert next(events)["event"] == "task_update"
    assert factory.agent.started.wait(timeout=1.0) is True

    events.close()

    deadline = time.time() + 2.0
    while time.time() < deadline and not factory.agent.cancelled.is_set():
        time.sleep(0.05)

    assert factory.agent.cancelled.is_set() is True
