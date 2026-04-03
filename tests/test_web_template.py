from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

INDEX_TEMPLATE_PATH = Path("app/web/templates/index.html")
INDEX_STYLESHEET_PATH = Path("app/web/static/web/index.css")
INDEX_SCRIPT_PATH = Path("app/web/static/web/index.js")


def _read_index_html() -> str:
    return INDEX_TEMPLATE_PATH.read_text(encoding="utf-8")


def _read_index_script() -> str:
    return INDEX_SCRIPT_PATH.read_text(encoding="utf-8")


def test_index_template_includes_stream_cancel_and_multiline_sse_guards() -> None:
    script = _read_index_script()

    assert 'dataLines.join("\\n")' in script
    assert 'setStreamTerminalStatus("已取消")' in script
    assert 'sendBtnEl.textContent = busy ? "停止 Stop" : "发送";' in script
    assert "runTestsBtnEl.disabled = busy;" in script
    assert "saveSettingsBtnEl.disabled = busy;" in script
    assert 'if (eventName === "error")' in script
    assert "scheduleChangeSummaryRender();" in script


def test_index_template_includes_workspace_editor_shell() -> None:
    html = _read_index_html()
    script = _read_index_script()

    assert 'id="workspaceTree"' in html
    assert 'id="sessionRail"' in html
    assert 'id="currentSessionRail"' in html
    assert 'id="sessionList"' in html
    assert 'id="fileTabs"' in html
    assert 'id="editorTextarea"' in html
    assert 'id="saveFileBtn"' in html
    assert "tree-file-marker" in script


def test_index_template_restores_active_session_and_uses_resilient_bootstrap() -> None:
    script = _read_index_script()

    assert "ACTIVE_SESSION_STORAGE_KEY" in script
    assert "persistActiveSessionId(" in script
    assert "Promise.allSettled([" in script
    assert 'if (evalResult.status === "rejected") renderEval(null);' in script


def test_index_template_uses_expected_resize_direction_for_assistant_panels() -> None:
    script = _read_index_script()

    assert "const MIN_EDITOR_WIDTH = 6;" in script
    assert "const MAX_SESSION_NAV_WIDTH = 520;" in script
    assert "const MAX_ASSISTANT_CHAT_WIDTH = 760;" in script
    assert "const startWidth = state.sessionNavWidth;" in script
    assert "state.sessionNavWidth = clampSessionNavWidth(startWidth + delta);" in script
    assert "const startWidth = state.assistantChatWidth;" in script
    assert "state.assistantChatWidth = clampAssistantChatWidth(startWidth + delta);" in script


def test_index_template_links_external_assets() -> None:
    html = _read_index_html()

    assert '<link rel="stylesheet" href="/static/web/index.css">' in html
    assert '<script src="/static/web/index.js"></script>' in html
    assert "<style>" not in html
    assert "<script>" not in html
    assert INDEX_STYLESHEET_PATH.exists()
    assert INDEX_SCRIPT_PATH.exists()


def test_index_script_is_valid_javascript() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available")

    result = subprocess.run(
        [node, "--check", str(INDEX_SCRIPT_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
