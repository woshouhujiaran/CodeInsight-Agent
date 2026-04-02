from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import pytest


def test_index_template_includes_stream_cancel_and_multiline_sse_guards() -> None:
    html = Path("app/web/templates/index.html").read_text(encoding="utf-8")

    assert 'dataLines.join("\\n")' in html
    assert 'setStreamTerminalStatus("已取消")' in html
    assert 'sendBtnEl.textContent = busy ? "停止 Stop" : "发送";' in html
    assert 'runTestsBtnEl.disabled = busy;' in html
    assert 'saveSettingsBtnEl.disabled = busy;' in html
    assert 'if (eventName === "error")' in html
    assert 'scheduleChangeSummaryRender();' in html


def test_index_template_includes_workspace_editor_shell() -> None:
    html = Path("app/web/templates/index.html").read_text(encoding="utf-8")

    assert 'id="workspaceTree"' in html
    assert 'id="sessionRail"' in html
    assert 'id="currentSessionRail"' in html
    assert 'id="sessionList"' in html
    assert 'id="fileTabs"' in html
    assert 'id="editorTextarea"' in html
    assert 'id="saveFileBtn"' in html
    assert 'tree-file-marker' in html


def test_index_template_restores_active_session_and_uses_resilient_bootstrap() -> None:
    html = Path("app/web/templates/index.html").read_text(encoding="utf-8")

    assert "ACTIVE_SESSION_STORAGE_KEY" in html
    assert "persistActiveSessionId(" in html
    assert "Promise.allSettled([" in html
    assert 'if (evalResult.status === "rejected") renderEval(null);' in html


def test_index_template_inline_script_is_valid_javascript() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available")

    html = Path("app/web/templates/index.html").read_text(encoding="utf-8")
    match = re.search(r"<script>([\s\S]*)</script>", html)
    assert match is not None

    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = Path(tmp_dir) / "index.inline.js"
        script_path.write_text(match.group(1), encoding="utf-8")
        result = subprocess.run(
            [node, "--check", str(script_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
