from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from app.tools.run_command_tool import RunCommandTool


def _ws(tmp_path: Path) -> Path:
    w = tmp_path / "ws"
    w.mkdir()
    return w


def test_run_command_rejects_non_allowlisted_argv(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    tool = RunCommandTool(workspace_root=ws)
    r = tool.run({"argv": ["curl", "https://example.com"]})
    assert r["status"] == "error"
    assert r.get("meta", {}).get("allowlist_rejected") is True


def test_run_command_rejects_shell_metachar_in_argv(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    tool = RunCommandTool(workspace_root=ws)
    r = tool.run({"argv": ["pytest", "a|b"]})
    assert r["status"] == "error"
    assert "元字符" in (r.get("error") or "")


def test_run_command_git_status_ok(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    subprocess.run(
        ["git", "init"],
        cwd=str(ws),
        check=False,
        capture_output=True,
        text=True,
    )
    (ws / "a.txt").write_text("x", encoding="utf-8")
    tool = RunCommandTool(workspace_root=ws, default_timeout_seconds=30.0)
    r = tool.run({"argv": ["git", "status", "-s"], "timeout_seconds": 15})
    assert r["status"] == "ok"
    data = json.loads(str(r.get("data") or "{}"))
    assert "returncode" in data
    assert data.get("timed_out") is False


def test_run_command_timeout_clear_error(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "test_slow.py").write_text(
        "import time\ndef test_slow():\n    time.sleep(30)\n",
        encoding="utf-8",
    )
    tool = RunCommandTool(workspace_root=ws, default_timeout_seconds=60.0)
    r = tool.run(
        {
            "argv": [sys.executable, "-m", "pytest", "test_slow.py", "-q"],
            "timeout_seconds": 2,
        }
    )
    assert r["status"] == "error"
    assert r.get("meta", {}).get("timed_out") is True
    assert "超时" in (r.get("error") or "")


def test_run_command_command_string_split(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    subprocess.run(["git", "init"], cwd=str(ws), check=False, capture_output=True)
    tool = RunCommandTool(workspace_root=ws)
    r = tool.run({"command": "git status -s"})
    assert r["status"] == "ok"
    body = json.loads(str(r.get("data") or "{}"))
    assert body.get("argv", [None])[0] == "git"
