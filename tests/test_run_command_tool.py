from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from app.tools.run_command_tool import RunCommandTool, argv_matches_allowlist


def _ws(tmp_path: Path) -> Path:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return workspace


def test_run_command_rejects_non_allowlisted_argv(tmp_path: Path) -> None:
    tool = RunCommandTool(workspace_root=_ws(tmp_path))
    result = tool.run({"argv": ["curl", "https://example.com"]})
    assert result["status"] == "error"
    assert result.get("meta", {}).get("allowlist_rejected") is True


def test_run_command_rejects_shell_metacharacters(tmp_path: Path) -> None:
    tool = RunCommandTool(workspace_root=_ws(tmp_path))
    result = tool.run({"argv": ["git", "status", "a|b"]})
    assert result["status"] == "error"
    assert "metacharacters" in str(result.get("error") or "")


def test_run_command_allows_safe_readonly_prefixes() -> None:
    assert argv_matches_allowlist(["rg", "--version"]) is True
    assert argv_matches_allowlist([sys.executable, "-m", "pytest", "-q"]) is True
    assert argv_matches_allowlist([sys.executable, "-m", "compileall", "--help"]) is True


def test_run_command_git_status_ok(tmp_path: Path) -> None:
    workspace = _ws(tmp_path)
    subprocess.run(["git", "init"], cwd=str(workspace), check=False, capture_output=True, text=True)
    (workspace / "a.txt").write_text("x", encoding="utf-8")
    tool = RunCommandTool(workspace_root=workspace, default_timeout_seconds=30.0)
    result = tool.run({"argv": ["git", "status", "-s"], "timeout_seconds": 15})
    assert result["status"] == "ok"
    payload = json.loads(str(result.get("data") or "{}"))
    assert "returncode" in payload
    assert payload.get("timed_out") is False


def test_run_command_allows_configured_test_command(tmp_path: Path) -> None:
    workspace = _ws(tmp_path)
    (workspace / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    tool = RunCommandTool(
        workspace_root=workspace,
        allowed_commands=[f'"{sys.executable}" -m pytest -q'],
        default_timeout_seconds=30.0,
    )

    result = tool.run({"argv": [sys.executable, "-m", "pytest", "-q"], "timeout_seconds": 15})

    assert result["status"] == "ok"
    payload = json.loads(str(result.get("data") or "{}"))
    assert payload.get("returncode") == 0
    assert payload.get("timed_out") is False


def test_run_command_pytest_prefix_executes(tmp_path: Path) -> None:
    workspace = _ws(tmp_path)
    (workspace / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    tool = RunCommandTool(workspace_root=workspace, default_timeout_seconds=30.0)

    result = tool.run({"argv": [sys.executable, "-m", "pytest", "-q"], "timeout_seconds": 15})

    assert result["status"] == "ok"
    payload = json.loads(str(result.get("data") or "{}"))
    assert payload.get("returncode") == 0


def test_run_command_command_string_split(tmp_path: Path) -> None:
    workspace = _ws(tmp_path)
    subprocess.run(["git", "init"], cwd=str(workspace), check=False, capture_output=True)
    tool = RunCommandTool(workspace_root=workspace)
    result = tool.run({"command": "git status -s"})
    assert result["status"] == "ok"
    payload = json.loads(str(result.get("data") or "{}"))
    assert payload.get("argv", [None])[0] == "git"
