from __future__ import annotations

from pathlib import Path
import sys

import pytest

from app.web.test_runner import run_project_test_command, split_command


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def test_split_command_strips_windows_quotes() -> None:
    command = f'"{sys.executable}" -m pytest -q'
    parts = split_command(command)
    assert parts[0] == sys.executable
    assert parts[1:] == ["-m", "pytest", "-q"]


def test_run_project_test_command_success(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    (ws / "test_ok.py").write_text("def test_ok():\n    assert 1 == 1\n", encoding="utf-8")

    summary = run_project_test_command(
        workspace_root=str(ws),
        command=f'"{sys.executable}" -m pytest -q',
        allow_shell=True,
        timeout_seconds=60,
    )

    assert summary["passed"] is True
    assert summary["failed"] is False
    assert summary["duration_ms"] >= 0
    assert "pytest" in summary["command"]


def test_run_project_test_command_failure(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    (ws / "test_fail.py").write_text("def test_fail():\n    assert 1 == 2\n", encoding="utf-8")

    summary = run_project_test_command(
        workspace_root=str(ws),
        command=f'"{sys.executable}" -m pytest -q',
        allow_shell=True,
        timeout_seconds=60,
    )

    assert summary["passed"] is False
    assert summary["failed"] is True
    assert "FAILED" in summary["raw_tail"] or "failed" in summary["raw_tail"].lower()


def test_run_project_test_command_requires_shell_permission(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    with pytest.raises(PermissionError):
        run_project_test_command(
            workspace_root=str(ws),
            command=f'"{sys.executable}" -m pytest -q',
            allow_shell=False,
        )
