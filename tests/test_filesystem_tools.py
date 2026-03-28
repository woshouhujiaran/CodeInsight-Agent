from __future__ import annotations

from pathlib import Path

from app.agent.plan_schema import validate_tool_args
from app.tools.filesystem_tools import GrepTool, ListDirTool, ReadFileTool


def _ws(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


def test_read_file_rejects_path_escape(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")

    tool = ReadFileTool(workspace_root=ws)
    rel = f"../{outside.name}"
    result = tool.run({"path": rel})
    assert result["status"] == "error"
    assert "越界" in (result.get("error") or "")


def test_read_file_truncates_by_max_chars(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "big.txt").write_text("a" * 5000, encoding="utf-8")
    tool = ReadFileTool(workspace_root=ws)
    result = tool.run({"path": "big.txt", "max_chars": 100})
    assert result["status"] == "ok"
    body = str(result.get("data") or "")
    assert "truncated" in body.lower()
    assert result.get("meta", {}).get("truncated") is True
    assert len(result.get("meta", {}).get("content_sha256", "")) == 64


def test_grep_finds_match_in_subdirectory(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    pkg = ws / "pkg"
    pkg.mkdir()
    (pkg / "mod.py").write_text("foo = 1\nhello grep world\n", encoding="utf-8")

    tool = GrepTool(workspace_root=ws)
    result = tool.run({"pattern": r"hello\s+grep", "path": ".", "glob": "*.py"})
    assert result["status"] == "ok"
    out = str(result.get("data") or "").replace("\\", "/")
    assert "pkg/mod.py" in out
    assert "hello grep" in out


def test_list_dir_lists_relative_paths(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "a.txt").write_text("x", encoding="utf-8")
    d = ws / "sub"
    d.mkdir()
    (d / "b.txt").write_text("y", encoding="utf-8")

    tool = ListDirTool(workspace_root=ws)
    result = tool.run({"path": ".", "depth": 2, "max_entries": 50})
    assert result["status"] == "ok"
    data = result.get("data")
    assert isinstance(data, dict)
    paths = data.get("paths") or []
    assert set(paths) >= {"a.txt", "sub", "sub/b.txt"}


def test_validate_tool_args_read_file_path_required() -> None:
    assert validate_tool_args("read_file_tool", {}) is not None
    assert validate_tool_args("read_file_tool", {"path": "x.py"}) is None
