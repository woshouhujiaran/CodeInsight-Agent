from __future__ import annotations

from pathlib import Path

from app.tools.filesystem_tools import ReadFileTool
from app.tools.write_tools import ApplyPatchTool, WriteFileTool


def _ws(tmp_path: Path) -> Path:
    w = tmp_path / "ws"
    w.mkdir()
    return w


def test_apply_patch_modifies_file(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "hello.txt").write_text("line1\nline2\n", encoding="utf-8")
    patch = (
        "--- a/hello.txt\n"
        "+++ b/hello.txt\n"
        "@@ -1,2 +1,3 @@\n"
        " line1\n"
        "+inserted\n"
        " line2\n"
    )
    tool = ApplyPatchTool(workspace_root=ws)
    r = tool.run({"patch": patch})
    assert r["status"] == "ok"
    assert "inserted" in (ws / "hello.txt").read_text(encoding="utf-8")


def test_apply_patch_rejects_path_outside_workspace(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    rel = f"../{outside.name}"
    patch = (
        f"--- a/{rel}\n"
        f"+++ b/{rel}\n"
        "@@ -1,1 +1,1 @@\n"
        "-x\n"
        "+y\n"
    )
    tool = ApplyPatchTool(workspace_root=ws)
    r = tool.run({"patch": patch})
    assert r["status"] == "error"
    assert r.get("meta", {}).get("invalid_patch_paths") is True


def test_write_file_rejects_escape(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    tool = WriteFileTool(workspace_root=ws)
    r = tool.run({"path": "../evil.txt", "content": "h"})
    assert r["status"] == "error"
    assert "越界" in (r.get("error") or "")


def test_write_file_expected_hash_mismatch(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "f.txt").write_text("alpha", encoding="utf-8")
    tool = WriteFileTool(workspace_root=ws)
    r = tool.run(
        {
            "path": "f.txt",
            "content": "beta",
            "expected_content_hash": "0" * 64,
        }
    )
    assert r["status"] == "error"
    assert "expected_content_hash" in (r.get("error") or "")


def test_write_file_accepts_hash_from_read_file_tool(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "f.txt").write_text("alpha\n", encoding="utf-8")
    read_tool = ReadFileTool(workspace_root=ws)
    write_tool = WriteFileTool(workspace_root=ws)

    read_result = read_tool.run({"path": "f.txt"})
    content_hash = read_result.get("meta", {}).get("content_sha256")
    write_result = write_tool.run(
        {
            "path": "f.txt",
            "content": "beta\n",
            "expected_content_hash": content_hash,
        }
    )

    assert write_result["status"] == "ok"
    assert (ws / "f.txt").read_text(encoding="utf-8") == "beta\n"
