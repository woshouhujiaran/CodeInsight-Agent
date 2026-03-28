from __future__ import annotations

import hashlib
import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

from patch_ng import PatchSet

from app.tools.base_tool import BaseTool, make_tool_result
from app.tools.filesystem_tools import _relative_posix_from_root, _safe_resolve_under_root
from app.utils.logger import get_logger


def _sha256_utf8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write_utf8(path: Path, content: str) -> None:
    """Write UTF-8 text via temp file in the same directory, then os.replace."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".ciaw_", dir=str(path.parent), text=False)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content.encode("utf-8"))
        os.replace(str(tmp_path), str(path))
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _is_null_patch_path(raw: bytes | None) -> bool:
    if raw is None:
        return True
    s = raw.strip().lower()
    return s in (b"/dev/null", b"nul", b"//dev/null", b"")


def _decode_patch_path_for_resolve(root: Path, raw: bytes) -> tuple[Path | None, str | None]:
    """Map a ---/+++ path label to a workspace path; None if /dev/null."""
    if _is_null_patch_path(raw):
        return None, None
    s = raw.decode("utf-8", errors="replace").strip().replace("\\", "/")
    for pref in ("a/", "b/"):
        if s.startswith(pref):
            s = s[len(pref) :]
            break
    if not s.strip():
        return None, "补丁片段包含空路径"
    resolved, err = _safe_resolve_under_root(root, s)
    return resolved, err


def _validate_patch_stays_in_workspace(ps: PatchSet, root: Path) -> str | None:
    for item in ps.items:
        for label in (item.source, item.target):
            _rp, err = _decode_patch_path_for_resolve(root, label)
            if err:
                return err
    return None


def _validate_patch_header_paths(patch_text: str, root: Path) -> str | None:
    """Validate ---/+++ paths from raw diff before patch-ng normalizes them (e.g. strips ..)."""
    for line in patch_text.splitlines():
        if not (line.startswith("--- ") or line.startswith("+++ ")):
            continue
        rest = line[4:].strip()
        if "\t" in rest:
            rest = rest.split("\t", 1)[0].strip()
        low = rest.lower()
        if not rest or low in ("/dev/null", "nul", "//dev/null"):
            continue
        s = rest.replace("\\", "/")
        for pref in ("a/", "b/"):
            if s.startswith(pref):
                s = s[len(pref) :]
                break
        if not s.strip():
            continue
        _, err = _safe_resolve_under_root(root, s)
        if err:
            return err
    return None


class WriteFileTool(BaseTool):
    """Overwrite or create a UTF-8 text file under the workspace root (atomic write)."""

    name = "write_file_tool"
    description = (
        "将 content 原子写入工作区内 path（UTF-8）。可选 create_only=true 则仅当文件不存在时写入；"
        "若文件已存在可传 expected_content_hash（与 read_file_tool 返回的 content_sha256 一致）以避免覆盖他人变更。"
    )

    def __init__(self, workspace_root: str | Path, logger_name: str = "codeinsight.tools.write_file") -> None:
        self._root = Path(workspace_root)
        self.logger = get_logger(logger_name)

    def run(self, input: dict[str, Any] | str) -> dict[str, Any]:
        args = input if isinstance(input, dict) else {}
        path_raw = args.get("path")
        content = args.get("content")
        if not isinstance(path_raw, str) or not path_raw.strip():
            return make_tool_result(status="error", data=None, error="write_file_tool 需要非空 path", meta={})
        if not isinstance(content, str):
            return make_tool_result(status="error", data=None, error="write_file_tool 需要字符串 content", meta={})

        create_only = bool(args.get("create_only", False))
        expected_hash = args.get("expected_content_hash")
        if expected_hash is not None and not isinstance(expected_hash, str):
            return make_tool_result(status="error", data=None, error="expected_content_hash 须为字符串", meta={})

        target, err = _safe_resolve_under_root(self._root, path_raw)
        if err or target is None:
            return make_tool_result(status="error", data=None, error=err or "无效路径", meta={})

        if target.exists() and target.is_dir():
            return make_tool_result(status="error", data=None, error="目标为目录，不能写入", meta={})

        if target.exists() and target.is_file():
            if create_only:
                return make_tool_result(
                    status="error",
                    data=None,
                    error="文件已存在且 create_only=true，拒绝写入",
                    meta={"path": _relative_posix_from_root(self._root, target)},
                )
            if expected_hash and expected_hash.strip():
                try:
                    current_bytes = target.read_bytes()
                except OSError as exc:
                    return make_tool_result(status="error", data=None, error=f"读取现有文件失败：{exc}", meta={})
                current_hex = hashlib.sha256(current_bytes).hexdigest()
                want = expected_hash.strip().lower()
                if current_hex != want:
                    return make_tool_result(
                        status="error",
                        data=None,
                        error=(
                            "expected_content_hash 与磁盘文件不一致，拒绝写入（请重新 read_file 并传最新 content_sha256）"
                        ),
                        meta={"current_content_sha256": current_hex, "expected": want},
                    )

        try:
            _atomic_write_utf8(target, content)
        except OSError as exc:
            return make_tool_result(status="error", data=None, error=f"写入失败：{exc}", meta={})

        rel = _relative_posix_from_root(self._root, target)
        new_hash = _sha256_utf8(content)
        self.logger.info("write_file_tool wrote path=%s bytes=%s", rel, len(content.encode("utf-8")))
        return make_tool_result(
            status="ok",
            data=f"wrote {_relative_posix_from_root(self._root, target)} ({len(content)} chars)",
            error="",
            meta={"relative_path": rel, "content_sha256": new_hash},
        )


class ApplyPatchTool(BaseTool):
    """Apply a unified diff under the workspace root (patch-ng; paths must stay under root)."""

    name = "apply_patch_tool"
    description = (
        "对工作区应用 unified diff（UTF-8 文本）。路径须在工作区内；支持 git 风格 a/ b/ 前缀。"
        "若失败请检查补丁上下文与文件是否与磁盘一致。"
    )

    def __init__(self, workspace_root: str | Path, logger_name: str = "codeinsight.tools.apply_patch") -> None:
        self._root = Path(workspace_root)
        self.logger = get_logger(logger_name)

    def run(self, input: dict[str, Any] | str) -> dict[str, Any]:
        args = input if isinstance(input, dict) else {}
        patch_text = args.get("patch")
        if not isinstance(patch_text, str) or not patch_text.strip():
            return make_tool_result(status="error", data=None, error="apply_patch_tool 需要非空字符串 patch", meta={})

        strip = args.get("strip")
        if strip is None:
            strip_candidates = (1, 0)
        else:
            if not isinstance(strip, int) or strip not in (0, 1):
                return make_tool_result(status="error", data=None, error="strip 仅能为 0 或 1", meta={})
            strip_candidates = (strip,)

        try:
            raw = patch_text.encode("utf-8")
        except UnicodeEncodeError as exc:
            return make_tool_result(status="error", data=None, error=f"补丁编码失败：{exc}", meta={})

        hdr_err = _validate_patch_header_paths(patch_text, self._root)
        if hdr_err:
            return make_tool_result(
                status="error",
                data=None,
                error=hdr_err,
                meta={"invalid_patch_paths": True},
            )

        try:
            ps = PatchSet(BytesIO(raw))
        except Exception as exc:  # noqa: BLE001
            return make_tool_result(status="error", data=None, error=f"解析补丁失败：{exc}", meta={})

        if getattr(ps, "errors", 0):
            return make_tool_result(status="error", data=None, error="补丁文本解析存在错误，请检查 unified diff 格式", meta={})

        v_err = _validate_patch_stays_in_workspace(ps, self._root)
        if v_err:
            return make_tool_result(status="error", data=None, error=v_err, meta={"invalid_patch_paths": True})

        root_s = str(self._root.resolve())
        for st in strip_candidates:
            ps2 = PatchSet(BytesIO(raw))
            if ps2.apply(root=root_s, strip=st, fuzz=False):
                self.logger.info("apply_patch_tool success strip=%s", st)
                return make_tool_result(
                    status="ok",
                    data=f"patch applied (strip={st})",
                    error="",
                    meta={"strip": st},
                )

        return make_tool_result(
            status="error",
            data=None,
            error="补丁未能应用到工作区（上下文不匹配或路径不对）；请 read_file 核对后重生成 diff",
            meta={"apply_failed": True},
        )
