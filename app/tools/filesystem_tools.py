from __future__ import annotations

from collections import deque
import fnmatch
import hashlib
import re
from pathlib import Path
from typing import Any

from app.tools.base_tool import BaseTool, make_tool_result
from app.utils.logger import get_logger

DEFAULT_READ_MAX_CHARS = 250_000
ABSOLUTE_READ_MAX_CHARS = 2_000_000
MAX_GREP_OUTPUT_CHARS = 100_000
DEFAULT_GREP_MAX_MATCHES = 300
DEFAULT_LIST_MAX_ENTRIES = 500
DEFAULT_LIST_DEPTH = 2
MAX_LIST_DEPTH = 8
MAX_LIST_ENTRIES_CAP = 5_000


def _safe_resolve_under_root(root: Path, user_path: str) -> tuple[Path | None, str | None]:
    """
    Resolve user_path to an absolute path that stays under root (resolved).

    Returns (path, None) on success, or (None, error_message).
    """
    root_resolved = root.resolve()
    if not isinstance(user_path, str) or not user_path.strip():
        return None, "path 必须为非空字符串"
    raw = user_path.strip()
    candidate = Path(raw)
    try:
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (root_resolved / candidate).resolve()
    except (OSError, RuntimeError) as exc:
        return None, f"路径解析失败：{exc}"
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        return None, "路径越界：必须位于工作区根目录内（禁止路径穿越）"
    return resolved, None


def _relative_posix_from_root(root: Path, path: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return path.name
    if str(rel) in (".", ""):
        return "."
    return rel.as_posix()


class ReadFileTool(BaseTool):
    """Read a text file under the workspace root with optional line range / char cap."""

    name = "read_file_tool"
    description = "读取工作区内文本文件；可选行号范围或 max_chars；超长内容截断并标注 truncated。"

    def __init__(self, workspace_root: str | Path, logger_name: str = "codeinsight.tools.read_file") -> None:
        self._root = Path(workspace_root)
        self.logger = get_logger(logger_name)

    def run(self, input: dict[str, Any] | str) -> dict[str, Any]:
        args = input if isinstance(input, dict) else {}
        path_raw = args.get("path")
        if not isinstance(path_raw, str):
            return make_tool_result(status="error", data=None, error="read_file_tool 需要字符串参数 path", meta={})

        target, err = _safe_resolve_under_root(self._root, path_raw)
        if err or target is None:
            self.logger.warning("read_file_tool rejected path=%s err=%s", path_raw, err)
            return make_tool_result(status="error", data=None, error=err or "无效路径", meta={})

        if not target.is_file():
            return make_tool_result(
                status="error",
                data=None,
                error=f"不是文件或不存在：{_relative_posix_from_root(self._root, target)}",
                meta={},
            )

        start_line = args.get("start_line")
        end_line = args.get("end_line")
        max_chars = args.get("max_chars")

        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return make_tool_result(status="error", data=None, error=f"读取失败：{exc}", meta={})

        if start_line is not None or end_line is not None:
            lines = text.splitlines()
            s_idx = 1
            if start_line is not None:
                if not isinstance(start_line, int) or start_line < 1:
                    return make_tool_result(status="error", data=None, error="start_line 须为 >=1 的整数", meta={})
                s_idx = start_line
            e_idx = len(lines)
            if end_line is not None:
                if not isinstance(end_line, int) or end_line < 1:
                    return make_tool_result(status="error", data=None, error="end_line 须为 >=1 的整数", meta={})
                e_idx = end_line
            if e_idx < s_idx:
                return make_tool_result(status="error", data=None, error="end_line 不能小于 start_line", meta={})
            chunk = "\n".join(lines[s_idx - 1 : e_idx])
        else:
            chunk = text

        cap = DEFAULT_READ_MAX_CHARS
        if max_chars is not None:
            if not isinstance(max_chars, int) or max_chars < 1:
                return make_tool_result(status="error", data=None, error="max_chars 须为 >=1 的整数", meta={})
            cap = min(max_chars, ABSOLUTE_READ_MAX_CHARS)
        else:
            cap = min(cap, ABSOLUTE_READ_MAX_CHARS)

        truncated = False
        if len(chunk) > cap:
            chunk = chunk[:cap]
            truncated = True

        rel = _relative_posix_from_root(self._root, target)
        suffix = "\n\n[truncated: 超出 max_chars 或默认长度上限]" if truncated else ""
        out = f"file={rel}\n{chunk}{suffix}"
        content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.logger.info("read_file_tool path=%s chars=%s truncated=%s", rel, len(chunk), truncated)
        return make_tool_result(
            status="ok",
            data=out,
            error="",
            meta={
                "relative_path": rel,
                "truncated": truncated,
                "returned_chars": len(chunk),
                "content_sha256": content_sha256,
            },
        )


class ListDirTool(BaseTool):
    """List files and directories under a path within the workspace (depth-limited)."""

    name = "list_dir_tool"
    description = "列出工作区目录下条目（相对工作区根的路径）；可选 depth、max_entries。"

    def __init__(self, workspace_root: str | Path, logger_name: str = "codeinsight.tools.list_dir") -> None:
        self._root = Path(workspace_root)
        self.logger = get_logger(logger_name)

    def run(self, input: dict[str, Any] | str) -> dict[str, Any]:
        args = input if isinstance(input, dict) else {}
        path_raw = args.get("path")
        if not isinstance(path_raw, str):
            return make_tool_result(status="error", data=None, error="list_dir_tool 需要字符串参数 path", meta={})

        depth = args.get("depth", DEFAULT_LIST_DEPTH)
        max_entries = args.get("max_entries", DEFAULT_LIST_MAX_ENTRIES)
        if not isinstance(depth, int) or depth < 1 or depth > MAX_LIST_DEPTH:
            return make_tool_result(
                status="error",
                data=None,
                error=f"depth 须为 1~{MAX_LIST_DEPTH} 的整数",
                meta={},
            )
        if not isinstance(max_entries, int) or max_entries < 1 or max_entries > MAX_LIST_ENTRIES_CAP:
            return make_tool_result(
                status="error",
                data=None,
                error=f"max_entries 须为 1~{MAX_LIST_ENTRIES_CAP} 的整数",
                meta={},
            )

        target, err = _safe_resolve_under_root(self._root, path_raw)
        if err or target is None:
            return make_tool_result(status="error", data=None, error=err or "无效路径", meta={})

        if not target.is_dir():
            return make_tool_result(
                status="error",
                data=None,
                error=f"不是目录：{_relative_posix_from_root(self._root, target)}",
                meta={},
            )

        results: list[str] = []
        q: deque[tuple[Path, int]] = deque([(target, 0)])
        truncated_entries = False

        while q and len(results) < max_entries:
            dir_path, d = q.popleft()
            try:
                children = sorted(dir_path.iterdir(), key=lambda p: p.name.lower())
            except OSError as exc:
                return make_tool_result(status="error", data=None, error=f"列出目录失败：{exc}", meta={})

            for child in children:
                if len(results) >= max_entries:
                    truncated_entries = True
                    break
                results.append(_relative_posix_from_root(self._root, child))
                if child.is_dir() and d + 1 < depth:
                    q.append((child, d + 1))
            if truncated_entries:
                break

        note = "\n[truncated: max_entries]" if truncated_entries else ""
        self.logger.info("list_dir_tool entries=%s truncated=%s", len(results), truncated_entries)
        return make_tool_result(
            status="ok",
            data={"paths": results, "note": note.strip() or None},
            error="",
            meta={"count": len(results), "truncated": truncated_entries},
        )


class GrepTool(BaseTool):
    """Regex search in a file or recursively under a directory (within workspace)."""

    name = "grep_tool"
    description = "在工作区内对文件或目录做正则匹配；可选 glob 过滤文件名、max_matches；输出 file:line:content。"

    def __init__(self, workspace_root: str | Path, logger_name: str = "codeinsight.tools.grep") -> None:
        self._root = Path(workspace_root)
        self.logger = get_logger(logger_name)

    def run(self, input: dict[str, Any] | str) -> dict[str, Any]:
        args = input if isinstance(input, dict) else {}
        pattern_raw = args.get("pattern")
        path_raw = args.get("path")
        if not isinstance(pattern_raw, str) or not pattern_raw.strip():
            return make_tool_result(status="error", data=None, error="grep_tool 需要非空字符串 pattern（正则）", meta={})
        if not isinstance(path_raw, str):
            return make_tool_result(status="error", data=None, error="grep_tool 需要字符串参数 path", meta={})

        glob_pat = args.get("glob")
        if glob_pat is not None and not isinstance(glob_pat, str):
            return make_tool_result(status="error", data=None, error="glob 须为字符串", meta={})

        max_matches = args.get("max_matches", DEFAULT_GREP_MAX_MATCHES)
        if not isinstance(max_matches, int) or max_matches < 1 or max_matches > 5_000:
            return make_tool_result(status="error", data=None, error="max_matches 须为 1~5000 的整数", meta={})

        try:
            compiled = re.compile(pattern_raw)
        except re.error as exc:
            return make_tool_result(status="error", data=None, error=f"正则无效：{exc}", meta={})

        target, err = _safe_resolve_under_root(self._root, path_raw)
        if err or target is None:
            return make_tool_result(status="error", data=None, error=err or "无效路径", meta={})

        lines_out: list[str] = []
        total_chars = 0
        match_count = 0
        output_truncated = False

        def append_line(rel: str, line_no: int, line: str) -> bool:
            nonlocal total_chars, match_count, output_truncated
            if match_count >= max_matches:
                return False
            piece = f"{rel}:{line_no}:{line.rstrip('\n')}\n"
            if total_chars + len(piece) > MAX_GREP_OUTPUT_CHARS:
                output_truncated = True
                return False
            lines_out.append(piece)
            total_chars += len(piece)
            match_count += 1
            return True

        def name_ok(name: str) -> bool:
            if not glob_pat or not glob_pat.strip():
                return True
            return fnmatch.fnmatch(name, glob_pat.strip())

        try:

            def scan_file(file_path: Path) -> None:
                nonlocal match_count, output_truncated
                if match_count >= max_matches or output_truncated:
                    return
                if not name_ok(file_path.name):
                    return
                rel = _relative_posix_from_root(self._root, file_path)
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    return
                for i, line in enumerate(content.splitlines(), start=1):
                    if match_count >= max_matches:
                        break
                    if output_truncated:
                        break
                    if compiled.search(line):
                        if not append_line(rel, i, line):
                            break

            if target.is_file():
                scan_file(target)
            elif target.is_dir():
                for p in sorted(target.rglob("*")):
                    if match_count >= max_matches or output_truncated:
                        break
                    if not p.is_file():
                        continue
                    try:
                        p.resolve().relative_to(self._root.resolve())
                    except ValueError:
                        continue
                    scan_file(p)
            else:
                return make_tool_result(
                    status="error",
                    data=None,
                    error=f"路径不存在或类型不支持：{_relative_posix_from_root(self._root, target)}",
                    meta={},
                )
        except OSError as exc:
            return make_tool_result(status="error", data=None, error=f"扫描失败：{exc}", meta={})

        text = "".join(lines_out)
        if output_truncated:
            text += f"\n[truncated: 输出超过 {MAX_GREP_OUTPUT_CHARS} 字符上限]"
        if match_count >= max_matches:
            text += f"\n[truncated: 已达到 max_matches={max_matches}]"

        self.logger.info("grep_tool matches=%s out_chars=%s", match_count, total_chars)
        return make_tool_result(
            status="ok",
            data=text if text.strip() else "(no matches)",
            error="",
            meta={"match_count": match_count, "output_truncated": output_truncated},
        )
