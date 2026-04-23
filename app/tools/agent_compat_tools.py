from __future__ import annotations

import re
from typing import Any

from app.tools.base_tool import BaseTool, ensure_tool_result, make_tool_result


class CodeSearchTool(BaseTool):
    """Compatibility alias for semantic code search."""

    name = "code_search"
    description = "Alias of search_tool. Semantic code search by query/input."

    def __init__(self, delegate: BaseTool) -> None:
        self._delegate = delegate

    def run(self, input: dict[str, Any] | str) -> dict[str, Any] | str:
        return self._delegate.run(input)


class OpenFileTool(BaseTool):
    """Compatibility alias for reading a workspace file."""

    name = "open_file"
    description = "Alias of read_file_tool. Open/read file content under workspace root."

    def __init__(self, delegate: BaseTool) -> None:
        self._delegate = delegate

    def run(self, input: dict[str, Any] | str) -> dict[str, Any] | str:
        return self._delegate.run(input)


class FindSymbolTool(BaseTool):
    """Find likely symbol definitions/usages by delegating to grep_tool."""

    name = "find_symbol"
    description = (
        "Find symbol definitions/usages in codebase. "
        "Args: symbol(required), path(optional), glob(optional), max_matches(optional)."
    )

    def __init__(self, grep_delegate: BaseTool) -> None:
        self._grep = grep_delegate

    def run(self, input: dict[str, Any] | str) -> dict[str, Any]:
        if isinstance(input, str):
            args: dict[str, Any] = {"symbol": input}
        elif isinstance(input, dict):
            args = dict(input)
        else:
            args = {}

        symbol = args.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            return make_tool_result(status="error", data=None, error="find_symbol requires non-empty symbol", meta={})
        symbol = symbol.strip()

        path = args.get("path", ".")
        if not isinstance(path, str) or not path.strip():
            return make_tool_result(status="error", data=None, error="find_symbol.path must be a non-empty string", meta={})

        pattern = _build_symbol_pattern(symbol)
        grep_args: dict[str, Any] = {
            "pattern": pattern,
            "path": path,
        }

        if "glob" in args and args.get("glob") is not None:
            grep_args["glob"] = args.get("glob")
        if "max_matches" in args and args.get("max_matches") is not None:
            grep_args["max_matches"] = args.get("max_matches")

        raw = self._grep.run(grep_args)
        normalized = ensure_tool_result(raw)
        meta = dict(normalized.get("meta") or {})
        meta.update({"symbol": symbol, "pattern": pattern})
        return make_tool_result(
            status=normalized["status"],
            data=normalized.get("data"),
            error=str(normalized.get("error") or ""),
            meta=meta,
        )


class RunCodeTool(BaseTool):
    """Compatibility alias for shell command execution."""

    name = "run_code"
    description = "Alias of run_command_tool. Execute allowlisted command via argv/command."

    def __init__(self, delegate: BaseTool) -> None:
        self._delegate = delegate

    def run(self, input: dict[str, Any] | str) -> dict[str, Any] | str:
        if isinstance(input, str):
            return self._delegate.run({"command": input})
        return self._delegate.run(input)


class RunTestsTool(BaseTool):
    """Run tests through run_command_tool with a default command."""

    name = "run_tests"
    description = "Run tests in workspace. Default command is pytest -q when not provided."

    def __init__(self, delegate: BaseTool, *, default_command: str = "pytest -q") -> None:
        self._delegate = delegate
        self._default_command = str(default_command or "").strip() or "pytest -q"

    def run(self, input: dict[str, Any] | str) -> dict[str, Any] | str:
        if isinstance(input, str):
            args: dict[str, Any] = {"command": input.strip()} if input.strip() else {}
        elif isinstance(input, dict):
            args = dict(input)
        else:
            args = {}

        if args.get("command") is not None and not isinstance(args.get("command"), str):
            return make_tool_result(status="error", data=None, error="run_tests.command must be a string", meta={})
        if args.get("command") is not None and isinstance(args.get("command"), str) and not args.get("command").strip():
            args.pop("command", None)

        has_argv = isinstance(args.get("argv"), list) and len(args.get("argv")) > 0
        has_command = isinstance(args.get("command"), str) and bool(args.get("command").strip())
        if not has_argv and not has_command:
            args["command"] = self._default_command
        return self._delegate.run(args)


def _build_symbol_pattern(symbol: str) -> str:
    esc = re.escape(symbol)
    # Keep a permissive fallback (word match), while preferring declaration-like patterns.
    patterns = [
        rf"^\s*async\s+def\s+{esc}\b",
        rf"^\s*def\s+{esc}\b",
        rf"^\s*class\s+{esc}\b",
        rf"^\s*(export\s+)?(async\s+)?function\s+{esc}\b",
        rf"^\s*(export\s+)?(const|let|var)\s+{esc}\s*=",
        rf"^\s*fn\s+{esc}\b",
        rf"^\s*func\s+(?:\([^)]*\)\s*)?{esc}\b",
        rf"\b{esc}\b",
    ]
    return "(?:" + "|".join(patterns) + ")"
