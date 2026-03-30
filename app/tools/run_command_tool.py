from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

from app.sandbox.runner import run_workspace_command
from app.tools.base_tool import BaseTool, make_tool_result
from app.utils.logger import get_logger

ALLOWLIST_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("git", "status"),
    ("git", "diff"),
)

_SHELL_METACHAR_RE = re.compile(r"[|&;<>$\n\r`]")


def normalize_argv_for_allowlist(argv: list[str]) -> list[str]:
    if not argv:
        return argv
    first_name = Path(argv[0]).name.lower()
    base = first_name.removesuffix(".exe")
    if base in ("python", "python3"):
        return ["python", *argv[1:]]
    if base == "pytest":
        return ["pytest", *argv[1:]]
    return list(argv)


def argv_matches_allowlist(argv: list[str]) -> bool:
    normalized = tuple(normalize_argv_for_allowlist(argv))
    return any(len(normalized) >= len(prefix) and normalized[: len(prefix)] == prefix for prefix in ALLOWLIST_PREFIXES)


def argv_has_shell_metacharacters(argv: list[str]) -> bool:
    return any(_SHELL_METACHAR_RE.search(arg) is not None for arg in argv)


def validate_run_command_arguments(arguments: Any) -> str | None:
    if not isinstance(arguments, dict):
        return "arguments must be an object"

    has_argv = "argv" in arguments and arguments.get("argv") is not None
    has_command = "command" in arguments and arguments.get("command") is not None
    if has_argv and has_command:
        return "provide either argv or command, not both"
    if not has_argv and not has_command:
        return "argv or command is required"

    if has_argv:
        argv = arguments.get("argv")
        if not isinstance(argv, list) or not argv:
            return "argv must be a non-empty list"
        if not all(isinstance(item, str) for item in argv):
            return "argv items must be strings"
    else:
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return "command must be a non-empty string"

    timeout = arguments.get("timeout_seconds")
    if timeout is not None and (not isinstance(timeout, (int, float)) or timeout < 1 or timeout > 600):
        return "timeout_seconds must be between 1 and 600"
    return None


class RunCommandTool(BaseTool):
    """Run low-risk allowlisted subprocess commands under the workspace."""

    name = "run_command_tool"
    description = (
        "Execute a low-risk allowlisted command without shell expansion inside the workspace. "
        "Allowed prefixes: git status, git diff."
    )

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        default_timeout_seconds: float = 120.0,
        max_output_chars: int = 8000,
        logger_name: str = "codeinsight.tools.run_command",
    ) -> None:
        self._root = Path(workspace_root).resolve()
        self._default_timeout = float(default_timeout_seconds)
        self._max_output_chars = max_output_chars
        self.logger = get_logger(logger_name)

    def run(self, input: dict[str, Any] | str) -> dict[str, Any]:
        args = input if isinstance(input, dict) else {}
        error = validate_run_command_arguments(args)
        if error:
            return make_tool_result(status="error", data=None, error=error, meta={})

        if args.get("argv") is not None:
            argv = list(args["argv"])
        else:
            command = str(args["command"]).strip()
            try:
                argv = shlex.split(command, posix=True)
            except ValueError as exc:
                return make_tool_result(status="error", data=None, error=f"failed to parse command: {exc}", meta={})

        if not argv:
            return make_tool_result(status="error", data=None, error="parsed argv is empty", meta={})

        if argv_has_shell_metacharacters(argv):
            return make_tool_result(
                status="error",
                data=None,
                error="shell metacharacters are not allowed in argv",
                meta={},
            )

        if not argv_matches_allowlist(argv):
            return make_tool_result(
                status="error",
                data=None,
                error="command is not allowlisted; only `git status` and `git diff` are permitted",
                meta={"allowlist_rejected": True},
            )

        timeout_seconds = float(args.get("timeout_seconds") or self._default_timeout)
        result = run_workspace_command(
            argv,
            cwd=self._root,
            timeout_seconds=timeout_seconds,
            max_output_chars=self._max_output_chars,
            logger=self.logger,
        )
        payload = {
            "returncode": result["returncode"],
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "timed_out": result["timed_out"],
            "cancelled": bool(result.get("cancelled")),
            "argv": argv,
        }
        text = json.dumps(payload, ensure_ascii=False)

        if result["timed_out"]:
            return make_tool_result(
                status="error",
                data=text,
                error=f"command timed out after {timeout_seconds}s and was terminated",
                meta={"timed_out": True, "returncode": result["returncode"], "argv": argv},
            )

        if result.get("cancelled"):
            return make_tool_result(
                status="error",
                data=text,
                error="command execution was cancelled",
                meta={"cancelled": True, "returncode": result["returncode"], "argv": argv},
            )

        self.logger.info("run_command_tool finished argv=%s returncode=%s", argv, result["returncode"])
        return make_tool_result(
            status="ok",
            data=text,
            error="",
            meta={"returncode": result["returncode"], "argv": argv, "timed_out": False},
        )
