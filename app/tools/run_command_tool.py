from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

from app.sandbox.runner import run_workspace_command
from app.tools.base_tool import BaseTool, make_tool_result
from app.utils.logger import get_logger

# 前缀匹配（规范化后）；额外参数允许
ALLOWLIST_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("pytest",),
    ("python", "-m", "pytest"),
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
    norm = normalize_argv_for_allowlist(argv)
    head = tuple(norm)
    for prefix in ALLOWLIST_PREFIXES:
        pl = len(prefix)
        if len(head) >= pl and head[:pl] == prefix:
            return True
    return False


def argv_has_shell_metacharacters(argv: list[str]) -> bool:
    return any(_SHELL_METACHAR_RE.search(a) is not None for a in argv)


def validate_run_command_arguments(arguments: Any) -> str | None:
    """Semantic validation (used by agentic layer after JSON Schema)."""
    if not isinstance(arguments, dict):
        return "arguments 必须是对象"
    has_argv = "argv" in arguments and arguments.get("argv") is not None
    has_cmd = "command" in arguments and arguments.get("command") is not None
    if has_argv and has_cmd:
        return "不能同时提供 argv 与 command"
    if not has_argv and not has_cmd:
        return "必须提供 argv（字符串数组）或 command（单字符串）之一"
    if has_argv:
        av = arguments.get("argv")
        if not isinstance(av, list) or len(av) < 1:
            return "argv 必须为非空数组"
        if not all(isinstance(x, str) for x in av):
            return "argv 中每一项必须为字符串"
    else:
        cmd = arguments.get("command")
        if not isinstance(cmd, str) or not cmd.strip():
            return "command 必须为非空字符串"
    ts = arguments.get("timeout_seconds")
    if ts is not None and (not isinstance(ts, (int, float)) or ts < 1 or ts > 600):
        return "timeout_seconds 须为 1~600 的数字"
    return None


class RunCommandTool(BaseTool):
    """Run allowlisted subprocess commands under workspace (shell=False). Agentic + AGENT_ALLOW_SHELL only."""

    name = "run_command_tool"
    description = (
        "在工作区根目录执行白名单命令（无 shell）：pytest、python -m pytest、git status、git diff。"
        "提供 argv 数组，或单个 command 字符串（按 shlex 拆分，非 shell）。禁止管道与重定向等特殊字符。"
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
        err = validate_run_command_arguments(args)
        if err:
            return make_tool_result(status="error", data=None, error=err, meta={})

        if args.get("argv") is not None:
            argv = list(args["argv"])
            assert all(isinstance(x, str) for x in argv)
        else:
            cmd = str(args["command"]).strip()
            try:
                argv = shlex.split(cmd, posix=True)
            except ValueError as exc:
                return make_tool_result(status="error", data=None, error=f"command 解析失败：{exc}", meta={})

        if not argv:
            return make_tool_result(status="error", data=None, error="解析后 argv 为空", meta={})

        if argv_has_shell_metacharacters(argv):
            return make_tool_result(
                status="error",
                data=None,
                error="参数中不允许包含 shell 元字符（如 |、>、;、&、$、反引号、换行等）",
                meta={},
            )

        if not argv_matches_allowlist(argv):
            return make_tool_result(
                status="error",
                data=None,
                error="命令不在白名单内。允许的前缀：pytest；python -m pytest；git status；git diff",
                meta={"allowlist_rejected": True},
            )

        timeout = args.get("timeout_seconds")
        if timeout is None:
            to = self._default_timeout
        else:
            to = float(timeout)

        result = run_workspace_command(
            argv,
            cwd=self._root,
            timeout_seconds=to,
            max_output_chars=self._max_output_chars,
            logger=self.logger,
        )

        payload = {
            "returncode": result["returncode"],
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "timed_out": result["timed_out"],
            "argv": argv,
        }
        text = json.dumps(payload, ensure_ascii=False)

        if result["timed_out"]:
            return make_tool_result(
                status="error",
                data=text,
                error=f"命令执行超时（>{to}s），已终止子进程。",
                meta={"timed_out": True, "returncode": result["returncode"], "argv": argv},
            )

        self.logger.info("run_command_tool finished argv=%s returncode=%s", argv, result["returncode"])
        return make_tool_result(
            status="ok",
            data=text,
            error="",
            meta={"returncode": result["returncode"], "argv": argv, "timed_out": False},
        )
