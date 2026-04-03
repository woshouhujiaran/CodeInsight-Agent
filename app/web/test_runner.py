from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from app.sandbox.runner import run_workspace_command
from app.tools.run_command_tool import split_command_string
from app.utils.logger import get_logger


def split_command(command: str) -> list[str]:
    try:
        return split_command_string(command)
    except ValueError as exc:
        message = str(exc)
        if "non-empty" in message or "parsed argv is empty" in message:
            raise ValueError("test_command 不能为空") from exc
        raise ValueError(f"test_command 无法解析：{exc}") from exc


def build_test_summary(
    *,
    command: str,
    returncode: int,
    stdout: str,
    stderr: str,
    duration_ms: int,
    timed_out: bool,
    cancelled: bool,
) -> dict[str, Any]:
    combined = ((stdout or "") + "\n" + (stderr or "")).strip()
    raw_tail = combined[-4000:] if combined else ""
    if cancelled and raw_tail:
        raw_tail = raw_tail + "\n[cancelled]"
    elif cancelled:
        raw_tail = "[cancelled]"
    elif timed_out and raw_tail:
        raw_tail = raw_tail + "\n[timeout]"
    elif timed_out:
        raw_tail = "[timeout]"
    passed = (returncode == 0) and not timed_out and not cancelled
    return {
        "passed": passed,
        "failed": not passed,
        "duration_ms": duration_ms,
        "command": command,
        "raw_tail": raw_tail,
        "returncode": returncode,
        "timed_out": timed_out,
        "cancelled": cancelled,
    }


def run_project_test_command(
    *,
    workspace_root: str,
    command: str,
    allow_shell: bool,
    timeout_seconds: float = 600.0,
    max_output_chars: int = 12000,
) -> dict[str, Any]:
    if not allow_shell:
        raise PermissionError("未开启命令执行权限，无法运行测试命令。")

    argv = split_command(command)
    started = time.perf_counter()
    logger = get_logger("codeinsight.web.test_runner")
    result = run_workspace_command(
        argv,
        cwd=Path(workspace_root).resolve(),
        timeout_seconds=timeout_seconds,
        max_output_chars=max_output_chars,
        logger=logger,
    )
    duration_ms = int((time.perf_counter() - started) * 1000)
    return build_test_summary(
        command=" ".join(argv),
        returncode=int(result.get("returncode", -1)),
        stdout=str(result.get("stdout") or ""),
        stderr=str(result.get("stderr") or ""),
        duration_ms=duration_ms,
        timed_out=bool(result.get("timed_out")),
        cancelled=bool(result.get("cancelled")),
    )
