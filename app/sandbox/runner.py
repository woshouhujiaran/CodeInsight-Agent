from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import local
import time
from typing import Any

from app.utils.logger import get_logger

_THREAD_STATE = local()


def minimal_subprocess_env() -> dict[str, str]:
    """Minimal environment for subprocess (PATH, Windows dirs, UTF-8)."""
    env: dict[str, str] = {}
    for key in (
        "SYSTEMROOT",
        "WINDIR",
        "PATH",
        "PYTHONPATH",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
    ):
        value = os.environ.get(key)
        if value:
            env[key] = value
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def hardened_pytest_env() -> dict[str, str]:
    env = minimal_subprocess_env()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return env


@contextmanager
def bind_cancellation_event(cancel_event: Any | None):
    previous = getattr(_THREAD_STATE, "cancel_event", None)
    _THREAD_STATE.cancel_event = cancel_event
    try:
        yield
    finally:
        _THREAD_STATE.cancel_event = previous


def current_cancellation_event() -> Any | None:
    return getattr(_THREAD_STATE, "cancel_event", None)


def truncate_process_output(text: str | None, max_chars: int) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    keep = max(max_chars - 80, 0)
    truncated = text[:keep]
    return f"{truncated}\n... [truncated output: original_len={len(text)}]"


def _stop_process(process: subprocess.Popen[str], logger: Any, *, reason: str) -> tuple[str, str]:
    logger.warning("Stopping subprocess pid=%s reason=%s", process.pid, reason)
    process.kill()
    stdout, stderr = process.communicate()
    return stdout or "", stderr or ""


def _run_subprocess(
    argv: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    max_output_chars: int,
    logger: Any,
    env: dict[str, str],
) -> dict[str, Any]:
    cancel_event = current_cancellation_event()
    process = subprocess.Popen(
        argv,
        cwd=str(cwd.resolve()),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    started = time.monotonic()
    poll_interval = 0.2

    while True:
        if cancel_event is not None and cancel_event.is_set():
            stdout, stderr = _stop_process(process, logger, reason="cancelled")
            return {
                "returncode": -1,
                "stdout": truncate_process_output(stdout, max_output_chars),
                "stderr": truncate_process_output(
                    (stderr + "\n[error: command cancelled]").strip(),
                    max_output_chars,
                ),
                "timed_out": False,
                "cancelled": True,
            }

        elapsed = time.monotonic() - started
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            stdout, stderr = _stop_process(process, logger, reason="timeout")
            return {
                "returncode": -1,
                "stdout": truncate_process_output(stdout, max_output_chars),
                "stderr": truncate_process_output(
                    (stderr + f"\n[error: 命令超时：{timeout_seconds}s]").strip(),
                    max_output_chars,
                ),
                "timed_out": True,
                "cancelled": False,
            }

        try:
            stdout, stderr = process.communicate(timeout=min(poll_interval, remaining))
            return {
                "returncode": process.returncode,
                "stdout": truncate_process_output(stdout, max_output_chars),
                "stderr": truncate_process_output(stderr, max_output_chars),
                "timed_out": False,
                "cancelled": False,
            }
        except subprocess.TimeoutExpired:
            continue


def run_workspace_command(
    argv: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    max_output_chars: int = 8000,
    logger: Any | None = None,
) -> dict[str, Any]:
    """
    Run argv with shell=False under cwd. Captures stdout/stderr with timeout and cooperative cancellation.
    Returns dict: returncode, stdout, stderr, timed_out (bool), cancelled (bool).
    """
    log = logger or get_logger("codeinsight.sandbox.workspace_cmd")
    log.info("run_workspace_command cwd=%s argv=%s", cwd, argv)
    return _run_subprocess(
        argv,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        max_output_chars=max_output_chars,
        logger=log,
        env=minimal_subprocess_env(),
    )


@dataclass
class ExecutionResult:
    """Result of sandbox test execution."""

    success: bool
    return_code: int
    stdout: str
    stderr: str
    timed_out: bool
    command: list[str]
    file_path: str


class SandboxRunner:
    """
    Execute generated test code in a subprocess with timeout protection.

    Security baseline:
    - Writes/executes only under `outputs/` directory
    - Uses timeout and cooperative cancellation to stop hanging subprocesses
    - Disables pytest plugin autoload inside the sandbox subprocess
    - Captures stdout/stderr for later evaluation
    """

    def __init__(
        self,
        outputs_dir: str = "outputs",
        timeout_seconds: int = 10,
        max_output_chars: int = 8000,
        logger_name: str = "codeinsight.sandbox.runner",
    ) -> None:
        self.outputs_dir = Path(outputs_dir).resolve()
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.logger = get_logger(logger_name)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def run_test_code(self, test_code: str, filename_prefix: str = "generated_test") -> ExecutionResult:
        test_file = self._write_test_file(test_code=test_code, filename_prefix=filename_prefix)
        command = [sys.executable, "-m", "pytest", str(test_file), "-q"]
        self.logger.info("Running sandbox command: %s", " ".join(command))
        result = _run_subprocess(
            command,
            cwd=self.outputs_dir,
            timeout_seconds=self.timeout_seconds,
            max_output_chars=self.max_output_chars,
            logger=self.logger,
            env=hardened_pytest_env(),
        )
        self.logger.info("Sandbox finished with return code: %d", result["returncode"])
        return ExecutionResult(
            success=(result["returncode"] == 0) and not result["timed_out"] and not result.get("cancelled"),
            return_code=result["returncode"],
            stdout=str(result.get("stdout") or ""),
            stderr=str(result.get("stderr") or ""),
            timed_out=bool(result.get("timed_out")),
            command=command,
            file_path=str(test_file),
        )

    def _write_test_file(self, test_code: str, filename_prefix: str) -> Path:
        safe_prefix = "".join(ch for ch in filename_prefix if ch.isalnum() or ch in ("_", "-")).strip()
        safe_prefix = safe_prefix or "generated_test"
        safe_prefix = re.sub(r"(\.\.|/|\\)", "", safe_prefix)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name = f"{safe_prefix}_{timestamp}.py"
        test_file = (self.outputs_dir / file_name).resolve()

        try:
            test_file.relative_to(self.outputs_dir)
        except ValueError:
            raise ValueError("Refusing to write test file outside outputs directory.")

        test_file.write_text(test_code, encoding="utf-8")
        self.logger.info("Wrote sandbox test file: %s", test_file)
        return test_file
