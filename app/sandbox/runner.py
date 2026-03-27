from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import sys

from app.utils.logger import get_logger


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
    - Uses subprocess timeout to prevent dead loops
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
        """
        Write test code to outputs/ and execute it via pytest in subprocess.
        """
        test_file = self._write_test_file(test_code=test_code, filename_prefix=filename_prefix)
        command = [sys.executable, "-m", "pytest", str(test_file), "-q"]
        self.logger.info("Running sandbox command: %s", " ".join(command))

        try:
            completed = subprocess.run(
                command,
                cwd=str(self.outputs_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
                env=self._minimal_env(),
            )
            success = completed.returncode == 0
            self.logger.info("Sandbox finished with return code: %d", completed.returncode)
            return ExecutionResult(
                success=success,
                return_code=completed.returncode,
                stdout=self._truncate_output(completed.stdout),
                stderr=self._truncate_output(completed.stderr),
                timed_out=False,
                command=command,
                file_path=str(test_file),
            )
        except subprocess.TimeoutExpired as exc:
            self.logger.warning("Sandbox execution timed out after %d seconds.", self.timeout_seconds)
            return ExecutionResult(
                success=False,
                return_code=-1,
                stdout=self._truncate_output(exc.stdout or ""),
                stderr=self._truncate_output(exc.stderr or ""),
                timed_out=True,
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

        # Ensure file path stays inside outputs directory.
        try:
            test_file.relative_to(self.outputs_dir)
        except ValueError:
            raise ValueError("Refusing to write test file outside outputs directory.")

        test_file.write_text(test_code, encoding="utf-8")
        self.logger.info("Wrote sandbox test file: %s", test_file)
        return test_file

    def _minimal_env(self) -> dict[str, str]:
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
        return env

    def _truncate_output(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        keep = max(self.max_output_chars - 80, 0)
        truncated = text[:keep]
        return f"{truncated}\n... [truncated output: original_len={len(text)}]"
