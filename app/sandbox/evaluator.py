from __future__ import annotations

from app.sandbox.runner import ExecutionResult


def evaluate_execution(result: ExecutionResult) -> dict[str, str | bool | int]:
    """Convert raw execution result into a compact evaluation summary."""
    if result.timed_out:
        verdict = "timeout"
    elif result.success:
        verdict = "passed"
    else:
        verdict = "failed"

    return {
        "verdict": verdict,
        "success": result.success,
        "timed_out": result.timed_out,
        "return_code": result.return_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "file_path": result.file_path,
    }
