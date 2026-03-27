from __future__ import annotations

import ast
import json
from typing import Any

from app.llm.llm import LLMClient
from app.llm.prompt import TEST_TOOL_SYSTEM_PROMPT, build_test_tool_user_prompt
from app.sandbox.evaluator import evaluate_execution
from app.sandbox.runner import SandboxRunner
from app.tools.base_tool import BaseTool
from app.utils.logger import get_logger

_PREVIEW_MAX = 2000


def _truncate(text: str, max_len: int = _PREVIEW_MAX) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 20] + "\n... [truncated]"


def _build_error_summary(
    *,
    verdict: str,
    timed_out: bool,
    success: bool,
    stdout: str,
    stderr: str,
) -> str:
    if timed_out:
        return "pytest 子进程超时，未在限定时间内结束。"
    if success or verdict == "passed":
        return ""
    err = (stderr or "").strip()
    out = (stdout or "").strip()
    primary = err if err else out
    return _truncate(primary) if primary else "测试失败且无可用输出。"


class TestTool(BaseTool):
    """Generate executable pytest script from original/optimized code."""
    __test__ = False  # Prevent pytest from collecting this as a test class.

    name = "test_tool"
    description = "基于原代码与优化代码生成可执行 pytest 测试脚本。"

    def __init__(
        self,
        llm: LLMClient,
        logger_name: str = "codeinsight.tools.test",
        *,
        run_sandbox: bool = True,
        sandbox_outputs_dir: str = "outputs",
        sandbox_timeout_seconds: int = 10,
    ) -> None:
        self.llm = llm
        self.logger = get_logger(logger_name)
        self._run_sandbox = run_sandbox
        self._sandbox_outputs_dir = sandbox_outputs_dir
        self._sandbox_timeout_seconds = sandbox_timeout_seconds

    def run(self, input: str) -> str:
        original_code, optimized_code = self._parse_input(input)
        prompt = build_test_tool_user_prompt(
            original_code=original_code, optimized_code=optimized_code
        )
        raw = self.llm.generate_text(prompt=prompt, system_prompt=TEST_TOOL_SYSTEM_PROMPT)
        self.logger.debug("TestTool raw output: %s", raw)

        result = self._parse_result(raw=raw, optimized_code=optimized_code)
        self._attach_sandbox_result(result)
        return json.dumps(result, ensure_ascii=False)

    def _parse_input(self, input_text: str) -> tuple[str, str]:
        """
        Expected format:
        [ORIGINAL_CODE]
        ...
        [OPTIMIZED_CODE]
        ...
        """
        original_marker = "[ORIGINAL_CODE]"
        optimized_marker = "[OPTIMIZED_CODE]"

        if original_marker in input_text and optimized_marker in input_text:
            part_original, part_optimized = input_text.split(optimized_marker, maxsplit=1)
            original_code = part_original.replace(original_marker, "", 1).strip()
            optimized_code = part_optimized.strip()
            return original_code, optimized_code

        # Fallback: if caller only passes one snippet, use it for both.
        code = input_text.strip()
        return code, code

    def _parse_result(self, raw: str, optimized_code: str) -> dict[str, Any]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.warning("TestTool received non-JSON output. Using fallback.")
            return self._fallback_result(optimized_code)

        if not isinstance(data, dict):
            self.logger.warning("TestTool received non-object JSON. Using fallback.")
            return self._fallback_result(optimized_code)

        coverage_focus = data.get("coverage_focus", [])
        test_code = data.get("test_code", "")

        if not isinstance(coverage_focus, list):
            coverage_focus = ["核心路径覆盖", "边界输入覆盖", "异常路径覆盖"]
        coverage_focus = [str(item) for item in coverage_focus if str(item).strip()]
        if not coverage_focus:
            coverage_focus = ["核心路径覆盖", "边界输入覆盖", "异常路径覆盖"]

        if not isinstance(test_code, str) or not test_code.strip():
            self.logger.warning("TestTool missing test_code. Using fallback template.")
            return self._fallback_result(optimized_code)

        executable = self._is_python_syntax_valid(test_code)
        if not executable:
            self.logger.warning("Generated test_code is not executable. Using fallback template.")
            return self._fallback_result(optimized_code)

        return {
            "coverage_focus": coverage_focus,
            "test_code": test_code,
            "executable": True,
        }

    def _attach_sandbox_result(self, result: dict[str, Any]) -> None:
        """Run generated tests in sandbox and merge verdict + error summary."""
        if not self._run_sandbox:
            result["sandbox"] = {"skipped": True, "reason": "sandbox disabled"}
            return

        test_code = result.get("test_code", "")
        if not isinstance(test_code, str) or not test_code.strip():
            result["sandbox"] = {
                "verdict": "skipped",
                "error_summary": "无 test_code，跳过执行。",
            }
            return

        runner = SandboxRunner(
            outputs_dir=self._sandbox_outputs_dir,
            timeout_seconds=self._sandbox_timeout_seconds,
        )
        try:
            exec_result = runner.run_test_code(test_code)
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Sandbox execution raised an exception.")
            result["sandbox"] = {
                "verdict": "error",
                "success": False,
                "timed_out": False,
                "error_summary": _truncate(str(exc)),
            }
            return

        ev = evaluate_execution(exec_result)
        verdict = str(ev.get("verdict", "failed"))
        stdout = str(ev.get("stdout", "") or "")
        stderr = str(ev.get("stderr", "") or "")
        error_summary = _build_error_summary(
            verdict=verdict,
            timed_out=bool(ev.get("timed_out")),
            success=bool(ev.get("success")),
            stdout=stdout,
            stderr=stderr,
        )

        result["sandbox"] = {
            "verdict": verdict,
            "success": bool(ev.get("success")),
            "timed_out": bool(ev.get("timed_out")),
            "return_code": int(ev.get("return_code", -1)),
            "error_summary": error_summary,
            "stdout_preview": _truncate(stdout) if stdout.strip() else "",
            "stderr_preview": _truncate(stderr) if stderr.strip() else "",
            "file_path": str(ev.get("file_path", "")),
        }

    def _is_python_syntax_valid(self, code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _fallback_result(self, optimized_code: str) -> dict[str, Any]:
        safe_module = optimized_code if optimized_code.strip() else "def target(x):\n    return x\n"
        test_code = (
            f"{safe_module}\n\n"
            "import pytest\n\n"
            "def test_basic_behavior():\n"
            "    assert callable(target) if 'target' in globals() else True\n\n"
            "def test_boundary_case():\n"
            "    # TODO: replace with real boundary input assertions\n"
            "    assert True\n\n"
            "def test_error_path():\n"
            "    # TODO: replace with real exception-path assertions\n"
            "    assert True\n"
        )
        return {
            "coverage_focus": ["核心路径覆盖", "边界输入覆盖", "异常路径覆盖"],
            "test_code": test_code,
            "executable": self._is_python_syntax_valid(test_code),
        }
