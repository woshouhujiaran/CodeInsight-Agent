from __future__ import annotations

import json
from pathlib import Path

from app.llm.llm import LLMClient
from app.sandbox.runner import SandboxRunner, hardened_pytest_env
from app.tools.test_tool import TestTool


class FixedLLM(LLMClient):
    """Deterministic LLM for tests: always returns the given JSON string."""

    def __init__(self, planner_payload: str) -> None:
        super().__init__(provider="none", model="fixed-test")
        self._planner_payload = planner_payload

    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        return self._planner_payload


def _tool_with_fixed_response(payload: dict, tmp_path: Path) -> TestTool:
    return TestTool(
        llm=FixedLLM(json.dumps(payload, ensure_ascii=False)),
        sandbox_outputs_dir=str(tmp_path / "sandbox_out"),
        sandbox_timeout_seconds=30,
    )


def test_sandbox_verdict_passed(tmp_path: Path) -> None:
    tool = _tool_with_fixed_response(
        {
            "coverage_focus": ["core"],
            "test_code": "def test_ok():\n    assert 1 == 1\n",
        },
        tmp_path,
    )
    out = tool.run("ignored")
    data = out["data"]
    assert data["sandbox"]["verdict"] == "passed"
    assert data["sandbox"]["success"] is True
    assert data["sandbox"]["timed_out"] is False
    assert data["sandbox"]["error_summary"] == ""


def test_sandbox_verdict_failed(tmp_path: Path) -> None:
    tool = _tool_with_fixed_response(
        {
            "coverage_focus": ["core"],
            "test_code": "def test_fail():\n    assert 1 == 2\n",
        },
        tmp_path,
    )
    out = tool.run("ignored")
    data = out["data"]
    assert data["sandbox"]["verdict"] == "failed"
    assert data["sandbox"]["success"] is False
    assert data["sandbox"]["error_summary"]


def test_sandbox_skipped_when_disabled(tmp_path: Path) -> None:
    tool = TestTool(
        llm=FixedLLM(
            json.dumps(
                {"coverage_focus": ["core"], "test_code": "def test_ok():\n    assert 1 == 1\n"},
                ensure_ascii=False,
            )
        ),
        run_sandbox=False,
    )
    out = tool.run("ignored")
    data = out["data"]
    assert data["sandbox"]["skipped"] is True


def test_sandbox_timeout_result(tmp_path: Path) -> None:
    runner = SandboxRunner(outputs_dir=str(tmp_path / "sandbox_out"), timeout_seconds=1)
    res = runner.run_test_code("import time\n\ndef test_sleep():\n    time.sleep(2)\n")
    assert res.timed_out is True
    assert res.success is False


def test_sandbox_malicious_prefix_stays_in_outputs(tmp_path: Path) -> None:
    runner = SandboxRunner(outputs_dir=str(tmp_path / "sandbox_out"), timeout_seconds=5)
    p = runner._write_test_file("def test_ok():\n    assert True\n", filename_prefix="..\\..\\evil")
    assert str(p.resolve()).startswith(str((tmp_path / "sandbox_out").resolve()))


def test_sandbox_output_truncation(tmp_path: Path) -> None:
    runner = SandboxRunner(outputs_dir=str(tmp_path / "sandbox_out"), timeout_seconds=5, max_output_chars=200)
    code = (
        "def test_output():\n"
        "    import sys\n"
        "    sys.stdout.write('x' * 1000)\n"
        "    assert True\n"
    )
    res = runner.run_test_code(code)
    assert res.success is True
    assert "[truncated output:" in res.stdout or len(res.stdout) <= 200


def test_test_tool_rejects_malicious_generated_code(tmp_path: Path) -> None:
    marker = tmp_path / "owned.txt"
    tool = _tool_with_fixed_response(
        {
            "coverage_focus": ["core"],
            "test_code": (
                "import os\n\n"
                "def test_evil():\n"
                f"    with open(r\"{marker}\", \"w\", encoding=\"utf-8\") as handle:\n"
                "        handle.write('owned')\n"
            ),
        },
        tmp_path,
    )
    out = tool.run("ignored")
    data = out["data"]
    assert out["status"] == "error"
    assert out.get("meta", {}).get("static_review_rejected") is True
    assert data["sandbox"]["verdict"] == "rejected"
    assert marker.exists() is False


def test_hardened_pytest_env_disables_plugin_autoload() -> None:
    env = hardened_pytest_env()
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
