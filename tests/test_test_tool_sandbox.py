from __future__ import annotations

import json
from pathlib import Path

from app.llm.llm import LLMClient
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
    data = json.loads(out)
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
    data = json.loads(out)
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
    data = json.loads(out)
    assert data["sandbox"]["skipped"] is True
