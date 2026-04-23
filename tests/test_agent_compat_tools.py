from __future__ import annotations

from pathlib import Path
from typing import Any

from app.agent.tool_specs import get_canonical_parameter_schema, validate_agentic_tool_call
from app.agent.tool_registry import ToolRegistry
from app.tools.agent_compat_tools import CodeSearchTool, FindSymbolTool, OpenFileTool, RunCodeTool, RunTestsTool
from app.tools.base_tool import BaseTool, make_tool_result
from app.tools.filesystem_tools import GrepTool, ReadFileTool


class _EchoDelegate(BaseTool):
    name = "echo_delegate"
    description = "echo"

    def __init__(self) -> None:
        self.last_input: Any = None

    def run(self, input: dict[str, Any] | str) -> dict[str, Any]:
        self.last_input = input
        return make_tool_result(status="ok", data={"echo": input}, error="", meta={})


def _ws(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


def test_code_search_and_open_file_delegate_calls(tmp_path: Path) -> None:
    delegate = _EchoDelegate()
    code_search = CodeSearchTool(delegate=delegate)
    open_file = OpenFileTool(delegate=delegate)

    r1 = code_search.run({"query": "auth login"})
    assert r1["status"] == "ok"
    assert delegate.last_input == {"query": "auth login"}

    r2 = open_file.run({"path": "app/runtime.py"})
    assert r2["status"] == "ok"
    assert delegate.last_input == {"path": "app/runtime.py"}


def test_find_symbol_uses_grep_delegate(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "mod.py").write_text("def hello_world():\n    return 1\n", encoding="utf-8")
    grep_tool = GrepTool(workspace_root=ws)
    tool = FindSymbolTool(grep_delegate=grep_tool)

    result = tool.run({"symbol": "hello_world", "path": ".", "glob": "*.py"})

    assert result["status"] == "ok"
    text = str(result.get("data") or "").replace("\\", "/")
    assert "mod.py" in text
    assert "hello_world" in text
    assert result.get("meta", {}).get("symbol") == "hello_world"


def test_run_code_and_run_tests_delegate_and_defaults() -> None:
    delegate = _EchoDelegate()
    run_code = RunCodeTool(delegate=delegate)
    run_tests = RunTestsTool(delegate=delegate, default_command="pytest -q")

    result_code = run_code.run("git status -s")
    assert result_code["status"] == "ok"
    assert delegate.last_input == {"command": "git status -s"}

    result_tests = run_tests.run({})
    assert result_tests["status"] == "ok"
    assert delegate.last_input == {"command": "pytest -q"}


def test_new_tool_schemas_and_validation_are_registered(tmp_path: Path) -> None:
    assert get_canonical_parameter_schema("code_search") is not None
    assert get_canonical_parameter_schema("open_file") is not None
    assert get_canonical_parameter_schema("find_symbol") is not None
    assert get_canonical_parameter_schema("run_code") is not None
    assert get_canonical_parameter_schema("run_tests") is not None

    ws = _ws(tmp_path)
    registry = ToolRegistry()
    registry.register(CodeSearchTool(delegate=_EchoDelegate()))
    registry.register(OpenFileTool(delegate=ReadFileTool(workspace_root=ws)))
    registry.register(FindSymbolTool(grep_delegate=GrepTool(workspace_root=ws)))
    registry.register(RunCodeTool(delegate=_EchoDelegate()))
    registry.register(RunTestsTool(delegate=_EchoDelegate()))

    assert validate_agentic_tool_call(registry, "code_search", {"query": "abc"}) is None
    assert validate_agentic_tool_call(registry, "open_file", {"path": "x.py"}) is None
    assert validate_agentic_tool_call(registry, "find_symbol", {"symbol": "foo"}) is None
    assert validate_agentic_tool_call(registry, "run_code", {"command": "git status"}) is None
    assert validate_agentic_tool_call(registry, "run_tests", {}) is None
