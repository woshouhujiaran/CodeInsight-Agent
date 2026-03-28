from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

try:
    import jsonschema
    from jsonschema import ValidationError
except ImportError:  # pragma: no cover
    jsonschema = None  # type: ignore[assignment]
    ValidationError = Exception  # type: ignore[misc, assignment]

from app.agent.plan_schema import ALLOWED_TOOLS, validate_tool_args
from app.tools.run_command_tool import validate_run_command_arguments

if TYPE_CHECKING:
    from app.agent.tool_registry import ToolRegistry

# 与 plan_schema.validate_tool_args 语义对齐的 parameters JSON Schema（OpenAI function.parameters）
SEARCH_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1,
            "description": "检索查询（与 input 二选一，优先 query）",
        },
        "input": {
            "type": "string",
            "minLength": 1,
            "description": "检索输入（与 query 二选一）",
        },
    },
    "anyOf": [
        {"required": ["query"]},
        {"required": ["input"]},
    ],
}

INPUT_ONLY_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["input"],
    "properties": {
        "input": {
            "type": "string",
            "minLength": 1,
            "description": "自然语言或带标记的代码/说明文本",
        },
    },
}

READ_FILE_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["path"],
    "properties": {
        "path": {"type": "string", "minLength": 1, "description": "相对工作区根的文件路径"},
        "start_line": {"type": "integer", "minimum": 1, "description": "起始行号（1-based，含）"},
        "end_line": {"type": "integer", "minimum": 1, "description": "结束行号（1-based，含）"},
        "max_chars": {
            "type": "integer",
            "minimum": 1,
            "maximum": 2_000_000,
            "description": "返回正文最大字符数（含截断提示前的正文）",
        },
    },
}

LIST_DIR_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["path"],
    "properties": {
        "path": {"type": "string", "minLength": 1, "description": "目录路径（相对工作区根，可用 .）"},
        "depth": {
            "type": "integer",
            "minimum": 1,
            "maximum": 8,
            "description": "相对 path 向下列出的深度（1 仅直接子项）",
        },
        "max_entries": {"type": "integer", "minimum": 1, "maximum": 5000, "description": "最多返回条目数"},
    },
}

GREP_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["pattern", "path"],
    "properties": {
        "pattern": {"type": "string", "minLength": 1, "description": "Python re 正则表达式"},
        "path": {"type": "string", "minLength": 1, "description": "文件或目录（相对工作区根）"},
        "glob": {"type": "string", "description": "文件名 glob，如 *.py"},
        "max_matches": {"type": "integer", "minimum": 1, "maximum": 5000, "description": "最多匹配条数"},
    },
}

WRITE_FILE_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["path", "content"],
    "properties": {
        "path": {"type": "string", "minLength": 1, "description": "相对工作区根的文件路径"},
        "content": {"type": "string", "description": "写入的完整 UTF-8 文本"},
        "create_only": {"type": "boolean", "description": "为 true 时仅当文件不存在才写入"},
        "expected_content_hash": {
            "type": "string",
            "minLength": 1,
            "description": "与 read_file_tool 返回的 meta.content_sha256 一致时才允许覆盖已存在文件",
        },
    },
}

APPLY_PATCH_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["patch"],
    "properties": {
        "patch": {"type": "string", "minLength": 1, "description": "unified diff 全文（UTF-8）"},
        "strip": {"type": "integer", "minimum": 0, "maximum": 1, "description": "路径前缀剥离层级；缺省由实现自动尝试 1 再 0"},
    },
}

RUN_COMMAND_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "argv": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": "参数列表（等价于 subprocess 的 argv，禁止 shell=True）",
        },
        "command": {
            "type": "string",
            "minLength": 1,
            "description": "整条命令字符串，将按 shlex(posix=True) 拆分为 argv",
        },
        "timeout_seconds": {
            "type": "number",
            "minimum": 1,
            "maximum": 600,
            "description": "超时秒数，默认由工具决定",
        },
    },
    "oneOf": [
        {"required": ["argv"]},
        {"required": ["command"]},
    ],
}

TOOL_PARAMETER_SCHEMAS: dict[str, dict[str, Any]] = {
    "search_tool": SEARCH_TOOL_PARAMETERS,
    "analyze_tool": INPUT_ONLY_PARAMETERS,
    "optimize_tool": INPUT_ONLY_PARAMETERS,
    "test_tool": INPUT_ONLY_PARAMETERS,
    "read_file_tool": READ_FILE_TOOL_PARAMETERS,
    "list_dir_tool": LIST_DIR_TOOL_PARAMETERS,
    "grep_tool": GREP_TOOL_PARAMETERS,
    "write_file_tool": WRITE_FILE_TOOL_PARAMETERS,
    "apply_patch_tool": APPLY_PATCH_TOOL_PARAMETERS,
    "run_command_tool": RUN_COMMAND_TOOL_PARAMETERS,
}

# 未在 TOOL_PARAMETER_SCHEMAS 中声明的已注册工具（如测试替身）使用宽松 schema
AD_HOC_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
}


def get_canonical_parameter_schema(tool_name: str) -> dict[str, Any] | None:
    """Return schema for built-in tools; None if name is not canonical."""
    return TOOL_PARAMETER_SCHEMAS.get(tool_name)


def openai_function_spec(*, name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Single OpenAI Chat Completions style tool declaration."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (description or "").strip(),
            "parameters": parameters,
        },
    }


def compact_tool_specs_for_prompt(specs: list[dict[str, Any]], *, max_description_chars: int = 280) -> str:
    """Token-friendly JSON array: name, truncated description, parameters schema."""
    slim: list[dict[str, Any]] = []
    for item in specs:
        fn = item.get("function") if isinstance(item, dict) else None
        if not isinstance(fn, dict):
            continue
        name = fn.get("name", "")
        desc = str(fn.get("description") or "")
        if len(desc) > max_description_chars:
            desc = desc[: max_description_chars - 3] + "..."
        slim.append(
            {
                "name": name,
                "description": desc,
                "parameters": fn.get("parameters") or {"type": "object"},
            }
        )
    return json.dumps(slim, ensure_ascii=False)


def validate_agentic_tool_call(registry: "ToolRegistry", tool_name: Any, arguments: Any) -> str | None:
    """
    Validate one agentic tool invocation before execution.

    Returns None if OK, otherwise a short human-readable error (Chinese).
    """
    if not isinstance(tool_name, str) or not tool_name.strip():
        return "tool_calls[].name 必须为非空字符串。"
    name = tool_name.strip()
    if registry.get_tool(name) is None:
        return f"未注册的工具 `{name}`，请仅从系统提示中的可用工具列表选择。"

    if not isinstance(arguments, dict):
        return "tool_calls[].arguments 必须是 JSON 对象（object）。"

    schema = registry.get_parameter_schema(name)

    if jsonschema is not None:
        try:
            jsonschema.validate(instance=arguments, schema=schema)
        except ValidationError as exc:  # type: ignore[misc]
            path = ".".join(str(p) for p in exc.path) if exc.path else "(root)"
            return f"参数不符合 JSON Schema（{path}）：{exc.message}"

    if name in ALLOWED_TOOLS:
        return validate_tool_args(name, arguments)
    if name == "run_command_tool":
        return validate_run_command_arguments(arguments)
    return None
