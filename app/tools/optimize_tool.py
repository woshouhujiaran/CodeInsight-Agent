from __future__ import annotations

import json
from typing import Any

from app.llm.llm import LLMClient
from app.llm.prompt import (
    OPTIMIZE_TOOL_SYSTEM_PROMPT,
    build_optimize_tool_user_prompt,
)
from app.tools.base_tool import BaseTool, make_tool_result
from app.utils.logger import get_logger


class OptimizeTool(BaseTool):
    """Use LLM to produce optimization suggestions and rewritten code."""

    name = "optimize_tool"
    description = "优化代码片段，返回建议、优化后代码与改动说明。"

    def __init__(self, llm: LLMClient, logger_name: str = "codeinsight.tools.optimize") -> None:
        self.llm = llm
        self.logger = get_logger(logger_name)

    def run(self, input: dict[str, Any] | str) -> dict[str, Any]:
        code_snippet = self._extract_input(input)
        if not code_snippet:
            return make_tool_result(
                status="error",
                data={},
                error="optimize_tool requires non-empty input.",
                meta={"input_length": 0},
            )
        prompt = build_optimize_tool_user_prompt(code_snippet=code_snippet)
        raw = self.llm.generate_text(prompt=prompt, system_prompt=OPTIMIZE_TOOL_SYSTEM_PROMPT)
        self.logger.debug("OptimizeTool raw output: %s", raw)

        result = self._parse_result(raw=raw, original_code=code_snippet)
        return make_tool_result(
            status="ok",
            data=result,
            meta={"input_length": len(code_snippet)},
        )

    def _extract_input(self, input_value: dict[str, Any] | str) -> str:
        if isinstance(input_value, str):
            return input_value.strip()
        text = input_value.get("input")
        if isinstance(text, str):
            return text.strip()
        return ""

    def _parse_result(self, raw: str, original_code: str) -> dict[str, Any]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.warning("OptimizeTool received non-JSON output. Using fallback.")
            return self._fallback_result(original_code)

        if not isinstance(data, dict):
            self.logger.warning("OptimizeTool received non-object JSON. Using fallback.")
            return self._fallback_result(original_code)

        suggestions = data.get("optimization_suggestions")
        optimized_code = data.get("optimized_code")
        change_log = data.get("change_log")

        if not isinstance(suggestions, list):
            suggestions = ["建议拆分复杂逻辑并补充必要注释。"]
        suggestions = [str(item) for item in suggestions if str(item).strip()]
        if not suggestions:
            suggestions = ["建议提升代码可读性并增加异常处理。"]

        if not isinstance(optimized_code, str) or not optimized_code.strip():
            optimized_code = original_code

        if not isinstance(change_log, list):
            change_log = ["未能解析到完整改动说明，保留原代码作为安全回退。"]
        change_log = [str(item) for item in change_log if str(item).strip()]
        if not change_log:
            change_log = ["保持原有行为不变，建议后续基于测试进一步优化。"]

        return {
            "optimization_suggestions": suggestions,
            "optimized_code": optimized_code,
            "change_log": change_log,
        }

    def _fallback_result(self, original_code: str) -> dict[str, Any]:
        return {
            "optimization_suggestions": [
                "建议补充边界条件校验与异常处理。",
                "建议拆分过长函数并统一命名风格。",
            ],
            "optimized_code": original_code,
            "change_log": [
                "由于模型输出不可解析，当前返回原始代码以避免引入行为偏差。",
                "请接入真实 LLM API 后获得高质量优化结果。",
            ],
        }
