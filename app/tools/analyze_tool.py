from __future__ import annotations

from typing import Any

from app.llm.llm import LLMClient
from app.tools.base_tool import BaseTool, make_tool_result
from app.utils.logger import get_logger


class AnalyzeTool(BaseTool):
    """Analyze code context and provide technical insights."""

    name = "analyze_tool"
    description = "分析代码片段并输出问题定位、风险和优化建议。"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.logger = get_logger("codeinsight.tools.analyze")

    def run(self, input: dict[str, Any] | str) -> dict[str, Any]:
        text = self._extract_input(input)
        self.logger.info("AnalyzeTool analyzing context length=%d", len(text))
        if not text:
            return make_tool_result(
                status="error",
                data="",
                error="analyze_tool requires non-empty input.",
                meta={"input_length": 0},
            )
        prompt = (
            "请基于以下上下文做代码分析，输出：1) 关键发现 2) 潜在风险 3) 建议动作。\n\n"
            f"{text}"
        )
        analysis = self.llm.generate_text(prompt=prompt)
        return make_tool_result(status="ok", data=analysis, meta={"input_length": len(text)})

    def _extract_input(self, input_value: dict[str, Any] | str) -> str:
        if isinstance(input_value, str):
            return input_value.strip()
        text = input_value.get("input")
        if isinstance(text, str):
            return text.strip()
        return ""
