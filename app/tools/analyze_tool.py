from __future__ import annotations

from app.llm.llm import LLMClient
from app.tools.base_tool import BaseTool
from app.utils.logger import get_logger


class AnalyzeTool(BaseTool):
    """Analyze code context and provide technical insights."""

    name = "analyze_tool"
    description = "分析代码片段并输出问题定位、风险和优化建议。"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.logger = get_logger("codeinsight.tools.analyze")

    def run(self, input: str) -> str:
        self.logger.info("AnalyzeTool analyzing context length=%d", len(input))
        prompt = (
            "请基于以下上下文做代码分析，输出：1) 关键发现 2) 潜在风险 3) 建议动作。\n\n"
            f"{input}"
        )
        return self.llm.generate_text(prompt=prompt)
