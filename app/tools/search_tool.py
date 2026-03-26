from __future__ import annotations

import json

from app.rag.retriever import CodeRetriever
from app.tools.base_tool import BaseTool
from app.utils.logger import get_logger


class SearchTool(BaseTool):
    """Retrieve top-k relevant code snippets from FAISS index."""

    name = "search_tool"
    description = "根据query检索最相关代码片段，返回文件路径与内容。"

    def __init__(self, retriever: CodeRetriever, top_k: int = 5) -> None:
        self.retriever = retriever
        self.top_k = top_k
        self.logger = get_logger("codeinsight.tools.search")

    def run(self, input: str) -> str:
        self.logger.info("SearchTool query: %s", input)
        results = self.retriever.retrieve(query=input, top_k=self.top_k)
        return json.dumps(results, ensure_ascii=False)
