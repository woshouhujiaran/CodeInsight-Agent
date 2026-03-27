from __future__ import annotations

from typing import Any

from app.rag.retriever import CodeRetriever
from app.tools.base_tool import BaseTool, make_tool_result
from app.utils.logger import get_logger


class SearchTool(BaseTool):
    """Retrieve top-k relevant code snippets from FAISS index."""

    name = "search_tool"
    description = "根据query检索最相关代码片段，返回文件路径与内容。"

    def __init__(self, retriever: CodeRetriever, top_k: int = 5) -> None:
        self.retriever = retriever
        self.top_k = top_k
        self.logger = get_logger("codeinsight.tools.search")

    def run(self, input: dict[str, Any] | str) -> dict[str, Any]:
        query = self._extract_query(input)
        self.logger.info("SearchTool query: %s", query)
        if not query:
            return make_tool_result(
                status="error",
                data=[],
                error="search_tool requires non-empty query/input.",
                meta={"top_k": self.top_k, "query_length": 0},
            )
        results = self.retriever.retrieve(query=query, top_k=self.top_k)
        return make_tool_result(
            status="ok",
            data=results,
            meta={"top_k": self.top_k, "query_length": len(query)},
        )

    def _extract_query(self, input_value: dict[str, Any] | str) -> str:
        if isinstance(input_value, str):
            return input_value.strip()
        query = input_value.get("query")
        if isinstance(query, str) and query.strip():
            return query.strip()
        fallback = input_value.get("input")
        if isinstance(fallback, str):
            return fallback.strip()
        return ""
