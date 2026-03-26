from __future__ import annotations

from app.tools.base_tool import BaseTool


class ToolRegistry:
    """Dictionary-based registry for all available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool, name: str | None = None) -> None:
        """Register a tool instance."""
        tool_name = name or tool.name
        if not tool_name:
            raise ValueError("Tool name cannot be empty.")
        self._tools[tool_name] = tool

    def get_tool(self, name: str | None) -> BaseTool | None:
        """Get tool instance by name."""
        if not name:
            return None
        return self._tools.get(name)

    def list_tool_descriptions(self) -> list[dict[str, str]]:
        """
        Return all tool descriptions for planner prompt assembly.

        Example:
        [
          {"name": "search_tool", "description": "Search relevant code"},
          {"name": "analyze_tool", "description": "Analyze code logic"}
        ]
        """
        items: list[dict[str, str]] = []
        for name, tool in self._tools.items():
            items.append(
                {
                    "name": name,
                    "description": tool.description or "",
                }
            )
        return items

    def as_prompt_text(self) -> str:
        """Render tool list to readable text for planner prompts."""
        lines: list[str] = []
        for item in self.list_tool_descriptions():
            lines.append(f"- {item['name']}: {item['description']}")
        return "\n".join(lines).strip()
