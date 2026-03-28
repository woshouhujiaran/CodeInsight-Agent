from __future__ import annotations

from typing import Any

from app.agent.tool_specs import (
    AD_HOC_TOOL_PARAMETERS,
    get_canonical_parameter_schema,
    openai_function_spec,
)
from app.tools.base_tool import BaseTool


class ToolRegistry:
    """Dictionary-based registry for all available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._parameter_schemas: dict[str, dict[str, Any]] = {}

    def register(
        self,
        tool: BaseTool,
        name: str | None = None,
        *,
        parameter_schema: dict[str, Any] | None = None,
    ) -> None:
        """Register a tool instance."""
        tool_name = name or tool.name
        if not tool_name:
            raise ValueError("Tool name cannot be empty.")
        self._tools[tool_name] = tool
        if parameter_schema is not None:
            self._parameter_schemas[tool_name] = parameter_schema
        else:
            canonical = get_canonical_parameter_schema(tool_name)
            self._parameter_schemas[tool_name] = canonical if canonical is not None else AD_HOC_TOOL_PARAMETERS

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

    def get_parameter_schema(self, tool_name: str) -> dict[str, Any]:
        """JSON Schema for tool `arguments` (OpenAI function.parameters)."""
        return self._parameter_schemas.get(tool_name, AD_HOC_TOOL_PARAMETERS)

    def list_specs(self) -> list[dict[str, Any]]:
        """OpenAI-style function specs for all registered tools (prompt + validation source)."""
        specs: list[dict[str, Any]] = []
        for name, tool in self._tools.items():
            specs.append(
                openai_function_spec(
                    name=name,
                    description=tool.description or "",
                    parameters=self.get_parameter_schema(name),
                )
            )
        return specs
