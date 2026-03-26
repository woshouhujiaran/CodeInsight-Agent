from __future__ import annotations

from typing import Any

from app.agent.tool_registry import ToolRegistry
from app.utils.logger import get_logger


class Executor:
    """Execute a list of tool calls in sequence."""

    def __init__(self, registry: ToolRegistry, logger_name: str = "codeinsight.executor") -> None:
        self.registry = registry
        self.logger = get_logger(logger_name)

    def execute_plan(self, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Execute planner output sequentially and return collected results.

        Input step examples:
        - {"tool": "search_tool", "input": "find auth logic"}
        - {"tool": "test_tool", "args": {"target": "tests/"}}
        """
        return self.execute_tools(plan)

    def execute_tools(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Core executor API.

        Args:
            tool_calls: ordered tool call list.
        Returns:
            list of execution results with tool/status/output.
        """
        results: list[dict[str, Any]] = []
        total_steps = len(tool_calls)
        self.logger.info("Executor received %d tool call(s).", total_steps)

        for idx, step in enumerate(tool_calls, start=1):
            tool_name = step.get("tool")
            args = self._normalize_step_args(step)
            self.logger.info(
                "Step %d/%d -> tool=%s, args_keys=%s",
                idx,
                total_steps,
                tool_name,
                list(args.keys()),
            )

            tool = self.registry.get_tool(tool_name)
            if tool is None:
                self.logger.error("Step %d/%d failed: tool `%s` not found.", idx, total_steps, tool_name)
                results.append(
                    {
                        "step": idx,
                        "tool": tool_name,
                        "status": "error",
                        "output": f"Tool `{tool_name}` not found.",
                    }
                )
                continue

            self.logger.info("Step %d/%d executing `%s`.", idx, total_steps, tool_name)
            try:
                tool_input = str(args.get("input", ""))
                output = tool.run(tool_input)
                self.logger.info("Step %d/%d `%s` completed.", idx, total_steps, tool_name)
                results.append({"step": idx, "tool": tool_name, "status": "ok", "output": output})
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("Step %d/%d `%s` crashed.", idx, total_steps, tool_name)
                results.append(
                    {"step": idx, "tool": tool_name, "status": "error", "output": str(exc)}
                )

        self.logger.info("Executor finished. Generated %d result item(s).", len(results))
        return results

    def _normalize_step_args(self, step: dict[str, Any]) -> dict[str, Any]:
        # Compatible with both:
        # 1) {"tool": "...", "args": {...}}
        # 2) {"tool": "...", "input": "..."}
        if isinstance(step.get("args"), dict):
            args = step["args"]
            if "input" in args:
                return {"input": args.get("input", "")}
            if "query" in args:
                return {"input": args.get("query", "")}
            return {"input": ""}
        if "input" in step:
            return {"input": step.get("input", "")}
        return {}
