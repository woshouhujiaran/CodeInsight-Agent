from __future__ import annotations

from typing import Any

from app.agent.plan_schema import args_to_tool_input, topological_sort_steps
from app.agent.tool_registry import ToolRegistry
from app.utils.logger import get_logger


class Executor:
    """Execute planner steps in dependency order with bounded retries."""

    def __init__(self, registry: ToolRegistry, logger_name: str = "codeinsight.executor") -> None:
        self.registry = registry
        self.logger = get_logger(logger_name)

    def execute_plan(self, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Execute structured plan steps (id/deps/tool/args/success_criteria[/max_retries]).
        Steps are topologically ordered by deps.
        """
        if not plan:
            return []
        ordered = topological_sort_steps(plan)
        return self.execute_tools(ordered)

    def execute_tools(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Core executor API: run each step in list order (caller may have sorted by deps).
        """
        results: list[dict[str, Any]] = []
        total_steps = len(tool_calls)
        self.logger.info("Executor received %d tool call(s).", total_steps)

        for idx, step in enumerate(tool_calls, start=1):
            tool_name = step.get("tool")
            step_id = step.get("id", f"step_{idx}")
            args = step.get("args")
            if not isinstance(args, dict):
                args = {}
            success_criteria = str(step.get("success_criteria", "") or "")
            deps = step.get("deps") if isinstance(step.get("deps"), list) else []

            max_retries = step.get("max_retries", 0)
            if not isinstance(max_retries, int):
                max_retries = 0
            max_retries = max(0, min(2, max_retries))
            total_attempts = 1 + max_retries

            self.logger.info(
                "Step %d/%d id=%s -> tool=%s deps=%s max_retries=%d",
                idx,
                total_steps,
                step_id,
                tool_name,
                deps,
                max_retries,
            )

            tool = self.registry.get_tool(tool_name)
            if tool is None:
                self.logger.error("Step %d/%d failed: tool `%s` not found.", idx, total_steps, tool_name)
                results.append(
                    {
                        "step": idx,
                        "step_id": step_id,
                        "tool": tool_name,
                        "status": "error",
                        "output": f"Tool `{tool_name}` not found.",
                        "success_criteria": success_criteria,
                        "attempts": 0,
                        "deps": deps,
                    }
                )
                continue

            last_exc: str | None = None
            last_output = ""
            final_status = "error"
            attempts_used = 0

            for attempt in range(total_attempts):
                attempts_used = attempt + 1
                tool_input = args_to_tool_input(str(tool_name), args)
                self.logger.info(
                    "Step %d/%d `%s` attempt %d/%d input_len=%d",
                    idx,
                    total_steps,
                    tool_name,
                    attempts_used,
                    total_attempts,
                    len(tool_input),
                )
                try:
                    last_output = tool.run(tool_input)
                    final_status = "ok"
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = str(exc)
                    self.logger.warning(
                        "Step %d/%d `%s` attempt %d failed: %s",
                        idx,
                        total_steps,
                        tool_name,
                        attempts_used,
                        exc,
                    )
                    if attempt == total_attempts - 1:
                        last_output = last_exc
                        final_status = "error"

            results.append(
                {
                    "step": idx,
                    "step_id": step_id,
                    "tool": tool_name,
                    "status": final_status,
                    "output": last_output,
                    "success_criteria": success_criteria,
                    "attempts": attempts_used,
                    "deps": deps,
                }
            )
            self.logger.info(
                "Step %d/%d `%s` finished status=%s attempts=%d",
                idx,
                total_steps,
                tool_name,
                final_status,
                attempts_used,
            )

        self.logger.info("Executor finished. Generated %d result item(s).", len(results))
        return results
