from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import time
from typing import Any
from uuid import uuid4

from app.agent.plan_schema import topological_sort_steps
from app.agent.tool_registry import ToolRegistry
from app.agent.tool_specs import validate_agentic_tool_call
from app.tools.base_tool import ensure_tool_result, make_tool_result, tool_result_to_legacy_output
from app.utils.logger import get_logger


class Executor:
    """Execute planner steps in dependency order with bounded retries."""

    _RETRY_BACKOFF_SECONDS = (0.3, 0.8, 1.5)

    def __init__(
        self,
        registry: ToolRegistry,
        logger_name: str = "codeinsight.executor",
        *,
        step_timeout_seconds: float | None = None,
    ) -> None:
        self.registry = registry
        self.logger = get_logger(logger_name)
        self.step_timeout_seconds = step_timeout_seconds

    def execute_plan(self, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Execute structured plan steps (id/deps/tool/args/success_criteria[/max_retries]).
        Steps are topologically ordered by deps.
        """
        if not plan:
            return []
        ordered = topological_sort_steps(plan)
        return self.execute_tools(ordered)

    def execute_agentic_calls(
        self,
        calls: list[dict[str, Any]],
        *,
        cancel_event: Any | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run agentic-style tool invocations in list order (no deps).

        Each call: {"name": str, "arguments": dict}.
        Invalid arguments produce an error result without invoking the tool.
        """
        if not calls:
            return []
        results: list[dict[str, Any]] = []
        for batch_idx, call in enumerate(calls, start=1):
            if cancel_event is not None and cancel_event.is_set():
                break
            step_started = time.perf_counter()
            if not isinstance(call, dict):
                name: Any = None
                args: dict[str, Any] = {}
            else:
                name = call.get("name")
                args = call.get("arguments")
                if not isinstance(args, dict):
                    args = {}
            step_id = f"ag_{batch_idx - 1}_{uuid4().hex[:10]}"

            err = validate_agentic_tool_call(self.registry, name, args)
            if err:
                results.append(
                    {
                        "step": batch_idx,
                        "step_id": step_id,
                        "tool": name,
                        "status": "error",
                        "output": err,
                        "tool_result": make_tool_result(
                            status="error",
                            data=None,
                            error=err,
                            meta={"invalid_arguments": True},
                        ),
                        "success_criteria": "",
                        "attempts": 0,
                        "error_type": "permanent",
                        "duration_ms": int((time.perf_counter() - step_started) * 1000),
                        "timed_out": False,
                        "deps": [],
                    }
                )
                continue

            batch = self.execute_tools(
                [
                    {
                        "id": step_id,
                        "deps": [],
                        "tool": name,
                        "args": args,
                        "success_criteria": "",
                        "max_retries": 0,
                    }
                ]
            )
            if batch:
                row = dict(batch[0])
                row["step"] = batch_idx
                results.append(row)
        return results

    def execute_tools(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Core executor API: run each step in list order (caller may have sorted by deps).
        """
        results: list[dict[str, Any]] = []
        total_steps = len(tool_calls)
        self.logger.info("Executor received %d tool call(s).", total_steps)

        for idx, step in enumerate(tool_calls, start=1):
            step_started = time.perf_counter()
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
            timeout_seconds = self._resolve_timeout(step.get("timeout_seconds"))

            self.logger.info(
                "Step %d/%d id=%s -> tool=%s deps=%s max_retries=%d timeout=%.2fs",
                idx,
                total_steps,
                step_id,
                tool_name,
                deps,
                max_retries,
                timeout_seconds if timeout_seconds is not None else 0.0,
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
                        "tool_result": make_tool_result(
                            status="error",
                            data=None,
                            error=f"Tool `{tool_name}` not found.",
                            meta={"missing_tool": True},
                        ),
                        "success_criteria": success_criteria,
                        "attempts": 0,
                        "error_type": "permanent",
                        "duration_ms": int((time.perf_counter() - step_started) * 1000),
                        "timed_out": False,
                        "deps": deps,
                    }
                )
                continue

            last_output = ""
            final_status = "error"
            attempts_used = 0
            timed_out = False
            error_type = ""
            normalized = make_tool_result(status="error", data=None, error="", meta={})

            for attempt in range(total_attempts):
                attempts_used = attempt + 1
                self.logger.info(
                    "Step %d/%d `%s` attempt %d/%d args_keys=%s",
                    idx,
                    total_steps,
                    tool_name,
                    attempts_used,
                    total_attempts,
                    sorted(args.keys()),
                )
                try:
                    raw = self._run_with_timeout(tool, args, timeout_seconds=timeout_seconds)
                    normalized = ensure_tool_result(raw)
                    last_output = tool_result_to_legacy_output(normalized)
                    final_status = str(normalized.get("status", "ok"))
                    if final_status == "error" and not normalized.get("error"):
                        normalized["error"] = "Tool returned error status without message."
                    if final_status == "error":
                        error_type = self._classify_error(str(normalized.get("error", "")), timed_out=False)
                    else:
                        error_type = ""
                    if final_status == "ok":
                        break
                except TimeoutError as exc:
                    timed_out = True
                    final_status = "error"
                    error_type = "transient"
                    normalized = make_tool_result(
                        status="error",
                        data=None,
                        error=str(exc),
                        meta={"timed_out": True},
                    )
                    last_output = str(exc)
                    self.logger.warning(
                        "Step %d/%d `%s` attempt %d timed out: %s",
                        idx,
                        total_steps,
                        tool_name,
                        attempts_used,
                        exc,
                    )
                except Exception as exc:  # noqa: BLE001
                    final_status = "error"
                    error_type = self._classify_error(str(exc), timed_out=False)
                    normalized = make_tool_result(
                        status="error",
                        data=None,
                        error=str(exc),
                        meta={"exception_raised": True},
                    )
                    last_output = str(exc)
                    self.logger.warning(
                        "Step %d/%d `%s` attempt %d failed: %s",
                        idx,
                        total_steps,
                        tool_name,
                        attempts_used,
                        exc,
                    )

                is_last_attempt = attempt == total_attempts - 1
                should_retry = final_status == "error" and (not is_last_attempt) and error_type == "transient"
                if should_retry:
                    delay = self._backoff_seconds(attempt)
                    self.logger.info(
                        "Step %d/%d `%s` transient failure, retry after %.1fs",
                        idx,
                        total_steps,
                        tool_name,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    break

            results.append(
                {
                    "step": idx,
                    "step_id": step_id,
                    "tool": tool_name,
                    "status": final_status,
                    "output": last_output,
                    "tool_result": normalized,
                    "success_criteria": success_criteria,
                    "attempts": attempts_used,
                    "error_type": error_type,
                    "duration_ms": int((time.perf_counter() - step_started) * 1000),
                    "timed_out": timed_out,
                    "deps": deps,
                }
            )
            self.logger.info(
                "Step %d/%d `%s` finished status=%s attempts=%d error_type=%s timed_out=%s",
                idx,
                total_steps,
                tool_name,
                final_status,
                attempts_used,
                error_type,
                timed_out,
            )

        self.logger.info("Executor finished. Generated %d result item(s).", len(results))
        return results

    def _resolve_timeout(self, value: Any) -> float | None:
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
        return self.step_timeout_seconds

    def _backoff_seconds(self, attempt: int) -> float:
        if attempt < len(self._RETRY_BACKOFF_SECONDS):
            return self._RETRY_BACKOFF_SECONDS[attempt]
        return self._RETRY_BACKOFF_SECONDS[-1]

    def _run_with_timeout(
        self,
        tool: Any,
        tool_input: dict[str, Any],
        *,
        timeout_seconds: float | None,
    ) -> Any:
        if timeout_seconds is None:
            return tool.run(tool_input)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(tool.run, tool_input)
            try:
                return future.result(timeout=timeout_seconds)
            except FutureTimeoutError as exc:
                raise TimeoutError(f"Step timed out after {timeout_seconds:.2f}s") from exc

    def _classify_error(self, message: str, *, timed_out: bool) -> str:
        if timed_out:
            return "transient"
        text = message.lower()
        transient_tokens = (
            "timeout",
            "timed out",
            "temporar",
            "rate limit",
            "connection",
            "network",
            "unavailable",
            "429",
            "503",
        )
        if any(token in text for token in transient_tokens):
            return "transient"
        return "permanent"
