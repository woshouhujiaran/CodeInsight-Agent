from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any
from urllib import error, request

from app.utils.logger import get_logger, log_event


@dataclass
class LLMClient:
    """
    Unified LLM wrapper.

    Default model is gpt-4o-mini. You can extend this class
    to route requests to DeepSeek Chat with the same interface.
    """

    model: str = "deepseek-chat"
    provider: str = "deepseek"
    timeout_seconds: int = 60

    def __post_init__(self) -> None:
        self.logger = get_logger("codeinsight.llm")

    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Generic text generation entry for planner/other modules.

        This is a bootstrap fallback implementation. Replace with actual
        provider SDK call (OpenAI/DeepSeek) while keeping the same method
        signature to avoid changing upper-layer modules.
        """
        started = time.perf_counter()
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        api_result = self._chat_completion(messages)
        if api_result is not None:
            log_event(
                self.logger,
                module="llm",
                action="generate_text",
                status="ok",
                duration_ms=int((time.perf_counter() - started) * 1000),
                provider=self.provider,
                model=self.model,
                source="api",
            )
            return api_result

        # ---- fallback when API key/provider unavailable ----
        system_text = (system_prompt or "").lower()
        prompt_text = prompt.lower()

        sp_low = (system_prompt or "").lower()

        # Recovery planner fallback (must run before generic planner — both contain 规划器)
        if "第二次规划" in sp_low or "恢复规划" in sp_low:
            out = json.dumps(
                [
                    {
                        "id": "r1",
                        "deps": [],
                        "tool": "search_tool",
                        "args": {"query": "项目 模块 入口 核心 代码结构 逻辑"},
                        "success_criteria": "放宽检索后得到非空相关片段",
                        "max_retries": 1,
                    },
                    {
                        "id": "r2",
                        "deps": ["r1"],
                        "tool": "analyze_tool",
                        "args": {
                            "input": "结合放宽后的检索结果分析；若仍无代码片段，说明假设并给出排查步骤。",
                        },
                        "success_criteria": "输出可执行结论或明确下一步",
                        "max_retries": 0,
                    },
                ],
                ensure_ascii=False,
            )
            log_event(
                self.logger,
                module="llm",
                action="generate_text",
                status="ok",
                duration_ms=int((time.perf_counter() - started) * 1000),
                provider=self.provider,
                model=self.model,
                source="fallback",
            )
            return out

        # Planner fallback (structured plan: id/deps/tool/args/success_criteria)
        if "规划器" in sp_low or ("json" in sp_low and "数组" in sp_low and "tool" in sp_low):
            out = json.dumps(
                [
                    {
                        "id": "s1",
                        "deps": [],
                        "tool": "search_tool",
                        "args": {"query": "检索与用户问题最相关的代码片段"},
                        "success_criteria": "检索到与问题相关的代码片段",
                        "max_retries": 1,
                    },
                    {
                        "id": "s2",
                        "deps": ["s1"],
                        "tool": "analyze_tool",
                        "args": {"input": "结合检索结果给出技术分析与结论"},
                        "success_criteria": "输出技术分析结论与可执行建议",
                        "max_retries": 0,
                    },
                ],
                ensure_ascii=False,
            )
            log_event(
                self.logger,
                module="llm",
                action="generate_text",
                status="ok",
                duration_ms=int((time.perf_counter() - started) * 1000),
                provider=self.provider,
                model=self.model,
                source="fallback",
            )
            return out

        # Optimize tool fallback
        if "优化助手" in (system_prompt or "") or "optimized_code" in prompt_text:
            out = json.dumps(
                {
                    "optimization_suggestions": [
                        "减少重复逻辑并提取函数",
                        "增加必要的参数校验与异常处理",
                    ],
                    "optimized_code": prompt,
                    "change_log": [
                        "保留原有行为的前提下给出结构优化建议",
                        "当前为占位输出，接入真实模型后可生成高质量重写代码",
                    ],
                },
                ensure_ascii=False,
            )
            log_event(
                self.logger,
                module="llm",
                action="generate_text",
                status="ok",
                duration_ms=int((time.perf_counter() - started) * 1000),
                provider=self.provider,
                model=self.model,
                source="fallback",
            )
            return out

        # Test tool fallback
        if "测试工程师" in (system_prompt or "") or "test_code" in prompt_text:
            out = json.dumps(
                {
                    "coverage_focus": ["主流程", "边界输入", "异常路径"],
                    "test_code": (
                        "import pytest\n\n"
                        "def target(x):\n"
                        "    return x\n\n"
                        "def test_main_path():\n"
                        "    assert target(1) == 1\n\n"
                        "def test_boundary_path():\n"
                        "    assert target(0) == 0\n\n"
                        "def test_error_path_placeholder():\n"
                        "    assert True\n"
                    ),
                },
                ensure_ascii=False,
            )
            log_event(
                self.logger,
                module="llm",
                action="generate_text",
                status="ok",
                duration_ms=int((time.perf_counter() - started) * 1000),
                provider=self.provider,
                model=self.model,
                source="fallback",
            )
            return out

        # Generic analysis fallback
        log_event(
            self.logger,
            module="llm",
            action="generate_text",
            status="ok",
            duration_ms=int((time.perf_counter() - started) * 1000),
            provider=self.provider,
            model=self.model,
            source="fallback",
        )
        return "已完成分析：建议先定位核心模块，再结合测试与性能数据做优化。"

    def generate_answer(
        self,
        user_query: str,
        context: str,
        history: list[dict[str, str]],
    ) -> str:
        prompt = (
            "请基于用户问题、历史对话和工具上下文，给出简洁专业的中文回答。\n\n"
            f"[用户问题]\n{user_query}\n\n"
            f"[历史消息数]\n{len(history)}\n\n"
            f"[工具上下文]\n{context}\n"
        )
        answer = self.generate_text(prompt=prompt)
        return answer

    def _chat_completion(self, messages: list[dict[str, str]]) -> str | None:
        provider = self.provider.lower().strip()
        api_key = self._resolve_api_key(provider)
        if not api_key:
            self.logger.warning("No API key found for provider=%s, using fallback mode.", provider)
            return None

        if provider == "deepseek":
            url = "https://api.deepseek.com/chat/completions"
        elif provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
        else:
            self.logger.warning("Unsupported provider=%s, using fallback mode.", provider)
            return None

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }

        req = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
                data = json.loads(body)
                return data["choices"][0]["message"]["content"]
        except error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            self.logger.error("LLM HTTP error: %s, body=%s", exc.code, error_body)
            return None
        except Exception as exc:  # noqa: BLE001
            self.logger.error("LLM request failed: %s", exc)
            return None

    def _resolve_api_key(self, provider: str) -> str:
        if provider == "deepseek":
            return os.getenv("DEEPSEEK_API_KEY", "").strip()
        if provider == "openai":
            return os.getenv("OPENAI_API_KEY", "").strip()
        return ""
