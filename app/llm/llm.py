from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any
from urllib import error, request

from app.utils.logger import get_logger


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
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        api_result = self._chat_completion(messages)
        if api_result is not None:
            return api_result

        # ---- fallback when API key/provider unavailable ----
        system_text = (system_prompt or "").lower()
        prompt_text = prompt.lower()

        # Planner fallback
        if "规划器" in (system_prompt or "") or "tool" in system_text and "json 数组" in (system_prompt or ""):
            return json.dumps(
                [
                    {"tool": "search_tool", "input": "检索与用户问题最相关的代码片段"},
                    {"tool": "analyze_tool", "input": "结合检索结果给出技术分析与结论"},
                ],
                ensure_ascii=False,
            )

        # Optimize tool fallback
        if "优化助手" in (system_prompt or "") or "optimized_code" in prompt_text:
            return json.dumps(
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

        # Test tool fallback
        if "测试工程师" in (system_prompt or "") or "test_code" in prompt_text:
            return json.dumps(
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

        # Generic analysis fallback
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
