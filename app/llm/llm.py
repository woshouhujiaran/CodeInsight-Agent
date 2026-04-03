from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import time
from typing import Any
from urllib import error, request

from app.utils.logger import get_logger, log_event

# Agentic system prompt：工具编排策略（与 RAG/实现无关，仅约束模型行为）
AGENTIC_TOOL_USE_POLICY = (
    "工具使用策略：\n"
    "1) 不确定符号、文件或模块位置时，先用 search_tool 做语义检索，再从返回片段中取得真实路径或关键词；\n"
    "2) 需要完整控制流、函数体、多行上下文或精确字符串定位时，使用 read_file_tool 与 grep_tool；路径须来自检索结果、"
    "list_dir_tool 或用户明确给出的工作区内相对路径；\n"
    "3) 严禁编造不存在的路径、行号或参数；若工具返回 error/status 非 ok 或提示路径/参数非法，必须阅读报错内容，"
    "修正 arguments 后再次发起 tool_calls，禁止无改正地重复相同调用。\n\n"
)

AGENTIC_JSON_SYSTEM_SUFFIX = (
    "你必须只输出一个 JSON 对象，不要 Markdown、不要代码围栏、不要额外说明。"
    '格式二选一：1) {"type":"final","content":"<给用户的最终回答>"}；'
    '2) {"type":"tool_calls","calls":[{"name":"<工具名>","arguments":{...}}]}。'
    "arguments 必须是 JSON 对象；不要编造工具名。"
)


def strip_json_fences(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```\s*$", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def parse_agentic_turn(raw: str) -> dict[str, Any] | None:
    """Parse and validate agentic turn JSON. Returns None if invalid."""
    try:
        data = json.loads(strip_json_fences(raw))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    typ = data.get("type")
    if typ == "final":
        content = data.get("content")
        if not isinstance(content, str):
            return None
        return {"type": "final", "content": content}
    if typ == "tool_calls":
        calls = data.get("calls")
        if not isinstance(calls, list) or not calls:
            return None
        normalized: list[dict[str, Any]] = []
        for item in calls:
            if not isinstance(item, dict):
                return None
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                return None
            args = item.get("arguments")
            if args is None:
                args = {}
            if not isinstance(args, dict):
                return None
            normalized.append({"name": name.strip(), "arguments": args})
        return {"type": "tool_calls", "calls": normalized}
    return None


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

        if "task_board_json" in sp_low:
            out = json.dumps(
                [
                    {
                        "id": "t1",
                        "title": "定位相关代码与入口",
                        "description": "检索工作区中与用户目标最相关的模块、入口和关键文件。",
                        "depends_on": [],
                        "status": "pending",
                        "acceptance": "能够指出后续修改需要依赖的文件或模块。",
                    },
                    {
                        "id": "t2",
                        "title": "分析变更方案",
                        "description": "基于已定位代码确认实现方式、影响范围和必要约束。",
                        "depends_on": ["t1"],
                        "status": "pending",
                        "acceptance": "形成可执行的修改方案或手动变更建议。",
                    },
                    {
                        "id": "t3",
                        "title": "执行代码修改或给出补丁",
                        "description": "在权限允许时实施代码变更，否则输出可复制 diff 或分步编辑说明。",
                        "depends_on": ["t2"],
                        "status": "pending",
                        "acceptance": "代码修改已落地，或给出结构化补丁建议。",
                    },
                    {
                        "id": "t4",
                        "title": "验证结果并总结",
                        "description": "检查修改结果，结合测试或静态信息输出最终结论与剩余风险。",
                        "depends_on": ["t3"],
                        "status": "pending",
                        "acceptance": "给出验证结论、未完成项和下一步建议。",
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

    def generate_agentic_json_turn(
        self,
        messages: list[dict[str, str]],
        *,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        One agentic step: chat completion expecting a single JSON object.

        On parse failure, retries once with an extra user nudge (no new deps).
        Falls back to a generic final message if the API is unavailable.
        """
        started = time.perf_counter()
        base: list[dict[str, str]] = []
        if system_prompt:
            base.append({"role": "system", "content": system_prompt})
        base.extend(messages)

        def complete(extra_user: str | None, *, json_mode: bool) -> str | None:
            msgs = list(base)
            if extra_user:
                msgs.append({"role": "user", "content": extra_user})
            return self._chat_completion_messages(msgs, response_format_json=json_mode)

        raw = complete(None, json_mode=True)
        if raw is None:
            raw = complete(None, json_mode=False)
        parsed = parse_agentic_turn(raw or "")
        if parsed is not None:
            log_event(
                self.logger,
                module="llm",
                action="generate_agentic_json_turn",
                status="ok",
                duration_ms=int((time.perf_counter() - started) * 1000),
                provider=self.provider,
                model=self.model,
                source="api" if raw else "fallback",
                attempts=1,
            )
            return parsed

        nudge = (
            "上一次输出不是合法 JSON 或不符合约定。请严格输出一个 JSON 对象："
            '{"type":"final","content":"..."} 或 {"type":"tool_calls","calls":[...]}，不要其它文字。'
        )
        raw2 = complete(nudge, json_mode=True)
        if raw2 is None:
            raw2 = complete(nudge, json_mode=False)
        parsed2 = parse_agentic_turn(raw2 or "")
        if parsed2 is not None:
            log_event(
                self.logger,
                module="llm",
                action="generate_agentic_json_turn",
                status="ok",
                duration_ms=int((time.perf_counter() - started) * 1000),
                provider=self.provider,
                model=self.model,
                source="api" if raw2 else "fallback",
                attempts=2,
            )
            return parsed2

        log_event(
            self.logger,
            module="llm",
            action="generate_agentic_json_turn",
            status="error",
            duration_ms=int((time.perf_counter() - started) * 1000),
            provider=self.provider,
            model=self.model,
            source="fallback",
            attempts=2,
        )
        return {
            "type": "final",
            "content": "模型未返回可解析的 JSON，请检查 API 或提示词。",
        }

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
        return self._chat_completion_messages(messages, response_format_json=False)

    def _chat_completion_messages(
        self,
        messages: list[dict[str, str]],
        *,
        response_format_json: bool,
    ) -> str | None:
        provider = self.provider.lower().strip()
        api_key = self._resolve_api_key(provider)
        if not api_key:
            self.logger.warning("No API key found for provider=%s, using fallback mode.", provider)
            return None

        if provider == "deepseek":
            url = "https://api.deepseek.com/chat/completions"
        elif provider == "openai":
            base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip().rstrip("/")
            url = f"{base_url}/chat/completions"
        else:
            self.logger.warning("Unsupported provider=%s, using fallback mode.", provider)
            return None

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}

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
            if response_format_json and exc.code in (400, 404):
                # Provider may not support response_format; caller can retry without it.
                self.logger.info("JSON response_format rejected; caller may retry without json_mode.")
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
