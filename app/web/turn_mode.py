from __future__ import annotations

from typing import Any

_TURN_MODE_ARBITER_SYSTEM = """你是「问答模式 / 任务模式」的路由判定器。根据用户输入，只输出一个小写英文单词：qa 或 agentic。
不要解释、不要标点、不要 Markdown、不要第二个词。

qa：用自然语言解释即可；不依赖读取用户本地工作区里的具体文件，也不需要执行命令或改代码就能较好回答。
agentic：需要结合具体仓库（检索、打开文件、搜索符号、修改代码、运行测试等）才能可靠完成，或用户明显在要你对项目动手/查代码。

若脱离具体仓库也能讲清楚技术概念或通用做法，选 qa；只要必须落到「这份代码/这个目录」上，选 agentic。"""


def arbitrate_turn_mode(llm: Any, user_content: str, *, fallback: str = "qa") -> str:
    """LLM 二次判定；解析失败时返回 fallback。"""
    raw = str(
        llm.generate_text(
            prompt=f"用户输入：\n{user_content.strip()}",
            system_prompt=_TURN_MODE_ARBITER_SYSTEM,
        )
        or ""
    ).strip()
    lowered = raw.lower()
    first = lowered.split()[0] if lowered else ""
    if first.startswith("agentic") or lowered.startswith("agentic"):
        return "agentic"
    if first.startswith("qa") or first == "q" or lowered == "qa":
        return "qa"
    return fallback if fallback in {"qa", "agentic"} else "qa"
