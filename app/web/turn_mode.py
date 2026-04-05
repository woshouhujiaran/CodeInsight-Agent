from __future__ import annotations

from typing import Any

_TURN_MODE_ARBITER_SYSTEM = """你是「问答模式 / 任务模式」的路由判定器。根据用户输入，只输出一个小写英文单词：qa 或 agentic。
不要解释、不要标点、不要 Markdown、不要第二个词。

qa：用自然语言解释即可；不依赖读取用户本地工作区里的具体文件，也不需要执行命令或改代码就能较好回答。
agentic：需要结合具体仓库（检索、打开文件、搜索符号、修改代码、运行测试等）才能可靠完成，或用户明显在要你对项目动手/查代码。

若脱离具体仓库也能讲清楚技术概念或通用做法，选 qa；只要必须落到「这份代码/这个目录」上，选 agentic。"""

_TURN_MODE_ARBITER_SYSTEM_WITH_WORKSPACE = """你是路由判定器。根据用户输入，只输出一个小写英文单词：qa、workspace_qa 或 agentic。
不要解释、不要标点、不要 Markdown、不要第二个词。

qa：不读取本地工作区里的具体文件也能较好回答（通用概念、语法、跨项目对比、原理等）。
workspace_qa：需要阅读当前工作区（README、配置、目录结构、入口脚本等）才能准确回答，但用户只要说明、概览、结构梳理或「如何本地运行」类指引；不要改代码、不要执行任务拆解式施工、不要替用户执行安装或跑服务。
agentic：用户要修改/修复代码、运行测试或命令、深度排查并动手改、多步开发任务，或明确要你「帮我运行/执行」。

若问题明显针对当前项目/文件夹但只需只读说明或运行步骤，优先 workspace_qa，而不是 agentic。"""


def arbitrate_turn_mode(
    llm: Any,
    user_content: str,
    *,
    fallback: str = "qa",
    workspace_bound: bool = False,
) -> str:
    """LLM 二次判定；解析失败时返回 fallback。已绑定工作区时可三分类，避免「必须读仓库」被误判成纯 qa 或完整任务模式。"""
    system = _TURN_MODE_ARBITER_SYSTEM_WITH_WORKSPACE if workspace_bound else _TURN_MODE_ARBITER_SYSTEM
    raw = str(
        llm.generate_text(
            prompt=f"用户输入：\n{user_content.strip()}",
            system_prompt=system,
        )
        or ""
    ).strip()
    lowered = raw.lower()
    first = lowered.split()[0] if lowered else ""
    if workspace_bound:
        if first.startswith("workspace_qa") or lowered.startswith("workspace_qa"):
            return "workspace_qa"
        if first.startswith("agentic") or lowered.startswith("agentic"):
            return "agentic"
        if first.startswith("qa") or first == "q" or lowered == "qa":
            return "qa"
        return fallback if fallback in {"qa", "agentic", "workspace_qa"} else "qa"
    if first.startswith("agentic") or lowered.startswith("agentic"):
        return "agentic"
    if first.startswith("qa") or first == "q" or lowered == "qa":
        return "qa"
    return fallback if fallback in {"qa", "agentic"} else "qa"
