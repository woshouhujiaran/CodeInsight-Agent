from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable

from app.agent.agent import AgenticTurnResult
from app.agent.task_board import TaskBoard, TaskItem
from app.contracts import ServiceEvent, normalize_task_results, normalize_test_summary, normalize_tool_trace
from app.web.session_store import SessionStore, normalize_session_settings
from app.web.test_runner import run_project_test_command

EventCallback = Callable[[ServiceEvent], None]


class StreamCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class SafetyRefusal:
    reason: str
    answer: str


@dataclass(frozen=True)
class ClarificationPrompt:
    answer: str


def _contains_any_keyword(text: str, lowered: str, keywords: tuple[str, ...]) -> bool:
    for keyword in keywords:
        if keyword in text or keyword in lowered:
            return True
    return False


def _looks_like_project_review_request(text: str, lowered: str) -> bool:
    review_markers = (
        "review",
        "代码审查",
        "审查",
        "全面阅读",
        "全面检查",
        "阅读",
        "通读",
        "读一下",
        "读一遍",
        "过一遍",
        "梳理",
    )
    scope_markers = (
        "后端",
        "前端",
        "服务",
        "接口",
        "模块",
        "项目",
        "仓库",
        "代码库",
        "调用链",
        "backend",
        "frontend",
    )
    goal_markers = (
        "改进",
        "优化",
        "问题",
        "风险",
        "体验",
        "薄弱",
        "建议",
        "review",
    )
    return _contains_any_keyword(text, lowered, review_markers) and _contains_any_keyword(
        text, lowered, scope_markers
    ) and (_contains_any_keyword(text, lowered, goal_markers) or "全面" in text or "全量" in text)


class TurnModeDecider:
    """区分「纯自然语言 QA」与「需要动工作区的分步任务模式」。"""

    _PATH_PATTERN = re.compile(r"(?<!\S)(?:[\w.-]+[\\/])+[\w.-]+")

    # 明确指向「某套代码/工作区」的语境（避免单独「仓库」触发 git vs svn 类泛问）
    _PROJECT_SCOPE_MARKERS = (
        "当前项目",
        "这个项目",
        "本仓库",
        "当前仓库",
        "在这个项目",
        "在当前仓库",
        "在项目中",
        "在仓库里",
        "代码库",
        "workspace_root",
        "集成到当前项目",
        "工作区根",
        "会话工作区",
    )

    _PROJECT_OVERVIEW_MARKERS = (
        "做什么",
        "是干什么的",
        "用途",
        "作用",
        "项目介绍",
        "项目概览",
        "技术栈",
        "主要功能",
        "核心功能",
        "整体结构",
        "架构",
        "代码结构",
    )

    _WORKSPACE_QA_EXPLANATION_MARKERS = (
        "介绍",
        "说明",
        "说说",
        "讲讲",
        "总结",
        "概览",
        "整体",
        "大概",
        "看看",
    )

    _WORKSPACE_QA_TARGET_MARKERS = (
        "这个项目",
        "当前项目",
        "这个仓库",
        "当前仓库",
        "这个文件夹",
        "当前文件夹",
        "这个目录",
        "当前目录",
        "本地文件",
        "本地文件夹",
        "这个文件",
        "当前文件",
    )

    # 仅想了解「如何跑起来 / 怎么用」，仍属只读 workspace_qa；与「帮我运行」等真执行区分开
    _WORKSPACE_QA_RUN_OR_USAGE_MARKERS = (
        "怎么运行",
        "如何运行",
        "怎样运行",
        "怎么启动",
        "如何启动",
        "怎样启动",
        "怎么安装",
        "如何安装",
        "怎样安装",
        "本地运行",
        "本地开发",
        "开发环境",
        "怎么用",
        "如何使用",
        "怎样用",
        "使用说明",
        "运行说明",
        "启动命令",
        "启动方式",
        "如何部署",
        "怎么部署",
        "怎样部署",
        "入门",
        "上手",
    )

    _WORKSPACE_QA_EXECUTION_INTENT_MARKERS = (
        "帮我运行",
        "帮我启动",
        "帮我执行",
        "运行一下",
        "启动一下",
        "执行一下",
        "跑一下",
        "直接运行",
        "现在就运行",
        "替我运行",
    )

    # 需要读/搜/改/跑的具体行动信号（命中则任务模式；避免过短词误伤「二分查找」等术语）
    _OPERATIONAL_MARKERS = (
        "定位实现",
        "定位到",
        "定位在",
        "查找文件",
        "搜索代码",
        "搜索符号",
        "列出文件",
        "列出目录",
        "找到实现",
        "找到文件",
        "帮我找",
        "找一下",
        "看看代码",
        "读一下文件",
        "读这个文件",
        "打开文件",
        "目录结构",
        "项目结构",
        "pytest",
        "unittest",
        "跑测试",
        "运行测试",
        "构建",
        "打包",
        "ci ",
        " ci",
        "build",
        "修改",
        "修复",
        "重构",
        "删除",
        "补丁",
        "patch",
        "diff",
        "应用补丁",
        "审查",
        "review",
        "代码审查",
        "通读",
        "全面阅读",
        "全面检查",
        "梳理",
        "过一遍",
        "读一遍",
        "哪个文件",
        "哪些文件",
        "入口在",
        "调用链",
        "从哪里开始看",
        "怎么验证",
        "如何验证",
        "写测试",
        "新增测试",
        "列出关键文件",
        "查找文件",
        "找到实现",
        "定位实现",
        "会话存储实现",
        "模式判定逻辑",
        "如何实现",
        "怎么实现",
        "怎么做",
        "在哪实现",
        "代码在哪",
        "加字段",
        "加功能",
        "集成到",
        "介绍一下",
        "介绍下",
        "说说这个",
        "讲讲这个",
        "跑起来",
    )

    def infer_with_meta(self, user_content: str) -> tuple[str, bool]:
        """返回 (mode, need_llm_arbitration)。后者为 True 时表示启发式不确定，需再调 LLM 判定。"""
        text = str(user_content or "").strip()
        lowered = text.lower()
        if not text:
            return ("qa", False)

        if self._looks_like_workspace_exists_check_request(text, lowered):
            return ("workspace_qa", False)

        if self._looks_like_workspace_file_explanation_request(text, lowered):
            return ("workspace_qa", False)

        if self._PATH_PATTERN.search(text):
            return ("agentic", False)

        forced_qa_keywords = (
            "qa 模式",
            "qa模式",
            "问答模式",
            "回到问答模式",
            "只做概念说明",
            "只做说明",
            "不要操作本地文件",
            "不要写本地",
        )
        if self._contains_any(text, lowered, forced_qa_keywords):
            return ("qa", False)

        direct_agentic_requests = (
            "分析并修改",
            "分析并修复",
            "请修改",
            "请修复",
            "修改当前项目",
            "修复当前项目",
            "改造当前项目",
            "补充最小测试",
            "帮我运行",
            "帮我启动",
            "帮我执行",
            "运行一下",
            "启动一下",
            "执行一下",
            "跑一下",
            "直接运行",
            "现在就运行",
            "替我运行",
        )
        if self._contains_any(text, lowered, direct_agentic_requests):
            return ("agentic", False)

        explicit_task_mode = ("agent 模式", "agent模式", "任务模式")
        if self._contains_any(text, lowered, explicit_task_mode):
            return ("agentic", False)

        if _looks_like_project_review_request(text, lowered):
            return ("agentic", False)

        if self._looks_like_workspace_qa_request(text, lowered):
            return ("workspace_qa", False)

        readonly_project_keywords = (
            "不修改文件",
            "不改文件",
            "先别动文件",
            "不要大改",
            "列出关键文件",
            "查找文件",
            "找到实现",
            "定位实现",
            "会话存储实现",
            "模式判定逻辑",
        )
        if self._contains_any(text, lowered, readonly_project_keywords):
            return ("agentic", False)

        project_ops = (
            "pytest",
            "unittest",
            "ci",
            "构建",
            "打包",
            "运行测试",
            "跑测试",
            "build",
            "test",
        )
        if self._contains_any(text, lowered, project_ops):
            return ("agentic", False)
        if ("测试" in text or "test" in lowered) and self._contains_any(
            text, lowered, ("跑", "运行", "挂了", "失败", "总结", "summary")
        ):
            return ("agentic", False)

        code_change_markers = ("新增", "添加", "修改", "修复", "重构", "删除", "补丁", "diff", "patch")
        code_targets = (
            "接口",
            "api",
            "文件",
            "代码",
            "bug",
            "测试",
            "文案",
            "打印",
            "功能",
            "逻辑",
            "登录",
            "报错",
            "会话",
            "健康检查",
            "配置",
        )
        if self._contains_any(text, lowered, code_change_markers) and self._contains_any(text, lowered, code_targets):
            return ("agentic", False)

        if self._contains_any(text, lowered, self._OPERATIONAL_MARKERS):
            return ("agentic", False)

        qa_keywords = (
            "算法",
            "数据结构",
            "时间复杂度",
            "空间复杂度",
            "语法",
            "标准库",
            "怎么实现",
            "如何实现",
            "解释",
            "讲解",
            "原理",
            "区别",
            "差别",
            "不同",
            "是什么",
            "为什么",
            "如何证明",
            "解释这段代码",
            "写一个示例",
            "示例",
            "示例代码",
            "python 示例",
            "python",
            "给一个例子",
            "示例函数",
            "example",
            "time complexity",
            "space complexity",
            "syntax",
            "stdlib",
            "建议",
            "思路",
            "概念",
            "对比",
            "更简洁",
            "更短",
            "更口语",
            "格式",
            "先问我",
            "澄清",
            "指什么",
            "何意",
            "是啥",
            "啥是",
            "意味着",
        )
        if self._contains_any(text, lowered, qa_keywords):
            return ("qa", False)

        if text.endswith(("?", "？")):
            return ("qa", False)

        if self._contains_any(text, lowered, self._PROJECT_SCOPE_MARKERS):
            return ("qa", True)

        return ("qa", False)

    def infer(self, user_content: str) -> str:
        mode, _ = self.infer_with_meta(user_content)
        return mode

    def _looks_like_workspace_qa_request(self, text: str, lowered: str) -> bool:
        if self._contains_any(text, lowered, self._WORKSPACE_QA_EXECUTION_INTENT_MARKERS):
            return False
        asks_for_explanation = self._contains_any(
            text,
            lowered,
            self._PROJECT_OVERVIEW_MARKERS
            + self._WORKSPACE_QA_EXPLANATION_MARKERS
            + self._WORKSPACE_QA_RUN_OR_USAGE_MARKERS,
        )
        mentions_workspace_target = self._contains_any(
            text,
            lowered,
            self._PROJECT_SCOPE_MARKERS + self._WORKSPACE_QA_TARGET_MARKERS,
        )
        write_or_run_markers = (
            "修改",
            "修复",
            "重构",
            "删除",
            "补丁",
            "patch",
            "diff",
            "跑测试",
            "运行测试",
            "pytest",
            "build",
            "构建",
            "打包",
            "review",
            "代码审查",
            "全面阅读",
            "全面检查",
        )
        action_markers = (
            "找到",
            "查找",
            "定位",
            "列出关键文件",
            "关键文件",
            "实现",
            "怎么验证",
            "如何验证",
            "验证",
            "入口",
            "调用链",
        )
        return asks_for_explanation and mentions_workspace_target and not self._contains_any(
            text,
            lowered,
            write_or_run_markers,
        ) and not self._contains_any(
            text,
            lowered,
            action_markers,
        )

    def _looks_like_workspace_exists_check_request(self, text: str, lowered: str) -> bool:
        mentions_workspace_target = self._contains_any(
            text,
            lowered,
            self._PROJECT_SCOPE_MARKERS + self._WORKSPACE_QA_TARGET_MARKERS,
        )
        existence_markers = (
            "是否已存在",
            "是否存在",
            "有没有",
            "确认",
            "看下有没有",
            "确认下",
            "是不是已经有",
            "是否已经有",
        )
        target_markers = (
            "api",
            "接口",
            "路由",
            "功能",
            "文件",
            "命令",
            "端点",
        )
        readonly_markers = (
            "不修改文件",
            "不改文件",
            "先别动文件",
            "不要修改文件",
            "不要改文件",
        )
        write_or_run_markers = (
            "修改",
            "修复",
            "新增",
            "添加",
            "重构",
            "删除",
            "补丁",
            "patch",
            "diff",
            "跑测试",
            "运行测试",
            "pytest",
            "build",
            "构建",
            "打包",
            "执行",
            "运行",
        )
        has_readonly_guard = self._contains_any(text, lowered, readonly_markers)
        return (
            mentions_workspace_target
            and self._contains_any(text, lowered, existence_markers)
            and self._contains_any(text, lowered, target_markers)
            and (has_readonly_guard or not self._contains_any(text, lowered, write_or_run_markers))
        )

    def _looks_like_workspace_file_explanation_request(self, text: str, lowered: str) -> bool:
        if not self._PATH_PATTERN.search(text):
            return False
        explanation_markers = (
            "分析",
            "介绍",
            "说明",
            "讲讲",
            "说说",
            "总结",
            "在做什么",
            "做什么",
            "什么意思",
            "作用",
        )
        write_or_run_markers = (
            "修改",
            "修复",
            "新增",
            "添加",
            "重构",
            "删除",
            "补丁",
            "patch",
            "diff",
            "跑测试",
            "运行测试",
            "pytest",
            "构建",
            "build",
            "执行",
            "运行",
        )
        return self._contains_any(text, lowered, explanation_markers) and not self._contains_any(
            text,
            lowered,
            write_or_run_markers,
        )

    @staticmethod
    def _contains_any(text: str, lowered: str, keywords: tuple[str, ...]) -> bool:
        return _contains_any_keyword(text, lowered, keywords)


class SafetyGuard:
    def review(self, user_content: str) -> SafetyRefusal | None:
        text = str(user_content or "").strip()
        lowered = text.lower()

        if self._is_delete_tests_request(text, lowered):
            return SafetyRefusal(
                reason="delete_tests_to_hide_ci",
                answer=(
                    "我不能帮助通过删除整个仓库的测试来掩盖 CI 问题。\n"
                    "更安全的替代做法是：\n"
                    "1. 先定位失败用例、报错栈和最近改动。\n"
                    "2. 如果是 flaky 测试，先隔离或临时跳过，并写明原因与回滚条件。\n"
                    "3. 修复根因后再恢复稳定校验。\n"
                    "如果你愿意，我可以继续帮你定位是哪几个测试在失败，以及应该如何最小化修复。"
                ),
            )

        if self._is_export_passwords_request(text, lowered):
            return SafetyRefusal(
                reason="credential_theft",
                answer=(
                    "我不能帮助导出、窃取或外传浏览器保存的密码。\n"
                    "如果你的目标是做安全排查，可以改为：\n"
                    "1. 检查本机是否存在恶意扩展或可疑进程。\n"
                    "2. 审计浏览器密码管理器的导入导出权限与访问日志。\n"
                    "3. 指导用户使用官方恢复、重置或安全检查流程。"
                ),
            )

        if self._is_log_secrets_request(text, lowered):
            return SafetyRefusal(
                reason="log_plaintext_secrets",
                answer=(
                    "我不能帮助把生产环境账号、密码或其他敏感凭证明文写进日志。\n"
                    "更安全的替代做法是：\n"
                    "1. 只记录连接是否成功、目标库名、错误码和请求链路 ID。\n"
                    "2. 对账号做部分掩码，对密钥只打印来源而不打印值。\n"
                    "3. 结合密钥管理系统、临时调试开关和审计日志来排查问题。"
                ),
            )

        if self._is_bypass_sso_request(text, lowered):
            return SafetyRefusal(
                reason="bypass_auth",
                answer=(
                    "我不能帮助绕过公司 SSO、批量登录或规避认证控制。\n"
                    "合规替代方案是：\n"
                    "1. 联系管理员开测试账号、测试租户或临时白名单。\n"
                    "2. 使用官方 OAuth / SSO 集成方式或测试环境进行联调。\n"
                    "3. 如果是排障，我可以帮你梳理合法的登录链路和排查步骤。"
                ),
            )

        return None

    def _is_delete_tests_request(self, text: str, lowered: str) -> bool:
        delete_markers = ("删掉", "删除", "去掉", "移除")
        test_markers = ("测试", "tests", "test")
        scope_markers = ("仓库", "项目", "repo", "repository")
        return (
            self._contains_any(text, lowered, delete_markers)
            and self._contains_any(text, lowered, test_markers)
            and self._contains_any(text, lowered, scope_markers)
        )

    def _is_export_passwords_request(self, text: str, lowered: str) -> bool:
        password_markers = ("密码", "password", "credentials")
        browser_markers = ("浏览器", "browser", "chrome", "edge", "firefox")
        export_markers = ("导出", "导出来", "发给我", "发送", "send", "导出脚本")
        return (
            self._contains_any(text, lowered, password_markers)
            and self._contains_any(text, lowered, browser_markers)
            and self._contains_any(text, lowered, export_markers)
        )

    def _is_log_secrets_request(self, text: str, lowered: str) -> bool:
        db_markers = ("生产数据库", "数据库", "db", "database")
        secret_markers = ("账号密码", "密码", "secret", "token", "凭证")
        log_markers = ("日志", "log", "打印")
        return (
            self._contains_any(text, lowered, db_markers)
            and self._contains_any(text, lowered, secret_markers)
            and self._contains_any(text, lowered, log_markers)
        )

    def _is_bypass_sso_request(self, text: str, lowered: str) -> bool:
        bypass_markers = ("绕过", "bypass", "跳过认证", "免登录")
        sso_markers = ("sso", "单点登录", "认证")
        automation_markers = ("脚本", "批量", "自动登录", "登录")
        return (
            self._contains_any(text, lowered, bypass_markers)
            and self._contains_any(text, lowered, sso_markers)
            and self._contains_any(text, lowered, automation_markers)
        )

    @staticmethod
    def _contains_any(text: str, lowered: str, keywords: tuple[str, ...]) -> bool:
        for keyword in keywords:
            if keyword in text or keyword in lowered:
                return True
        return False


class ClarificationGuard:
    def review(self, user_content: str) -> ClarificationPrompt | None:
        text = str(user_content or "").strip()
        lowered = text.lower()
        if not text:
            return None
        if not self._is_ambiguous_troubleshooting_request(text, lowered):
            return None
        error_code = self._normalize_error_code(text)
        return ClarificationPrompt(answer=self._build_troubleshooting_clarification(error_code))

    def _is_ambiguous_troubleshooting_request(self, text: str, lowered: str) -> bool:
        troubleshooting_markers = (
            "排查",
            "排障",
            "定位",
            "思路",
            "先帮我看",
            "先帮我排查",
            "先看下",
            "troubleshoot",
            "debug",
        )
        incident_markers = (
            "接口",
            "api",
            "服务",
            "请求",
            "报错",
            "错误",
            "异常",
            "500",
            "50 0",
            "5 0 0",
            "502",
            "503",
            "504",
            "超时",
            "timeout",
            "偶发",
            "失败",
        )
        context_markers = (
            "日志",
            "log",
            "报错栈",
            "堆栈",
            "stack",
            "trace",
            "复现",
            "环境",
            "路径",
            "路由",
            "接口名",
            "endpoint",
            "请求体",
            "响应体",
            "response",
            "参数",
            "header",
            "host",
            "实例",
        )
        return self._contains_any(text, lowered, troubleshooting_markers) and self._contains_any(
            text, lowered, incident_markers
        ) and not self._contains_any(text, lowered, context_markers)

    def _normalize_error_code(self, text: str) -> str | None:
        collapsed = re.sub(r"\s+", "", text)
        if "500" in collapsed:
            return "500"
        if "502" in collapsed:
            return "502"
        if "503" in collapsed:
            return "503"
        if "504" in collapsed:
            return "504"
        return None

    def _build_troubleshooting_clarification(self, error_code: str | None) -> str:
        opening = (
            f"我先按“{error_code}”理解。为了避免拍脑袋判断，请先补充这 3 点："
            if error_code
            else "我先按这是一次接口故障排查来理解。为了避免拍脑袋判断，请先补充这 3 点："
        )
        return (
            f"{opening}\n"
            "1. 是哪个接口或路径，发生在什么环境？\n"
            "2. 当时的应用日志、网关日志或报错栈里最关键的一段是什么？\n"
            "3. 复现条件是什么：是否和特定参数、并发、用户状态或时间段有关？\n"
            "你贴出这些信息后，我再按现象、日志、依赖、数据和并发几个方向帮你缩小范围。"
        )

    @staticmethod
    def _contains_any(text: str, lowered: str, keywords: tuple[str, ...]) -> bool:
        for keyword in keywords:
            if keyword in text or keyword in lowered:
                return True
        return False


class AssistantResponseRenderer:
    _WRITE_MARKERS = ("修改", "修复", "补丁", "写入", "落地", "apply_patch", "保存", "编辑")
    _VERIFY_EXECUTION_MARKERS = ("运行测试", "pytest", "回归", "测试命令", "验证改动", "执行测试", "run tests")
    _PLAN_ONLY_MARKERS = ("说明验证方法", "给出验证方法", "验证思路", "排查思路", "总结", "定位", "审查", "分析")

    def evaluate_task_result(self, task: TaskItem | None, turn: AgenticTurnResult) -> dict[str, Any]:
        answer = str(turn.answer or "").strip()
        answer_ok = bool(answer)
        tool_trace = list(turn.tool_trace or [])
        ok_count = sum(1 for item in tool_trace if item.get("status") == "ok")
        error_count = sum(1 for item in tool_trace if item.get("status") == "error")
        write_ok = any(
            item.get("tool") in {"apply_patch_tool", "write_file_tool"} and item.get("status") == "ok"
            for item in tool_trace
        )
        test_ok = any(
            item.get("tool") in {"run_command_tool", "test_tool"} and item.get("status") == "ok"
            for item in tool_trace
        )
        patch_suggested = self._contains_patch(answer)
        task_text = self._task_text(task)
        requires_write = self._contains_any(task_text, self._WRITE_MARKERS) and not self._contains_any(
            task_text, self._PLAN_ONLY_MARKERS
        )
        requires_executed_verification = self._contains_any(task_text, self._VERIFY_EXECUTION_MARKERS)

        succeeded = False
        reason = "missing_result"
        if requires_write:
            succeeded = write_ok or patch_suggested
            reason = "write_applied" if write_ok else "patch_guidance" if patch_suggested else "write_not_confirmed"
        elif requires_executed_verification:
            succeeded = test_ok
            reason = "tests_ran" if test_ok else "verification_not_run"
        elif tool_trace:
            succeeded = answer_ok and ok_count > 0
            reason = "tool_backed_answer" if succeeded else "no_successful_tool"
        else:
            succeeded = answer_ok
            reason = "answer_only" if succeeded else "missing_answer"

        return {
            "succeeded": succeeded,
            "reason": reason,
            "tool_success_count": ok_count,
            "tool_error_count": error_count,
            "used_write_tool": write_ok,
            "used_test_tool": test_ok,
            "patch_suggested": patch_suggested,
        }

    def task_succeeded(self, task: TaskItem | None, turn: AgenticTurnResult) -> bool:
        return bool(self.evaluate_task_result(task, turn)["succeeded"])

    def summarize_task_result(self, turn: AgenticTurnResult, evaluation: dict[str, Any] | None = None) -> str:
        evaluation = evaluation or self.evaluate_task_result(None, turn)
        answer = turn.answer.strip()
        if answer:
            return answer[:240]
        if turn.tool_trace:
            ok_count = int(evaluation.get("tool_success_count", 0))
            err_count = int(evaluation.get("tool_error_count", 0))
            outcome = "任务成立" if evaluation.get("succeeded") else "任务未完成"
            return f"完成了 {len(turn.tool_trace)} 次工具调用，其中 {ok_count} 次成功、{err_count} 次失败。{outcome}。"
        return "本步骤没有拿到有效结果。"

    def compose_step_plan(self, board: TaskBoard) -> str:
        tasks = board.ordered_tasks()
        if not tasks:
            return "我会先整理步骤，再继续处理。"
        lines = ["我会按下面这些步骤推进："]
        for index, task in enumerate(tasks, start=1):
            detail = self._clean_sentence(task.description or task.acceptance or task.title)
            lines.append(f"{index}. {task.title}：{detail}")
        return "\n".join(lines)

    def compose_step_update(self, task: TaskItem, *, step_index: int) -> str:
        detail = self._clean_sentence(task.summary or task.acceptance or task.description or task.title)
        if task.status == "done":
            return f"步骤 {step_index} 已完成：{task.title}。{detail}"
        if task.status == "failed":
            return f"步骤 {step_index} 未完成：{task.title}。{detail}"
        return f"步骤 {step_index} 正在处理：{task.title}。"

    def compose_final_answer(
        self,
        board: TaskBoard,
        last_answer: str,
        last_test_summary: dict[str, Any] | None,
    ) -> str:
        tasks = board.ordered_tasks()
        finished = [task for task in tasks if task.status == "done"]
        failed = [task for task in tasks if task.status == "failed"]
        parts: list[str] = []

        if tasks:
            if failed:
                done_text = "、".join(task.title for task in finished) if finished else "前置步骤"
                failed_text = "；".join(
                    f"{task.title}：{self._clean_sentence(task.summary or task.acceptance or task.description or task.title)}"
                    for task in failed
                )
                parts.append(f"这次任务已经按步骤推进。已完成的部分包括 {done_text}；仍需继续处理的是 {failed_text}。")
            else:
                step_titles = "、".join(task.title for task in tasks)
                parts.append(f"这次任务已经按“{step_titles}”这些步骤处理完成。")

        answer = str(last_answer or "").strip()
        if answer:
            parts.append(answer)
        elif tasks:
            summaries = [
                self._clean_sentence(task.summary or task.acceptance or task.description or task.title)
                for task in tasks
            ]
            parts.append(f"关键结果是：{'；'.join(summaries)}。")

        if last_test_summary:
            status_text = "通过" if last_test_summary.get("passed") else "失败"
            command = str(last_test_summary.get("command") or "").strip()
            duration_ms = int(last_test_summary.get("duration_ms") or 0)
            if command:
                parts.append(f"最后一次测试{status_text}，耗时 {duration_ms} ms，命令是 {command}。")
            else:
                parts.append(f"最后一次测试{status_text}，耗时 {duration_ms} ms。")

        return "\n\n".join(part for part in parts if part).strip()

    def chunk_text(self, text: str, *, chunk_size: int = 160) -> list[str]:
        return [str(text or "")]

    def _clean_sentence(self, text: str) -> str:
        return str(text or "").strip().rstrip("。；;")

    def _task_text(self, task: TaskItem | None) -> str:
        if task is None:
            return ""
        return " ".join(
            str(part or "")
            for part in (task.title, task.description, task.acceptance)
        ).lower()

    def _contains_any(self, text: str, markers: tuple[str, ...]) -> bool:
        return any(marker.lower() in text for marker in markers)

    def _contains_patch(self, text: str) -> bool:
        body = str(text or "")
        return "--- " in body or "@@" in body or "diff" in body.lower()


@dataclass
class AgenticExecutionResult:
    snapshot: dict[str, Any]
    board: TaskBoard
    task_results: list[dict[str, Any]]
    combined_tool_trace: list[dict[str, Any]]
    last_nonempty_answer: str


class SessionTestCoordinator:
    def __init__(
        self,
        *,
        session_store: SessionStore,
        require_workspace_root: Callable[[dict[str, Any]], str],
    ) -> None:
        self.session_store = session_store
        self.require_workspace_root = require_workspace_root

    def should_auto_run_tests(self, settings: dict[str, Any], tool_trace: list[dict[str, Any]]) -> bool:
        if not settings.get("auto_run_tests"):
            return False
        if not settings.get("allow_shell"):
            return False
        if not str(settings.get("test_command") or "").strip():
            return False
        return any(
            item.get("tool") in {"apply_patch_tool", "write_file_tool"} and item.get("status") == "ok"
            for item in tool_trace
        )

    def run_for_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        emit: EventCallback | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        settings = normalize_session_settings(snapshot.get("settings"))
        command = str(settings.get("test_command") or "").strip()
        if not command:
            raise ValueError("当前会话未配置 test_command。")

        summary = normalize_test_summary(
            run_project_test_command(
                workspace_root=self.require_workspace_root(snapshot),
                command=command,
                allow_shell=bool(settings.get("allow_shell")),
            )
        )
        assert summary is not None
        snapshot["last_test_summary"] = summary
        updated = self.session_store.save_session(snapshot)
        if emit is not None:
            emit({"event": "test_summary", "data": summary})
        return updated, summary


class AgenticTaskCoordinator:
    def __init__(
        self,
        *,
        session_store: SessionStore,
        agent_factory: Callable[..., Any],
        renderer: AssistantResponseRenderer,
    ) -> None:
        self.session_store = session_store
        self.agent_factory = agent_factory
        self.renderer = renderer

    def execute(
        self,
        *,
        snapshot: dict[str, Any],
        user_content: str,
        history_before_turn: list[dict[str, str]],
        memory: Any,
        workspace_root: str,
        settings: dict[str, Any],
        emit: EventCallback,
        cancel_event: Any | None = None,
    ) -> AgenticExecutionResult:
        agent = self.agent_factory(
            workspace_root,
            memory=memory,
            top_k=5,
            force_reindex=False,
            allow_write=bool(settings.get("allow_write")),
            allow_shell=bool(settings.get("allow_shell")),
            test_command=str(settings.get("test_command") or ""),
        )
        board = self._build_board(
            agent=agent,
            user_content=user_content,
            history_before_turn=history_before_turn,
        )
        snapshot["tasks"] = board.to_dicts()
        snapshot = self.session_store.save_session(snapshot)
        emit({"event": "task_board", "data": snapshot["tasks"]})
        emit({"event": "assistant_delta", "data": {"content": f"{self.renderer.compose_step_plan(board)}\n\n"}})

        combined_tool_trace: list[dict[str, Any]] = []
        task_results: list[dict[str, Any]] = []
        last_nonempty_answer = ""

        ordered_tasks = board.ordered_tasks()

        for step_index, task in enumerate(ordered_tasks, start=1):
            self._ensure_not_cancelled(cancel_event)
            if self._dependency_failed(task, board):
                failed_task = board.mark_failed(task.id, summary="依赖任务失败，当前任务未执行。")
                snapshot = self._save_board(snapshot, board)
                task_result = normalize_task_results(
                    [
                        {
                            "task_id": failed_task.id,
                            "title": failed_task.title,
                            "status": failed_task.status,
                            "summary": failed_task.summary,
                            "answer": "",
                            "tool_trace": [],
                            "task_outcome": "blocked",
                            "task_reason": "dependency_failed",
                            "tool_success_count": 0,
                            "tool_error_count": 0,
                        }
                    ]
                )[0]
                task_results.append(task_result)
                emit({"event": "task_update", "data": task_result})
                emit(
                    {
                        "event": "assistant_delta",
                        "data": {"content": f"{self.renderer.compose_step_update(failed_task, step_index=step_index)}\n\n"},
                    }
                )
                continue

            running_task = board.mark_in_progress(task.id)
            snapshot = self._save_board(snapshot, board)
            emit({"event": "task_update", "data": running_task.to_dict()})

            task_prompt = self._build_task_prompt(
                original_goal=user_content,
                task=running_task,
                board=board,
                workspace_root=workspace_root,
                settings=settings,
                prior_tool_trace=combined_tool_trace,
            )
            turn = agent.run_agentic(
                task_prompt,
                max_turns=int(settings.get("max_turns", 8)),
                workspace_root=workspace_root,
                persist_memory=False,
                cancel_event=cancel_event,
            )
            self._ensure_not_cancelled(cancel_event)

            tool_trace = normalize_tool_trace(turn.tool_trace)
            combined_tool_trace.extend(tool_trace)
            if turn.answer.strip():
                candidate_answer = turn.answer.strip()
                if self._prefer_as_final_answer(candidate_answer, last_nonempty_answer):
                    last_nonempty_answer = candidate_answer

            evaluation = self.renderer.evaluate_task_result(running_task, turn)
            summary = self.renderer.summarize_task_result(turn, evaluation)
            if evaluation["succeeded"]:
                final_task = board.mark_done(task.id, summary=summary)
            else:
                final_task = board.mark_failed(task.id, summary=summary)

            snapshot = self._save_board(snapshot, board)
            task_result = normalize_task_results(
                [
                    {
                        "task_id": final_task.id,
                        "title": final_task.title,
                        "status": final_task.status,
                        "summary": final_task.summary,
                        "answer": turn.answer,
                        "tool_trace": tool_trace,
                        "task_outcome": "completed" if evaluation["succeeded"] else "incomplete",
                        "task_reason": str(evaluation["reason"]),
                        "tool_success_count": int(evaluation["tool_success_count"]),
                        "tool_error_count": int(evaluation["tool_error_count"]),
                    }
                ]
            )[0]
            task_results.append(task_result)
            emit({"event": "task_update", "data": task_result})
            emit(
                {
                    "event": "assistant_delta",
                    "data": {"content": f"{self.renderer.compose_step_update(final_task, step_index=step_index)}\n\n"},
                }
            )

        return AgenticExecutionResult(
            snapshot=snapshot,
            board=board,
            task_results=task_results,
            combined_tool_trace=combined_tool_trace,
            last_nonempty_answer=last_nonempty_answer,
        )

    def _save_board(self, snapshot: dict[str, Any], board: TaskBoard) -> dict[str, Any]:
        snapshot["tasks"] = board.to_dicts()
        return self.session_store.save_session(snapshot)

    def _build_board(
        self,
        *,
        agent: Any,
        user_content: str,
        history_before_turn: list[dict[str, str]],
    ) -> TaskBoard:
        if self._looks_like_review_request(user_content):
            return self._build_review_task_board()
        if self._looks_like_readonly_analysis_request(user_content):
            return TaskBoard.from_dicts(
                [
                    {
                        "id": "t1",
                        "title": "定位相关文件",
                        "description": "在当前项目中定位与用户问题最相关的文件、模块或入口。",
                        "depends_on": [],
                        "status": "pending",
                        "acceptance": "列出关键文件路径，并说明每个文件为什么相关。",
                    },
                    {
                        "id": "t2",
                        "title": "总结当前实现",
                        "description": "阅读关键文件并提炼当前实现逻辑、约束和风险点。",
                        "depends_on": ["t1"],
                        "status": "pending",
                        "acceptance": "给出与用户问题直接相关的实现说明，不扩展无关改动。",
                    },
                    {
                        "id": "t3",
                        "title": "说明验证方法",
                        "description": "基于现有实现，给出最小验证步骤或后续排查建议。",
                        "depends_on": ["t2"],
                        "status": "pending",
                        "acceptance": "输出简洁可执行的验证方法或下一步建议。",
                    },
                ]
            )
        return TaskBoard.from_dicts(agent.planner.make_task_board(user_content, history_before_turn))

    def _dependency_failed(self, task: TaskItem, board: TaskBoard) -> bool:
        return any(board.get(dep_id).status == "failed" for dep_id in task.depends_on)

    def _looks_like_readonly_analysis_request(self, user_content: str) -> bool:
        text = str(user_content or "").strip()
        lowered = text.lower()
        readonly_markers = (
            "不修改文件",
            "不改文件",
            "先别动文件",
            "不要大改",
            "只做说明",
            "仅分析",
            "先查因",
            "列出关键文件",
            "说明你会怎么验证",
            "说明怎么验证",
            "review",
            "代码审查",
            "全面阅读",
            "全面检查",
        )
        write_or_run_markers = (
            "跑测试",
            "运行测试",
            "pytest",
            "写入",
            "应用补丁",
            "修改文件",
            "改完",
            "新增测试",
            "创建提交",
            "build",
            "构建",
            "打包",
        )
        return any(marker in text or marker in lowered for marker in readonly_markers) and not any(
            marker in text or marker in lowered for marker in write_or_run_markers
        )

    def _looks_like_review_request(self, user_content: str) -> bool:
        text = str(user_content or "").strip()
        lowered = text.lower()
        write_or_run_markers = (
            "跑测试",
            "运行测试",
            "pytest",
            "写入",
            "应用补丁",
            "修改文件",
            "改完",
            "新增测试",
            "创建提交",
            "build",
            "构建",
            "打包",
        )
        return _looks_like_project_review_request(text, lowered) and not any(
            marker in text or marker in lowered for marker in write_or_run_markers
        )

    def _build_review_task_board(self) -> TaskBoard:
        return TaskBoard.from_dicts(
            [
                {
                    "id": "t1",
                    "title": "定位关键后端入口",
                    "description": "先确认后端入口、服务层、状态流转和关键模块边界。",
                    "depends_on": [],
                    "status": "pending",
                    "acceptance": "列出最值得深入阅读的关键文件，并说明各自职责。",
                },
                {
                    "id": "t2",
                    "title": "审查输入输出与状态流转",
                    "description": "检查请求输入、响应输出、错误处理、权限边界和步骤衔接是否稳定。",
                    "depends_on": ["t1"],
                    "status": "pending",
                    "acceptance": "指出会直接影响体验的实现问题，并给出原因。",
                },
                {
                    "id": "t3",
                    "title": "汇总结论与改进建议",
                    "description": "按影响面整理高优先级问题、改进方向和验证思路。",
                    "depends_on": ["t2"],
                    "status": "pending",
                    "acceptance": "输出可执行、可排序的结论，而不是泛化建议。",
                },
            ]
        )

    def _build_task_prompt(
        self,
        *,
        original_goal: str,
        task: TaskItem,
        board: TaskBoard,
        workspace_root: str,
        settings: dict[str, Any],
        prior_tool_trace: list[dict[str, Any]],
    ) -> str:
        completed = board.completed_summaries()
        completed_text = "\n".join(f"- {line}" for line in completed) if completed else "- 暂无"
        prior_tool_text = self._format_prior_tool_context(prior_tool_trace)
        test_command = str(settings.get("test_command") or "").strip()
        return (
            f"原始用户目标：{original_goal}\n\n"
            f"当前任务：{task.title}\n"
            f"任务描述：{task.description}\n"
            f"验收标准：{task.acceptance}\n\n"
            f"已完成任务摘要：\n{completed_text}\n\n"
            f"前序关键工具结果：\n{prior_tool_text}\n\n"
            f"工作区根目录：{workspace_root}\n"
            f"写权限：{'开启' if settings.get('allow_write') else '关闭'}\n"
            f"命令执行权限：{'开启' if settings.get('allow_shell') else '关闭'}\n\n"
            f"测试命令：{test_command or '未配置'}\n\n"
            "请围绕当前任务行动；若前序结果已经确认文件路径、函数名、报错或测试命令，请直接复用，"
            "不要无意义地重新从零检索。若权限不足，输出结构化补丁建议或手动编辑步骤。\n\n"
            "面向用户的最终说明优先用连贯自然段组织；内部步骤已在任务板体现，无需再把正文写成第二份任务分解清单，"
            "除非用户明确要求分点或验收条目必须逐条列出。"
        )

    def _format_prior_tool_context(self, tool_trace: list[dict[str, Any]], *, max_items: int = 6) -> str:
        relevant = [
            item
            for item in tool_trace
            if str(item.get("tool") or "").strip() and str(item.get("output") or "").strip()
        ]
        if not relevant:
            return "- 暂无"
        lines: list[str] = []
        for item in relevant[-max_items:]:
            tool = str(item.get("tool") or "unknown_tool")
            status = str(item.get("status") or "unknown")
            preview = self._preview_tool_output(item.get("output"))
            lines.append(f"- {tool} [{status}] {preview}")
        return "\n".join(lines)

    def _preview_tool_output(self, output: Any, *, max_chars: int = 320) -> str:
        text = str(output or "")
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 16] + "... [truncated]"

    def _prefer_as_final_answer(self, candidate: str, current: str) -> bool:
        if not current:
            return True
        candidate_has_patch = self._contains_patch(candidate)
        current_has_patch = self._contains_patch(current)
        if candidate_has_patch != current_has_patch:
            return candidate_has_patch
        return len(candidate) >= len(current)

    def _contains_patch(self, text: str) -> bool:
        body = str(text or "")
        return "--- " in body or "@@" in body or "diff" in body.lower()

    def _ensure_not_cancelled(self, cancel_event: Any | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise StreamCancelled("stream cancelled")
