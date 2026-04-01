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


class TurnModeDecider:
    _PATH_PATTERN = re.compile(r"(?<!\S)(?:[\w.-]+[\\/])+[\w.-]+")

    def infer(self, user_content: str) -> str:
        text = str(user_content or "").strip()
        lowered = text.lower()

        if self._PATH_PATTERN.search(text):
            return "agentic"

        direct_agentic_requests = (
            "分析并修改",
            "分析并修复",
            "请修改",
            "请修复",
            "修改当前项目",
            "修复当前项目",
            "改造当前项目",
            "补充最小测试",
        )
        if self._contains_any(text, lowered, direct_agentic_requests):
            return "agentic"

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
            return "qa"

        forced_agentic_keywords = (
            "agent 模式",
            "agent模式",
            "任务模式",
            "这个项目",
            "当前项目",
            "当前仓库",
            "代码库",
            "仓库",
            "workspace",
            "workspace_root",
            "在这个项目里",
            "在当前仓库里",
            "集成到当前项目",
        )
        if self._contains_any(text, lowered, forced_agentic_keywords):
            return "agentic"

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
            return "agentic"

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
            return "agentic"
        if ("测试" in text or "test" in lowered) and self._contains_any(
            text, lowered, ("跑", "运行", "挂了", "失败", "总结", "summary")
        ):
            return "agentic"

        code_change_markers = ("新增", "修改", "修复", "重构", "删除", "补丁", "diff", "patch")
        code_targets = (
            "接口",
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
            return "agentic"

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
        )
        if self._contains_any(text, lowered, qa_keywords):
            return "qa"

        if text.endswith(("?", "？")):
            return "qa"

        return "qa"

    @staticmethod
    def _contains_any(text: str, lowered: str, keywords: tuple[str, ...]) -> bool:
        for keyword in keywords:
            if keyword in text or keyword in lowered:
                return True
        return False


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
    def task_succeeded(self, turn: AgenticTurnResult) -> bool:
        answer_ok = bool(turn.answer.strip())
        if not turn.tool_trace:
            return answer_ok
        return answer_ok and any(item.get("status") == "ok" for item in turn.tool_trace)

    def summarize_task_result(self, turn: AgenticTurnResult) -> str:
        answer = turn.answer.strip()
        if answer:
            return answer[:240]
        if turn.tool_trace:
            ok_count = sum(1 for item in turn.tool_trace if item.get("status") == "ok")
            return f"完成 {len(turn.tool_trace)} 次工具调用，其中 {ok_count} 次成功。"
        return "未获得有效结果。"

    def compose_final_answer(
        self,
        board: TaskBoard,
        last_answer: str,
        last_test_summary: dict[str, Any] | None,
    ) -> str:
        lines: list[str] = ["任务执行结果："]
        for index, task in enumerate(board.ordered_tasks(), start=1):
            detail = task.summary or task.acceptance
            lines.append(f"{index}. {task.title} [{task.status}] {detail}")
        if last_test_summary:
            status_text = "通过" if last_test_summary.get("passed") else "失败"
            lines.append("")
            lines.append(
                "最近一次测试："
                f"{status_text}，耗时 {last_test_summary.get('duration_ms', 0)} ms，"
                f"命令 {last_test_summary.get('command', '')}"
            )
        if last_answer:
            lines.append("")
            lines.append("最终回复：")
            lines.append(last_answer)
        return "\n".join(lines).strip()

    def chunk_text(self, text: str, *, chunk_size: int = 160) -> list[str]:
        body = str(text or "")
        if not body:
            return [""]
        return [body[index : index + chunk_size] for index in range(0, len(body), chunk_size)]


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
        )
        board = self._build_board(
            agent=agent,
            user_content=user_content,
            history_before_turn=history_before_turn,
        )
        snapshot["tasks"] = board.to_dicts()
        snapshot = self.session_store.save_session(snapshot)
        emit({"event": "task_board", "data": snapshot["tasks"]})

        combined_tool_trace: list[dict[str, Any]] = []
        task_results: list[dict[str, Any]] = []
        last_nonempty_answer = ""

        for task in board.ordered_tasks():
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
                        }
                    ]
                )[0]
                task_results.append(task_result)
                emit({"event": "task_update", "data": task_result})
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
                last_nonempty_answer = turn.answer.strip()

            summary = self.renderer.summarize_task_result(turn)
            if self.renderer.task_succeeded(turn):
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
                    }
                ]
            )[0]
            task_results.append(task_result)
            emit({"event": "task_update", "data": task_result})
            emit(
                {
                    "event": "assistant_delta",
                    "data": {"content": f"[{final_task.title}] {final_task.status}: {final_task.summary}\n"},
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

    def _build_task_prompt(
        self,
        *,
        original_goal: str,
        task: TaskItem,
        board: TaskBoard,
        workspace_root: str,
        settings: dict[str, Any],
    ) -> str:
        completed = board.completed_summaries()
        completed_text = "\n".join(f"- {line}" for line in completed) if completed else "- 暂无"
        return (
            f"原始用户目标：{original_goal}\n\n"
            f"当前任务：{task.title}\n"
            f"任务描述：{task.description}\n"
            f"验收标准：{task.acceptance}\n\n"
            f"已完成任务摘要：\n{completed_text}\n\n"
            f"工作区根目录：{workspace_root}\n"
            f"写权限：{'开启' if settings.get('allow_write') else '关闭'}\n"
            f"命令执行权限：{'开启' if settings.get('allow_shell') else '关闭'}\n\n"
            "请围绕当前任务行动；若权限不足，输出结构化补丁建议或手动编辑步骤。"
        )

    def _ensure_not_cancelled(self, cancel_event: Any | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise StreamCancelled("stream cancelled")
