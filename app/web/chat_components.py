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


class TurnModeDecider:
    def infer(self, user_content: str) -> str:
        text = str(user_content or "").strip()
        lowered = text.lower()

        if re.search(r"(?<!\S)(?:[\w.-]+[\\/])+[\w.-]+", text):
            return "agentic"

        agentic_keywords = (
            "这个项目",
            "当前项目",
            "当前仓库",
            "代码库",
            "仓库",
            "workspace",
            "workspace_root",
            "在这个项目里",
            "在当前仓库里",
            "帮我修改",
            "帮我修复",
            "修复 bug",
            "实现功能",
            "增加接口",
            "新增接口",
            "重构",
            "添加测试",
            "增加测试",
            "集成到当前项目",
            "run tests",
            "run pytest",
        )
        if any(keyword in text or keyword in lowered for keyword in agentic_keywords):
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
            "示例代码",
            "python 示例",
            "给一个例子",
            "示例函数",
            "example",
            "time complexity",
            "space complexity",
            "syntax",
            "stdlib",
        )
        if any(keyword in text or keyword in lowered for keyword in qa_keywords):
            return "qa"

        project_ops = ("pytest", "unittest", "ci", "构建", "打包", "运行测试", "跑测试", "build")
        if any(keyword in text or keyword in lowered for keyword in project_ops):
            return "agentic"

        return "agentic"


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
        board = TaskBoard.from_dicts(agent.planner.make_task_board(user_content, history_before_turn))
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

    def _dependency_failed(self, task: TaskItem, board: TaskBoard) -> bool:
        return any(board.get(dep_id).status == "failed" for dep_id in task.depends_on)

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
