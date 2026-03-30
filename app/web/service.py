from __future__ import annotations

from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Callable, Iterator

from app.agent.agent import AgenticTurnResult
from app.agent.memory import ConversationMemory
from app.agent.task_board import TaskBoard, TaskItem
from app.main import create_agent_from_env
from app.web.session_store import (
    SessionStore,
    coerce_session_snapshot,
    derive_session_title,
    normalize_session_settings,
)
from app.web.test_runner import run_project_test_command

EventCallback = Callable[[dict[str, Any]], None]


class WebAgentService:
    def __init__(
        self,
        session_store: SessionStore | None = None,
        *,
        agent_factory: Callable[..., Any] = create_agent_from_env,
        repo_root: str | Path | None = None,
        outputs_dir: str | Path = "outputs",
    ) -> None:
        self.session_store = session_store or SessionStore()
        self.agent_factory = agent_factory
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
        self.outputs_dir = Path(outputs_dir).resolve()

    def resolve_workspace_root(self, raw_path: str) -> str:
        text = str(raw_path or "").strip()
        if not text:
            raise ValueError("workspace_root 不能为空。")
        candidate = Path(text)
        resolved = candidate.resolve() if candidate.is_absolute() else (self.repo_root / candidate).resolve()
        if not resolved.exists():
            raise ValueError(f"工作区不存在：{resolved}")
        if not resolved.is_dir():
            raise ValueError(f"工作区必须是目录：{resolved}")
        return str(resolved)

    def create_session(self, *, workspace_root: str, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        resolved_root = self.resolve_workspace_root(workspace_root)
        return self.session_store.create_session(
            workspace_root=resolved_root,
            settings=normalize_session_settings(settings),
        )

    def list_sessions(self) -> list[dict[str, Any]]:
        rows = self.session_store.list_sessions()
        return [
            {
                "session_id": row["session_id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "workspace_root": row["workspace_root"],
            }
            for row in rows
        ]

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self.session_store.get_session(session_id)

    def update_session(
        self,
        session_id: str,
        *,
        workspace_root: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_root = self.resolve_workspace_root(workspace_root) if workspace_root is not None else None
        normalized_settings = normalize_session_settings(settings) if settings is not None else None
        return self.session_store.update_session(
            session_id,
            workspace_root=resolved_root,
            settings=normalized_settings,
        )

    def chat(self, session_id: str, content: str) -> dict[str, Any]:
        events: list[dict[str, Any]] = []
        result = self._process_chat_turn(session_id, content, emit=events.append)
        result["events"] = events
        return result

    def stream_chat(self, session_id: str, content: str) -> Iterator[dict[str, Any]]:
        queue: Queue[dict[str, Any] | None] = Queue()

        def worker() -> None:
            try:
                self._process_chat_turn(session_id, content, emit=queue.put)
            except Exception as exc:  # noqa: BLE001
                queue.put(
                    {
                        "event": "error",
                        "data": {"message": str(exc)},
                    }
                )
            finally:
                queue.put(None)

        Thread(target=worker, daemon=True).start()
        while True:
            item = queue.get()
            if item is None:
                break
            yield item

    def run_session_tests(self, session_id: str, *, emit: EventCallback | None = None) -> dict[str, Any]:
        snapshot = self.session_store.get_session(session_id)
        updated, summary = self._run_tests_for_snapshot(snapshot, emit=emit)
        return {"session": updated, "summary": summary}

    def get_latest_eval_result(self) -> dict[str, Any]:
        preferred = self.outputs_dir / "eval_result.json"
        candidate = preferred if preferred.is_file() else None
        if candidate is None:
            json_files = sorted(self.outputs_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
            candidate = json_files[0] if json_files else None
        if candidate is None:
            return {"path": None, "payload": None}
        import json

        with candidate.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return {"path": str(candidate.resolve()), "payload": payload}

    def _process_chat_turn(self, session_id: str, content: str, *, emit: EventCallback) -> dict[str, Any]:
        user_content = str(content or "").strip()
        if not user_content:
            raise ValueError("消息内容不能为空。")

        snapshot = self.session_store.get_session(session_id)
        emit({"event": "session", "data": snapshot})

        settings = normalize_session_settings(snapshot.get("settings"))
        memory = ConversationMemory.from_snapshot(snapshot)
        agent = self.agent_factory(
            snapshot["workspace_root"],
            memory=memory,
            top_k=5,
            force_reindex=False,
            allow_write=bool(settings.get("allow_write")),
            allow_shell=bool(settings.get("allow_shell")),
        )

        board = TaskBoard.from_dicts(agent.planner.make_task_board(user_content, memory.get_messages()))
        snapshot["tasks"] = board.to_dicts()
        snapshot = self.session_store.save_session(snapshot)
        emit({"event": "task_board", "data": snapshot["tasks"]})

        combined_tool_trace: list[dict[str, Any]] = []
        task_results: list[dict[str, Any]] = []
        last_nonempty_answer = ""

        for task in board.ordered_tasks():
            if self._dependency_failed(task, board):
                failed_task = board.mark_failed(task.id, summary="依赖任务失败，当前任务未执行。")
                snapshot["tasks"] = board.to_dicts()
                snapshot = self.session_store.save_session(snapshot)
                task_result = {
                    "task_id": failed_task.id,
                    "title": failed_task.title,
                    "status": failed_task.status,
                    "summary": failed_task.summary,
                    "answer": "",
                    "tool_trace": [],
                }
                task_results.append(task_result)
                emit({"event": "task_update", "data": task_result})
                continue

            running_task = board.mark_in_progress(task.id)
            snapshot["tasks"] = board.to_dicts()
            snapshot = self.session_store.save_session(snapshot)
            emit({"event": "task_update", "data": running_task.to_dict()})

            task_prompt = self._build_task_prompt(
                original_goal=user_content,
                task=running_task,
                board=board,
                workspace_root=snapshot["workspace_root"],
                settings=settings,
            )
            turn = agent.run_agentic(
                task_prompt,
                max_turns=int(settings.get("max_turns", 8)),
                workspace_root=snapshot["workspace_root"],
                persist_memory=False,
            )
            combined_tool_trace.extend(turn.tool_trace)
            if turn.answer.strip():
                last_nonempty_answer = turn.answer.strip()

            summary = self._summarize_task_result(turn)
            if self._task_succeeded(turn):
                final_task = board.mark_done(task.id, summary=summary)
            else:
                final_task = board.mark_failed(task.id, summary=summary)

            snapshot["tasks"] = board.to_dicts()
            snapshot = self.session_store.save_session(snapshot)
            task_result = {
                "task_id": final_task.id,
                "title": final_task.title,
                "status": final_task.status,
                "summary": final_task.summary,
                "answer": turn.answer,
                "tool_trace": turn.tool_trace,
            }
            task_results.append(task_result)
            emit({"event": "task_update", "data": task_result})

        last_test_summary = snapshot.get("last_test_summary")
        if self._should_auto_run_tests(settings, combined_tool_trace):
            snapshot, last_test_summary = self._run_tests_for_snapshot(snapshot, emit=emit)

        final_answer = self._compose_final_answer(board, last_nonempty_answer, last_test_summary)
        for chunk in self._chunk_text(final_answer):
            emit({"event": "assistant_delta", "data": {"content": chunk}})

        memory.add_user_message(user_content)
        memory.add_assistant_message(final_answer)
        memory.add_turn_metadata(
            plan=[],
            tool_results=combined_tool_trace,
            recovery_applied=False,
            extra={
                "agentic": True,
                "tasks": board.to_dicts(),
                "task_results": task_results,
                "last_test_summary": last_test_summary,
            },
        )

        snapshot["messages"] = memory.get_messages()
        snapshot["turn_metadata"] = memory.get_turn_metadata()
        snapshot["tasks"] = board.to_dicts()
        snapshot["last_test_summary"] = last_test_summary
        snapshot["title"] = derive_session_title(snapshot["messages"])
        snapshot = self.session_store.save_session(snapshot)

        emit({"event": "assistant_final", "data": {"content": final_answer}})
        emit({"event": "session", "data": snapshot})
        return {
            "session": snapshot,
            "assistant": final_answer,
            "task_results": task_results,
            "last_test_summary": last_test_summary,
        }

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

    def _dependency_failed(self, task: TaskItem, board: TaskBoard) -> bool:
        for dep_id in task.depends_on:
            dep = board.get(dep_id)
            if dep.status == "failed":
                return True
        return False

    def _task_succeeded(self, turn: AgenticTurnResult) -> bool:
        answer_ok = bool(turn.answer.strip())
        if not turn.tool_trace:
            return answer_ok
        return answer_ok and any(item.get("status") == "ok" for item in turn.tool_trace)

    def _summarize_task_result(self, turn: AgenticTurnResult) -> str:
        answer = turn.answer.strip()
        if answer:
            return answer[:240]
        if turn.tool_trace:
            ok_count = sum(1 for item in turn.tool_trace if item.get("status") == "ok")
            return f"完成 {len(turn.tool_trace)} 次工具调用，其中 {ok_count} 次成功。"
        return "未获得有效结果。"

    def _should_auto_run_tests(self, settings: dict[str, Any], tool_trace: list[dict[str, Any]]) -> bool:
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

    def _run_tests_for_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        emit: EventCallback | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        settings = normalize_session_settings(snapshot.get("settings"))
        command = str(settings.get("test_command") or "").strip()
        if not command:
            raise ValueError("当前会话未配置 test_command。")
        summary = run_project_test_command(
            workspace_root=str(snapshot.get("workspace_root") or ""),
            command=command,
            allow_shell=bool(settings.get("allow_shell")),
        )
        snapshot["last_test_summary"] = summary
        updated = self.session_store.save_session(snapshot)
        if emit is not None:
            emit({"event": "test_summary", "data": summary})
        return updated, summary

    def _compose_final_answer(
        self,
        board: TaskBoard,
        last_answer: str,
        last_test_summary: dict[str, Any] | None,
    ) -> str:
        lines: list[str] = ["任务执行结果："]
        for idx, task in enumerate(board.ordered_tasks(), start=1):
            detail = task.summary or task.acceptance
            lines.append(f"{idx}. {task.title} [{task.status}] {detail}")
        if last_test_summary:
            status_text = "通过" if last_test_summary.get("passed") else "失败"
            lines.append("")
            lines.append(
                f"最近一次测试：{status_text}，耗时 {last_test_summary.get('duration_ms', 0)} ms，命令 {last_test_summary.get('command', '')}"
            )
        if last_answer:
            lines.append("")
            lines.append("最终回复：")
            lines.append(last_answer)
        return "\n".join(lines).strip()

    def _chunk_text(self, text: str, *, chunk_size: int = 160) -> list[str]:
        body = str(text or "")
        if not body:
            return [""]
        return [body[i : i + chunk_size] for i in range(0, len(body), chunk_size)]
