from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable, Iterator

from app.agent.memory import ConversationMemory
from app.contracts import ServiceEvent, normalize_task_results, normalize_tool_trace
from app.runtime import create_agent_from_env, create_llm_from_env
from app.tools.base_tool import ensure_tool_result
from app.tools.filesystem_tools import ListDirTool, ReadFileTool
from app.tools.write_tools import WriteFileTool
from app.web.chat_components import (
    AgenticTaskCoordinator,
    AssistantResponseRenderer,
    ClarificationGuard,
    EventCallback,
    SafetyGuard,
    SessionTestCoordinator,
    StreamCancelled,
    TurnModeDecider,
)
from app.web.session_store import SessionStore, derive_session_title, normalize_session_settings
from app.web.streaming import StreamWorker
from app.web.turn_mode import arbitrate_turn_mode

_QA_SYSTEM_PROMPT = """你是面向 Web 用户的代码问答助手。
先判断用户问题属于哪一类（概念、因果、操作步骤、对比、简短事实等），再选用最贴切的表达方式：解释原理、设计取舍、背景原因时优先用连贯自然段；不要把每个观点都改成编号条目。只有在操作教程、调试顺序、并列对比、参数或检查清单等场景，才用分级标题或小列表。整体读起来应像清晰的技术说明，而不是机械罗列的要点。"""

_WORKSPACE_QA_SYSTEM_PROMPT = """你是面向 Web 用户的本地工作区问答助手。
这类问题允许你借助工作区内的文件和目录来理解当前项目，但你的对外表现仍然应当像正常 QA，而不是任务执行器。
回答要求：
1. 先阅读必要的本地文件/目录，再直接回答用户问题。
2. 不要把回答写成任务拆解、任务看板、模块施工计划或阶段汇报，除非用户明确要求。
3. 默认只读，不要修改文件，不要声称已经修改文件，也不要运行测试或命令。
4. 重点围绕“这个项目/文件夹/文件是什么、做什么、当前结构如何”来总结，避免无关扩展。
5. 若信息不足，先明确说依据了哪些文件，再给出条件化判断。"""


class WebAgentService:
    def __init__(
        self,
        session_store: SessionStore | None = None,
        *,
        agent_factory: Callable[..., Any] = create_agent_from_env,
        llm_factory: Callable[[], Any] = create_llm_from_env,
        repo_root: str | Path | None = None,
        outputs_dir: str | Path = "outputs",
    ) -> None:
        self.session_store = session_store or SessionStore()
        self.agent_factory = agent_factory
        self.llm_factory = llm_factory
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
        self.outputs_dir = Path(outputs_dir).resolve()
        self.mode_decider = TurnModeDecider()
        self.safety_guard = SafetyGuard()
        self.clarification_guard = ClarificationGuard()
        self.renderer = AssistantResponseRenderer()
        self.task_coordinator = AgenticTaskCoordinator(
            session_store=self.session_store,
            agent_factory=self.agent_factory,
            renderer=self.renderer,
        )
        self.test_coordinator = SessionTestCoordinator(
            session_store=self.session_store,
            require_workspace_root=self._require_workspace_root,
        )
        self._latest_eval_cache: dict[str, Any] = {
            "dir_mtime_ns": None,
            "path": None,
            "file_mtime_ns": None,
            "payload": None,
        }

    def resolve_workspace_root(self, raw_path: str | None, *, required: bool = True) -> str:
        text = str(raw_path or "").strip()
        if not text:
            if required:
                raise ValueError("workspace_root 不能为空。")
            return ""

        candidate = Path(text).expanduser()
        resolved = candidate.resolve() if candidate.is_absolute() else (self.repo_root / candidate).resolve()
        if not resolved.exists():
            raise ValueError(f"工作区不存在：{resolved}")
        if not resolved.is_dir():
            raise ValueError(f"workspace_root 必须是目录：{resolved}")
        return str(resolved)

    def create_session(
        self,
        *,
        workspace_root: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_root = self.resolve_workspace_root(workspace_root, required=False)
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
                "pinned": bool(row.get("pinned")),
                "archived": bool(row.get("archived")),
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
        title: str | None = None,
        pinned: bool | None = None,
        archived: bool | None = None,
    ) -> dict[str, Any]:
        resolved_root = self.resolve_workspace_root(workspace_root, required=False) if workspace_root is not None else None
        normalized_settings = normalize_session_settings(settings) if settings is not None else None
        return self.session_store.update_session(
            session_id,
            workspace_root=resolved_root,
            settings=normalized_settings,
            title=title,
            pinned=pinned,
            archived=archived,
        )

    def delete_session(self, session_id: str) -> dict[str, Any]:
        self.session_store.delete_session(session_id)
        return {"deleted": True, "session_id": session_id}

    def chat(self, session_id: str, content: str) -> dict[str, Any]:
        events: list[ServiceEvent] = []
        result = self._process_chat_turn(session_id, content, emit=events.append)
        result["events"] = events
        return result

    def create_stream_chat_worker(self, session_id: str, content: str) -> StreamWorker:
        return StreamWorker(
            lambda cancel_event, emit: self._process_chat_turn(
                session_id,
                content,
                emit=emit,
                cancel_event=cancel_event,
            )
        )

    def stream_chat(self, session_id: str, content: str) -> Iterator[ServiceEvent]:
        return self.create_stream_chat_worker(session_id, content).iter_events()

    def run_session_tests(self, session_id: str, *, emit: EventCallback | None = None) -> dict[str, Any]:
        snapshot = self.session_store.get_session(session_id)
        updated, summary = self.test_coordinator.run_for_snapshot(snapshot, emit=emit)
        return {"session": updated, "summary": summary}

    def list_workspace_tree(
        self,
        session_id: str,
        *,
        path: str = ".",
        depth: int = 4,
        max_entries: int = 400,
        include_metadata: bool = False,
    ) -> dict[str, Any]:
        snapshot = self.session_store.get_session(session_id)
        workspace_root = self._require_workspace_root(snapshot)
        result = self._run_workspace_tool(
            ListDirTool(workspace_root=workspace_root),
            {"path": path, "depth": depth, "max_entries": max_entries},
        )
        payload = result.get("data") if isinstance(result.get("data"), dict) else {}
        entries: list[dict[str, Any]] = []
        root_path = Path(workspace_root).resolve()
        for rel_path in payload.get("paths") or []:
            absolute = (root_path / str(rel_path)).resolve()
            stat = None
            is_dir = absolute.is_dir()
            if include_metadata:
                try:
                    stat = absolute.stat()
                except OSError:
                    stat = None
            entries.append(
                {
                    "path": str(rel_path),
                    "name": Path(str(rel_path)).name or str(rel_path),
                    "is_dir": is_dir,
                    "size_bytes": None if is_dir else int(stat.st_size) if stat else None,
                    "modified_ns": int(stat.st_mtime_ns) if stat else None,
                }
            )
        return {
            "workspace_root": workspace_root,
            "path": path,
            "entries": entries,
            "note": payload.get("note"),
        }

    def read_workspace_file(
        self,
        session_id: str,
        *,
        path: str,
        max_chars: int = 2_000_000,
    ) -> dict[str, Any]:
        snapshot = self.session_store.get_session(session_id)
        workspace_root = self._require_workspace_root(snapshot)
        result = self._run_workspace_tool(
            ReadFileTool(workspace_root=workspace_root),
            {"path": path, "max_chars": max_chars},
        )
        meta = result.get("meta") or {}
        rel_path = str(meta.get("relative_path") or path)
        body = str(result.get("data") or "")
        prefix = f"file={rel_path}\n"
        if body.startswith(prefix):
            body = body[len(prefix) :]
        if meta.get("truncated") and "\n\n[truncated:" in body:
            body = body.rsplit("\n\n[truncated:", 1)[0]
        return {
            "workspace_root": workspace_root,
            "path": rel_path,
            "content": body,
            "content_sha256": str(meta.get("content_sha256") or ""),
            "truncated": bool(meta.get("truncated")),
            "returned_chars": int(meta.get("returned_chars") or len(body)),
        }

    def write_workspace_file(
        self,
        session_id: str,
        *,
        path: str,
        content: str,
        expected_content_hash: str | None = None,
    ) -> dict[str, Any]:
        snapshot = self.session_store.get_session(session_id)
        settings = normalize_session_settings(snapshot.get("settings"))
        if not settings.get("allow_write"):
            raise PermissionError("当前会话未开启 allow_write，不能保存文件。")

        workspace_root = self._require_workspace_root(snapshot)
        args: dict[str, Any] = {"path": path, "content": content}
        if expected_content_hash is not None:
            args["expected_content_hash"] = expected_content_hash
        result = self._run_workspace_tool(WriteFileTool(workspace_root=workspace_root), args)
        meta = result.get("meta") or {}
        return {
            "workspace_root": workspace_root,
            "path": str(meta.get("relative_path") or path),
            "content_sha256": str(meta.get("content_sha256") or ""),
            "message": str(result.get("data") or ""),
        }

    def get_latest_eval_result(self) -> dict[str, Any]:
        preferred = self.outputs_dir / "eval_result.json"
        candidate = preferred if preferred.is_file() else None
        dir_mtime_ns = None
        if candidate is None:
            try:
                dir_mtime_ns = self.outputs_dir.stat().st_mtime_ns
            except OSError:
                dir_mtime_ns = None
            if (
                self._latest_eval_cache.get("dir_mtime_ns") == dir_mtime_ns
                and self._latest_eval_cache.get("path") is not None
            ):
                return {
                    "path": self._latest_eval_cache["path"],
                    "payload": deepcopy(self._latest_eval_cache["payload"]),
                }
            json_files = sorted(self.outputs_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
            candidate = json_files[0] if json_files else None
        if candidate is None:
            return {"path": None, "payload": None}

        try:
            file_mtime_ns = candidate.stat().st_mtime_ns
        except OSError:
            return {"path": None, "payload": None}
        candidate_path = str(candidate.resolve())
        if (
            self._latest_eval_cache.get("path") == candidate_path
            and self._latest_eval_cache.get("file_mtime_ns") == file_mtime_ns
        ):
            return {"path": candidate_path, "payload": deepcopy(self._latest_eval_cache["payload"])}

        with candidate.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self._latest_eval_cache = {
            "dir_mtime_ns": dir_mtime_ns,
            "path": candidate_path,
            "file_mtime_ns": file_mtime_ns,
            "payload": payload,
        }
        return {"path": candidate_path, "payload": deepcopy(payload)}

    def pick_local_path(self, selection: str) -> dict[str, Any]:
        picker = str(selection or "").strip().lower()
        if picker not in {"file", "folder"}:
            raise ValueError("selection 必须是 file 或 folder。")

        chosen = str(self._show_native_picker(picker) or "").strip()
        if not chosen:
            return {
                "selected": False,
                "path": None,
                "workspace_root": "",
                "file_path": None,
                "relative_path": None,
            }

        resolved = Path(chosen).expanduser().resolve()
        if picker == "folder":
            if not resolved.exists():
                raise ValueError(f"所选目录不存在：{resolved}")
            if not resolved.is_dir():
                raise ValueError(f"所选路径不是目录：{resolved}")
            return {
                "selected": True,
                "path": str(resolved),
                "workspace_root": str(resolved),
                "file_path": None,
                "relative_path": None,
            }

        if not resolved.exists():
            raise ValueError(f"所选文件不存在：{resolved}")
        if not resolved.is_file():
            raise ValueError(f"所选路径不是文件：{resolved}")
        return {
            "selected": True,
            "path": str(resolved),
            "workspace_root": str(resolved.parent),
            "file_path": str(resolved),
            "relative_path": resolved.name,
        }

    def _process_chat_turn(
        self,
        session_id: str,
        content: str,
        *,
        emit: EventCallback,
        cancel_event: Any | None = None,
    ) -> dict[str, Any]:
        user_content = str(content or "").strip()
        if not user_content:
            raise ValueError("消息内容不能为空。")

        emit({"event": "stream_profile", "data": {"kind": "phased"}})
        safety_refusal = self.safety_guard.review(user_content)
        snapshot = self.session_store.get_session(session_id)
        if safety_refusal is not None:
            mode = "qa"
        else:
            mode, need_mode_arbitration = self.mode_decider.infer_with_meta(user_content)
            if need_mode_arbitration:
                mode = arbitrate_turn_mode(self.llm_factory(), user_content, fallback=mode)
            workspace_raw = str(snapshot.get("workspace_root") or "").strip()
            if mode in {"agentic", "workspace_qa"} and not workspace_raw:
                mode = "qa"
        clarification_prompt = None
        if safety_refusal is None and mode == "qa":
            clarification_prompt = self.clarification_guard.review(user_content)
        settings = normalize_session_settings(snapshot.get("settings"))
        memory = ConversationMemory.from_snapshot(snapshot)
        history_before_turn = memory.get_messages()

        memory.add_user_message(user_content)
        snapshot["messages"] = memory.get_messages()
        if mode in {"qa", "workspace_qa"}:
            snapshot["tasks"] = []
        self._sync_session_title(snapshot)
        snapshot = self.session_store.save_session(snapshot)

        emit({"event": "mode", "data": {"mode": mode, "agentic": mode == "agentic"}})
        emit({"event": "session", "data": deepcopy(snapshot)})

        if safety_refusal is not None:
            return self._finalize_text_turn(
                memory=memory,
                snapshot=snapshot,
                answer=safety_refusal.answer,
                emit=emit,
                cancel_event=cancel_event,
                metadata_extra={
                    "agentic": False,
                    "mode": "qa",
                    "safety_blocked": True,
                    "safety_reason": safety_refusal.reason,
                },
            )

        if clarification_prompt is not None:
            return self._finalize_text_turn(
                memory=memory,
                snapshot=snapshot,
                answer=clarification_prompt.answer,
                emit=emit,
                cancel_event=cancel_event,
                metadata_extra={
                    "agentic": False,
                    "mode": "qa",
                    "clarification_requested": True,
                },
            )

        if mode == "qa":
            return self._run_qa_turn(
                llm=self.llm_factory(),
                memory=memory,
                snapshot=snapshot,
                user_content=user_content,
                history=history_before_turn,
                emit=emit,
                cancel_event=cancel_event,
            )

        if mode == "workspace_qa":
            return self._run_workspace_qa_turn(
                memory=memory,
                snapshot=snapshot,
                user_content=user_content,
                history=history_before_turn,
                emit=emit,
                cancel_event=cancel_event,
            )

        workspace_root = self._require_workspace_root(snapshot)
        execution = self.task_coordinator.execute(
            snapshot=snapshot,
            user_content=user_content,
            history_before_turn=history_before_turn,
            memory=memory,
            workspace_root=workspace_root,
            settings=settings,
            emit=emit,
            cancel_event=cancel_event,
        )
        self._ensure_not_cancelled(cancel_event)

        snapshot = execution.snapshot
        task_results = normalize_task_results(execution.task_results)
        combined_tool_trace = execution.combined_tool_trace
        last_test_summary = snapshot.get("last_test_summary")
        if self.test_coordinator.should_auto_run_tests(settings, combined_tool_trace):
            snapshot, last_test_summary = self.test_coordinator.run_for_snapshot(snapshot, emit=emit)
        self._ensure_not_cancelled(cancel_event)

        final_answer = self.renderer.compose_final_answer(
            execution.board,
            execution.last_nonempty_answer,
            last_test_summary,
        )
        self._emit_assistant_text(emit, final_answer)
        self._ensure_not_cancelled(cancel_event)

        memory.add_assistant_message(final_answer)
        memory.add_turn_metadata(
            plan=[],
            tool_results=combined_tool_trace,
            recovery_applied=False,
            extra={
                "agentic": True,
                "mode": "agentic",
                "tasks": execution.board.to_dicts(),
                "task_results": task_results,
                "last_test_summary": last_test_summary,
            },
        )

        snapshot["messages"] = memory.get_messages()
        snapshot["turn_metadata"] = memory.get_turn_metadata()
        snapshot["tasks"] = execution.board.to_dicts()
        snapshot["last_test_summary"] = last_test_summary
        self._sync_session_title(snapshot)
        snapshot = self.session_store.save_session(snapshot)

        emit({"event": "assistant_final", "data": {"content": final_answer}})
        emit({"event": "session", "data": deepcopy(snapshot)})
        return {
            "session": snapshot,
            "assistant": final_answer,
            "task_results": task_results,
            "last_test_summary": last_test_summary,
        }

    def _run_qa_turn(
        self,
        *,
        llm: Any,
        memory: ConversationMemory,
        snapshot: dict[str, Any],
        user_content: str,
        history: list[dict[str, str]],
        emit: EventCallback,
        cancel_event: Any | None = None,
    ) -> dict[str, Any]:
        prompt = self._build_qa_prompt(user_content=user_content, history=history)
        answer = str(llm.generate_text(prompt=prompt, system_prompt=_QA_SYSTEM_PROMPT) or "")
        return self._finalize_text_turn(
            memory=memory,
            snapshot=snapshot,
            answer=answer,
            emit=emit,
            cancel_event=cancel_event,
            metadata_extra={"agentic": False, "mode": "qa"},
        )

    def _run_workspace_qa_turn(
        self,
        *,
        memory: ConversationMemory,
        snapshot: dict[str, Any],
        user_content: str,
        history: list[dict[str, str]],
        emit: EventCallback,
        cancel_event: Any | None = None,
    ) -> dict[str, Any]:
        workspace_root = self._require_workspace_root(snapshot)
        agent = self.agent_factory(
            workspace_root,
            memory=memory,
            top_k=5,
            force_reindex=False,
            allow_write=False,
            allow_shell=False,
            test_command="",
        )
        prompt = self._build_workspace_qa_prompt(
            user_content=user_content,
            history=history,
            workspace_root=workspace_root,
        )
        turn = agent.run_agentic(
            prompt,
            max_turns=6,
            workspace_root=workspace_root,
            persist_memory=False,
            cancel_event=cancel_event,
        )
        tool_trace = normalize_tool_trace(turn.tool_trace)
        return self._finalize_text_turn(
            memory=memory,
            snapshot=snapshot,
            answer=turn.answer,
            emit=emit,
            cancel_event=cancel_event,
            metadata_extra={"agentic": False, "mode": "workspace_qa", "tool_results": tool_trace},
            tool_results=tool_trace,
        )

    def _finalize_text_turn(
        self,
        *,
        memory: ConversationMemory,
        snapshot: dict[str, Any],
        answer: str,
        emit: EventCallback,
        cancel_event: Any | None,
        metadata_extra: dict[str, Any],
        tool_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self._ensure_not_cancelled(cancel_event)

        self._emit_assistant_text(emit, answer)

        memory.add_assistant_message(answer)
        memory.add_turn_metadata(
            plan=[],
            tool_results=list(tool_results or []),
            recovery_applied=False,
            extra=metadata_extra,
        )

        snapshot["messages"] = memory.get_messages()
        snapshot["turn_metadata"] = memory.get_turn_metadata()
        snapshot["tasks"] = []
        self._sync_session_title(snapshot)
        snapshot = self.session_store.save_session(snapshot)

        emit({"event": "assistant_final", "data": {"content": answer}})
        emit({"event": "session", "data": deepcopy(snapshot)})

        return {
            "session": snapshot,
            "assistant": answer,
            "task_results": [],
            "last_test_summary": snapshot.get("last_test_summary"),
        }

    def _build_qa_prompt(self, *, user_content: str, history: list[dict[str, str]]) -> str:
        recent_history = [
            f"{item.get('role', 'user')}: {str(item.get('content') or '').strip()}"
            for item in history[-10:]
            if str(item.get("content") or "").strip()
        ]
        history_text = "\n".join(recent_history) if recent_history else "(no prior messages)"
        return (
            "当前处于 QA 模式：你是纯对话问答助手，不是会读写本地仓库、执行命令的项目代理。\n\n"
            "【回答体裁】请针对「当前问题」选形式，避免凡事先列一堆要点。\n"
            "概念、原理、因果、设计取舍、架构思路：以多段自然叙述为主，可少量强调关键术语；不要为了整齐把本可连成段的文字拆成「1.2.3.」。\n"
            "操作步骤、安装部署、严格顺序的调试流程：用编号步骤或有序列表。\n"
            "并列对比（如方案 A/B、技术选型）：可用小标题或简短列表，仍优先保证每块内有完整句子。\n"
            "一句话能答清的事实或定义：一至两段即可，不必搭章节。\n"
            "若不确定，默认用连贯段落；列表仅用于确有并列或顺序关系的内容。\n\n"
            "【行为边界】仅根据用户问题与下方对话历史作答；不要假设已读取工作区、真实路径或运行过任何命令。"
            "不要声称已修改文件、创建文件、跑测试或执行 shell。"
            "需要代码时只给示例片段，并标明为示例、不会自动写入用户磁盘。"
            "不要生成任务看板、代办清单或对当前对话做「子任务拆解」式排版（与任务模式区分）。"
            "关键信息缺失或存在多种合理解读时，先提出一至三个简短澄清问题，勿擅自替用户选定假设。"
            "若用户本条主要是在调整回答风格（更短、更口语、更少列表等），可先一句话确认其偏好，再按该偏好作答。"
            "有明显错别字或断词时按最合理含义理解，必要时先说明你的理解。"
            "涉及越权、泄露敏感信息、绕过认证或破坏性操作时，明确拒绝并给出更安全的替代做法。\n\n"
            f"[对话历史]\n{history_text}\n\n"
            f"[当前问题]\n{user_content}\n"
        )

    def _build_workspace_qa_prompt(
        self,
        *,
        user_content: str,
        history: list[dict[str, str]],
        workspace_root: str,
    ) -> str:
        recent_history = [
            f"{item.get('role', 'user')}: {str(item.get('content') or '').strip()}"
            for item in history[-10:]
            if str(item.get("content") or "").strip()
        ]
        history_text = "\n".join(recent_history) if recent_history else "(no prior messages)"
        return (
            "当前处于 workspace_qa 模式：你可以读取当前工作区来回答，但不要进入任务拆解式输出。\n"
            "请优先通过 list_dir_tool、search_tool、read_file_tool 理解项目或文件夹内容，然后直接给用户自然语言总结。\n"
            "除非用户明确要求，否则不要输出任务板、步骤拆分、修改方案、补丁、测试执行结果或阶段性施工描述。\n"
            "如果用户是在问“这个项目/文件夹/文件是做什么的”，先概括目标，再补充关键目录、主要模块和判断依据。\n"
            "如果某个结论依赖你刚读到的文件，请在表述中简短点明依据文件。\n\n"
            f"[工作区根目录]\n{workspace_root}\n\n"
            f"[对话历史]\n{history_text}\n\n"
            f"[当前问题]\n{user_content}\n"
        )

    def _require_workspace_root(self, snapshot: dict[str, Any]) -> str:
        workspace_root = self.resolve_workspace_root(snapshot.get("workspace_root"), required=False)
        if workspace_root:
            return workspace_root
        raise ValueError("当前会话未设置 workspace_root。QA 模式可以留空；任务模式请先配置真实工作区。")

    def _sync_session_title(self, snapshot: dict[str, Any]) -> None:
        if snapshot.get("title_overridden"):
            return
        snapshot["title"] = derive_session_title(snapshot.get("messages") or [])

    def _emit_assistant_text(self, emit: EventCallback, text: str) -> None:
        emit({"event": "assistant_delta", "data": {"content": str(text or "")}})

    def _run_workspace_tool(self, tool: Any, args: dict[str, Any]) -> dict[str, Any]:
        result = ensure_tool_result(tool.run(args))
        if result.get("status") == "error":
            raise ValueError(str(result.get("error") or "工具执行失败。"))
        return result

    def _show_native_picker(self, selection: str) -> str:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as exc:
            raise RuntimeError("当前环境不支持本地文件选择器。") from exc

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        try:
            root.update_idletasks()
            if selection == "file":
                chosen = filedialog.askopenfilename(title="选择文件")
            else:
                chosen = filedialog.askdirectory(title="选择文件夹", mustexist=True)
        finally:
            root.destroy()
        return str(chosen or "")

    def _ensure_not_cancelled(self, cancel_event: Any | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise StreamCancelled("stream cancelled")
