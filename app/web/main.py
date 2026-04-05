from __future__ import annotations

import asyncio
from contextlib import suppress
from contextlib import asynccontextmanager
from pathlib import Path
import json
import os
from typing import Any

from app.utils.env_loader import load_env_file
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import iterate_in_threadpool
import uvicorn

from app.web.schemas import (
    ChatResponseModel,
    EvalLatestResponseModel,
    LocalPathPickerResponseModel,
    MessageCreateModel,
    SessionCreateModel,
    SessionSnapshotModel,
    SessionSummaryModel,
    SessionUpdateModel,
    WorkspaceFileResponseModel,
    WorkspaceFileUpdateModel,
    WorkspaceFileWriteResponseModel,
    WorkspaceTreeResponseModel,
)
from app.web.service import WebAgentService

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _dump_model(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _sse_message(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _static_asset_version(path: Path) -> str:
    try:
        return str(path.stat().st_mtime_ns)
    except OSError:
        return "0"


def _bootstrap_env() -> None:
    """与 CLI 一致：从仓库根目录加载 .env，避免 Web 进程读不到 API Key。"""
    repo_root = Path(__file__).resolve().parents[2]
    load_env_file(str(repo_root / ".env"))
    load_env_file(".env")


@asynccontextmanager
async def _app_lifespan(app: FastAPI):
    _bootstrap_env()
    yield


def create_app(service: WebAgentService | None = None) -> FastAPI:
    app = FastAPI(title="CodeInsight-Agent Web", version="1.0.0", lifespan=_app_lifespan)
    app.state.service = service or WebAgentService()
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.middleware("http")
    async def ensure_utf8_charset(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        content_type = str(response.headers.get("content-type") or "")
        lowered = content_type.lower()
        if lowered.startswith("application/json") and "charset=" not in lowered:
            response.headers["content-type"] = "application/json; charset=utf-8"
        elif lowered.startswith("text/event-stream") and "charset=" not in lowered:
            response.headers["content-type"] = "text/event-stream; charset=utf-8"
        return response

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        template_path = TEMPLATES_DIR / "index.html"
        html = template_path.read_text(encoding="utf-8")
        css_version = _static_asset_version(STATIC_DIR / "web" / "index.css")
        js_version = _static_asset_version(STATIC_DIR / "web" / "index.js")
        html = html.replace('/static/web/index.css"', f'/static/web/index.css?v={css_version}"')
        html = html.replace('/static/web/index.js"', f'/static/web/index.js?v={js_version}"')
        return HTMLResponse(html)

    @app.get("/sessions", response_model=list[SessionSummaryModel])
    def list_sessions() -> list[dict[str, Any]]:
        return app.state.service.list_sessions()

    @app.post("/sessions", response_model=SessionSnapshotModel)
    def create_session(payload: SessionCreateModel) -> dict[str, Any]:
        try:
            return app.state.service.create_session(
                workspace_root=payload.workspace_root,
                settings=_dump_model(payload.settings),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/sessions/{session_id}", response_model=SessionSnapshotModel)
    def get_session(session_id: str) -> dict[str, Any]:
        try:
            return app.state.service.get_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="会话不存在。") from exc

    @app.patch("/sessions/{session_id}", response_model=SessionSnapshotModel)
    def update_session(session_id: str, payload: SessionUpdateModel) -> dict[str, Any]:
        try:
            settings = _dump_model(payload.settings) if payload.settings is not None else None
            return app.state.service.update_session(
                session_id,
                workspace_root=payload.workspace_root,
                settings=settings,
                title=payload.title,
                pinned=payload.pinned,
                archived=payload.archived,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="会话不存在。") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str) -> dict[str, Any]:
        try:
            return app.state.service.delete_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="会话不存在。") from exc

    @app.post("/sessions/{session_id}/messages", response_model=ChatResponseModel)
    def post_message(
        session_id: str,
        payload: MessageCreateModel,
        request: Request,
        stream: bool = Query(default=False),
    ) -> Any:
        try:
            if stream:
                worker = app.state.service.create_stream_chat_worker(session_id, payload.content)
                disconnect_flag = {"requested": False}

                async def watch_disconnect() -> None:
                    while not disconnect_flag["requested"]:
                        if await request.is_disconnected():
                            disconnect_flag["requested"] = True
                            worker.cancel()
                            break
                        await asyncio.sleep(0.1)

                async def event_stream() -> Any:
                    disconnect_task = asyncio.create_task(watch_disconnect())
                    try:
                        async for item in iterate_in_threadpool(
                            worker.iter_events(should_stop=lambda: disconnect_flag["requested"])
                        ):
                            if disconnect_flag["requested"]:
                                break
                            yield _sse_message(item["event"], item["data"])
                    finally:
                        disconnect_flag["requested"] = True
                        worker.cancel()
                        disconnect_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await disconnect_task

                return StreamingResponse(event_stream(), media_type="text/event-stream")
            return app.state.service.chat(session_id, payload.content)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="会话不存在。") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/tests/run")
    def run_tests(session_id: str) -> dict[str, Any]:
        try:
            return app.state.service.run_session_tests(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="会话不存在。") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/sessions/{session_id}/workspace/tree", response_model=WorkspaceTreeResponseModel)
    def get_workspace_tree(
        session_id: str,
        path: str = Query(default="."),
        depth: int = Query(default=4, ge=1, le=8),
        max_entries: int = Query(default=400, ge=1, le=5000),
        include_metadata: bool = Query(default=False),
    ) -> dict[str, Any]:
        try:
            return app.state.service.list_workspace_tree(
                session_id,
                path=path,
                depth=depth,
                max_entries=max_entries,
                include_metadata=include_metadata,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="会话不存在。") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/sessions/{session_id}/workspace/file", response_model=WorkspaceFileResponseModel)
    def get_workspace_file(
        session_id: str,
        path: str = Query(..., min_length=1),
        max_chars: int = Query(default=2_000_000, ge=1, le=2_000_000),
    ) -> dict[str, Any]:
        try:
            return app.state.service.read_workspace_file(
                session_id,
                path=path,
                max_chars=max_chars,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="会话不存在。") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/sessions/{session_id}/workspace/file", response_model=WorkspaceFileWriteResponseModel)
    def put_workspace_file(session_id: str, payload: WorkspaceFileUpdateModel) -> dict[str, Any]:
        try:
            return app.state.service.write_workspace_file(
                session_id,
                path=payload.path,
                content=payload.content,
                expected_content_hash=payload.expected_content_hash,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="会话不存在。") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/eval/latest", response_model=EvalLatestResponseModel)
    def get_latest_eval() -> dict[str, Any]:
        return app.state.service.get_latest_eval_result()

    @app.post("/system/pick-folder", response_model=LocalPathPickerResponseModel)
    def pick_folder() -> dict[str, Any]:
        try:
            return app.state.service.pick_local_path("folder")
        except RuntimeError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/system/pick-file", response_model=LocalPathPickerResponseModel)
    def pick_file() -> dict[str, Any]:
        try:
            return app.state.service.pick_local_path("file")
        except RuntimeError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()


def run() -> None:
    _bootstrap_env()
    port = int(os.getenv("WEB_PORT", "8765"))
    uvicorn.run("app.web.main:app", host="127.0.0.1", port=port, reload=False)


if __name__ == "__main__":
    run()
