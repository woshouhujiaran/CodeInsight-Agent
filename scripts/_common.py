from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.embeddings import create_embedding_backend
from app.rag.load_or_build import load_or_build_vector_store
from app.runtime import default_index_dir
from app.utils.env_loader import load_env_file

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs"
EVAL_RESULT_PATH = OUTPUTS_DIR / "eval_result.json"


def bootstrap_environment() -> Path:
    load_env_file(str(REPO_ROOT / ".env"))
    load_env_file(".env")
    return REPO_ROOT


def resolve_workspace_root(raw_path: str | Path | None = None) -> Path:
    candidate = Path(raw_path) if raw_path is not None else REPO_ROOT
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"workspace_root does not exist: {candidate}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"workspace_root must be a directory: {candidate}")
    return candidate


def ensure_outputs_dir() -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_index_for_workspace(
    workspace_root: str | Path,
    *,
    force_reindex: bool = False,
) -> dict[str, Any]:
    bootstrap_environment()
    resolved_root = resolve_workspace_root(workspace_root)
    embedding = create_embedding_backend()
    index_dir = default_index_dir(str(resolved_root)).resolve()
    store, meta = load_or_build_vector_store(
        str(resolved_root),
        index_dir,
        embedding,
        force_reindex=force_reindex,
    )
    result = dict(meta)
    result["workspace_root"] = str(resolved_root)
    result["index_dir"] = str(index_dir)
    result["document_count"] = int(getattr(store.index, "ntotal", 0))
    return result


def iso_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def environment_summary(*, workspace_root: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "workspace_root": str(workspace_root),
    }

