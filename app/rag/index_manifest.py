from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def compute_codebase_snapshot(
    target_dir: str,
    *,
    include_suffixes: tuple[str, ...],
    excluded_dirs: frozenset[str],
) -> str:
    """
    Stable fingerprint of tracked files under target_dir (relative path + mtime + size).
    Used to skip re-ingest when persisted index is still valid.
    """
    base = Path(target_dir).resolve()
    if not base.exists():
        return hashlib.sha256(b"").hexdigest()

    entries: list[tuple[str, int, int]] = []

    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        for filename in files:
            path = Path(root) / filename
            if include_suffixes and path.suffix.lower() not in include_suffixes:
                continue
            try:
                st = path.stat()
                rel = str(path.relative_to(base))
                entries.append((rel, int(st.st_mtime_ns), int(st.st_size)))
            except OSError:
                continue

    entries.sort()
    blob = json.dumps(entries, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def compute_file_manifest(
    target_dir: str,
    *,
    include_suffixes: tuple[str, ...],
    excluded_dirs: frozenset[str],
) -> dict[str, dict[str, int]]:
    """
    Per-file manifest for incremental RAG updates.
    Value payload: {"mtime_ns": int, "size": int}
    """
    base = Path(target_dir).resolve()
    manifest: dict[str, dict[str, int]] = {}
    if not base.exists():
        return manifest

    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        for filename in files:
            path = Path(root) / filename
            if include_suffixes and path.suffix.lower() not in include_suffixes:
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            rel = str(path.relative_to(base))
            manifest[rel] = {
                "mtime_ns": int(st.st_mtime_ns),
                "size": int(st.st_size),
            }
    return manifest


def diff_file_manifests(
    previous: dict[str, dict[str, Any]] | None,
    current: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Return (changed_or_new_rel_paths, deleted_rel_paths)."""
    prev = previous or {}
    changed: list[str] = []
    deleted: list[str] = []
    for rel, cur_meta in current.items():
        old = prev.get(rel)
        if not isinstance(old, dict):
            changed.append(rel)
            continue
        if int(old.get("mtime_ns", -1)) != int(cur_meta.get("mtime_ns", -2)) or int(old.get("size", -1)) != int(
            cur_meta.get("size", -2)
        ):
            changed.append(rel)
    for rel in prev:
        if rel not in current:
            deleted.append(rel)
    return changed, deleted


def load_file_manifest(index_dir: Path) -> dict[str, dict[str, int]] | None:
    path = Path(index_dir) / "file_manifest.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return None
    out: dict[str, dict[str, int]] = {}
    for rel, meta in payload.items():
        if not isinstance(meta, dict):
            continue
        out[str(rel)] = {
            "mtime_ns": int(meta.get("mtime_ns", 0)),
            "size": int(meta.get("size", 0)),
        }
    return out


def write_file_manifest(index_dir: Path, manifest: dict[str, dict[str, int]]) -> None:
    path = Path(index_dir) / "file_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
