from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


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
