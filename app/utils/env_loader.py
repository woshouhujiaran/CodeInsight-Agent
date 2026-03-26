from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str = ".env") -> None:
    """
    Lightweight .env loader without external dependencies.
    Existing environment variables are kept unless they are empty.
    """
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        current = os.environ.get(key)
        if current is None or current == "":
            os.environ[key] = value
