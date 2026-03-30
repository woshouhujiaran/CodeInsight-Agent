from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._common import build_index_for_workspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load or build the repository RAG index.")
    parser.add_argument(
        "--workspace-root",
        default=".",
        help="Workspace root to index. Defaults to the current repository root when run from the repo.",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Rebuild the index even when the persisted snapshot still matches.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_index_for_workspace(
            args.workspace_root,
            force_reindex=args.force_reindex,
        )
    except FileNotFoundError as exc:
        print(f"[build_index] {exc}", file=sys.stderr)
        return 1
    except NotADirectoryError as exc:
        print(f"[build_index] {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"[build_index] failed to initialize embedding or index backend: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[build_index] failed to load/build index: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

