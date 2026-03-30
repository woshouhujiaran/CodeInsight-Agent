from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._common import REPO_ROOT


@dataclass(frozen=True)
class CleanupTarget:
    path: Path
    is_dir: bool
    reason: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remove generated artifacts under the repository.")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be removed.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--keep-eval-result",
        dest="keep_eval_result",
        action="store_true",
        default=True,
        help="Keep outputs/eval_result.json. This is the default behavior.",
    )
    group.add_argument(
        "--remove-eval-result",
        dest="keep_eval_result",
        action="store_false",
        help="Also remove outputs/eval_result.json.",
    )
    parser.add_argument(
        "--include-pytest-cache",
        action="store_true",
        help="Also remove outputs/.pytest_cache and the repository root .pytest_cache.",
    )
    parser.add_argument(
        "--include-pycache",
        action="store_true",
        help="Also remove __pycache__ directories inside the repository.",
    )
    return parser


def collect_cleanup_targets(
    repo_root: Path,
    *,
    keep_eval_result: bool,
    include_pytest_cache: bool,
    include_pycache: bool,
) -> list[CleanupTarget]:
    targets: list[CleanupTarget] = []
    outputs_dir = repo_root / "outputs"

    for path in sorted(outputs_dir.glob("generated_test_*.py")):
        if path.is_file():
            targets.append(CleanupTarget(path=path, is_dir=False, reason="generated_test"))

    eval_result_path = outputs_dir / "eval_result.json"
    if not keep_eval_result and eval_result_path.is_file():
        targets.append(CleanupTarget(path=eval_result_path, is_dir=False, reason="eval_result"))

    if include_pytest_cache:
        for path in (outputs_dir / ".pytest_cache", repo_root / ".pytest_cache"):
            if path.exists():
                targets.append(CleanupTarget(path=path, is_dir=path.is_dir(), reason="pytest_cache"))

    if include_pycache:
        for path in sorted(repo_root.rglob("__pycache__")):
            if path.is_dir():
                targets.append(CleanupTarget(path=path, is_dir=True, reason="pycache"))

    return _dedupe_targets(targets)


def _dedupe_targets(targets: Iterable[CleanupTarget]) -> list[CleanupTarget]:
    seen: set[Path] = set()
    unique: list[CleanupTarget] = []
    for target in targets:
        resolved = target.path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(CleanupTarget(path=resolved, is_dir=target.is_dir, reason=target.reason))
    return unique


def _count_tree(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    if path.is_file():
        return 1, 0

    file_count = 0
    dir_count = 1
    for _root, dirs, files in os.walk(path):
        file_count += len(files)
        dir_count += len(dirs)
    return file_count, dir_count


def apply_cleanup(targets: Sequence[CleanupTarget], *, dry_run: bool) -> dict[str, float | int]:
    deleted_files = 0
    deleted_dirs = 0
    started = time.perf_counter()

    for target in targets:
        file_count, dir_count = _count_tree(target.path)
        deleted_files += file_count
        deleted_dirs += dir_count
        action = "Would remove" if dry_run else "Removed"
        print(f"{action}: {target.path} ({target.reason})")
        if dry_run or not target.path.exists():
            continue
        if target.is_dir:
            shutil.rmtree(target.path)
        else:
            target.path.unlink()

    return {
        "deleted_files": deleted_files,
        "deleted_dirs": deleted_dirs,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    targets = collect_cleanup_targets(
        REPO_ROOT,
        keep_eval_result=args.keep_eval_result,
        include_pytest_cache=args.include_pytest_cache,
        include_pycache=args.include_pycache,
    )

    if not targets:
        print("No matching artifacts found.")
        print("deleted_files=0 deleted_dirs=0 elapsed_seconds=0.0")
        return 0

    summary = apply_cleanup(targets, dry_run=args.dry_run)
    print(
        "deleted_files={deleted_files} deleted_dirs={deleted_dirs} elapsed_seconds={elapsed_seconds}".format(
            **summary
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
