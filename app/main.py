from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from app.agent.agent import CodeAgent
from app.agent.executor import Executor
from app.agent.planner import Planner
from app.agent.tool_registry import ToolRegistry
from app.llm.llm import LLMClient
from app.rag.embeddings import create_embedding_backend
from app.rag.load_or_build import load_or_build_vector_store
from app.rag.retriever import CodeRetriever
from app.tools.analyze_tool import AnalyzeTool
from app.tools.optimize_tool import OptimizeTool
from app.tools.search_tool import SearchTool
from app.tools.test_tool import TestTool
from app.utils.env_loader import load_env_file
from app.utils.logger import get_logger


def render_turn_result(result: object) -> None:
    """Render agent output in a clearer CLI layout."""
    plan = getattr(result, "plan", [])
    tool_results = getattr(result, "tool_results", [])
    answer = getattr(result, "answer", "")

    print("\n" + "=" * 18 + " PLAN " + "=" * 18)
    if plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        print("(empty)")

    print("\n" + "=" * 14 + " TOOL RESULTS " + "=" * 14)
    if tool_results:
        for idx, item in enumerate(tool_results, start=1):
            tool_name = item.get("tool", "unknown_tool")
            status = item.get("status", "unknown")
            output = item.get("output", "")
            print(f"\n[{idx}] {tool_name} ({status})")
            print("-" * 42)
            if isinstance(output, str):
                # Try pretty JSON first; fallback to plain text.
                try:
                    parsed = json.loads(output)
                    print(json.dumps(parsed, ensure_ascii=False, indent=2))
                except json.JSONDecodeError:
                    print(output)
            else:
                print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print("(empty)")

    print("\n" + "=" * 17 + " ANSWER " + "=" * 17)
    print(answer or "(empty)")


def default_index_dir(codebase_dir: str) -> Path:
    """Stable per-codebase path under data/index/ to avoid mixing indexes."""
    h = hashlib.sha256(str(Path(codebase_dir).resolve()).encode("utf-8")).hexdigest()[:16]
    return Path("data/index") / h


def build_agent(
    codebase_dir: str,
    top_k: int = 5,
    *,
    index_dir: Path | None = None,
    force_reindex: bool = False,
) -> CodeAgent:
    logger = get_logger("codeinsight.main")
    llm_provider = os.getenv("LLM_PROVIDER", "deepseek")
    llm_model = os.getenv("LLM_MODEL", "deepseek-chat")
    llm = LLMClient(model=llm_model, provider=llm_provider)

    embedding = create_embedding_backend()
    idx_path = index_dir or default_index_dir(codebase_dir)
    store, rag_meta = load_or_build_vector_store(
        codebase_dir,
        idx_path,
        embedding,
        force_reindex=force_reindex,
    )
    logger.info("RAG: %s", rag_meta)

    retriever = CodeRetriever(store=store)
    registry = ToolRegistry()
    registry.register(SearchTool(retriever=retriever, top_k=top_k))
    registry.register(AnalyzeTool(llm=llm))
    registry.register(OptimizeTool(llm=llm))
    registry.register(TestTool(llm=llm))

    planner = Planner(llm=llm)
    executor = Executor(registry=registry)
    return CodeAgent(planner=planner, executor=executor, llm=llm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CodeInsight-Agent CLI")
    parser.add_argument(
        "--codebase-dir",
        default="data/codebase",
        help="Directory to ingest into FAISS index",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Retriever top-k")
    parser.add_argument(
        "--index-dir",
        default="",
        help="Directory to store/load FAISS index (default: data/index/<hash of codebase path>)",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Rebuild FAISS index from codebase even if a matching snapshot exists",
    )
    parser.add_argument("--query", default="", help="Single-turn query mode")
    return parser.parse_args()


def main() -> None:
    # Ensure Windows consoles can print full Unicode text (e.g. emoji).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    load_env_file(".env")
    args = parse_args()
    codebase_dir = Path(args.codebase_dir)
    if not codebase_dir.exists():
        codebase_dir.mkdir(parents=True, exist_ok=True)

    index_path = Path(args.index_dir) if args.index_dir.strip() else None
    agent = build_agent(
        codebase_dir=str(codebase_dir),
        top_k=args.top_k,
        index_dir=index_path,
        force_reindex=args.force_reindex,
    )

    if args.query.strip():
        result = agent.run(args.query)
        render_turn_result(result)
        return

    print("CodeInsight-Agent interactive mode. 输入 exit 退出。")
    while True:
        user_query = input("\n你> ").strip()
        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        result = agent.run(user_query)
        render_turn_result(result)


if __name__ == "__main__":
    main()
