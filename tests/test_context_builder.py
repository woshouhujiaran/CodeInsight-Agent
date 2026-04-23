from __future__ import annotations

import json

from app.agent.context_builder import build_context


def test_build_context_dedupes_search_hits_and_tracks_dependency_depth() -> None:
    hits = [
        {
            "chunk_id": "c1",
            "file_path": "app/a.py",
            "start_line": 10,
            "end_line": 20,
            "content": "def alpha():\n    return 1",
            "rerank_score": 0.9,
        },
        {
            "chunk_id": "c1",
            "file_path": "app/a.py",
            "start_line": 10,
            "end_line": 20,
            "content": "def alpha():\n    return 1",
            "rerank_score": 0.8,
        },
    ]
    primary_results = [
        {
            "step_id": "s1",
            "tool": "search_tool",
            "status": "ok",
            "output": json.dumps(hits, ensure_ascii=False),
            "deps": [],
        },
        {
            "step_id": "s2",
            "tool": "analyze_tool",
            "status": "ok",
            "output": "analysis text",
            "deps": ["s1"],
        },
    ]

    context = build_context(
        user_query="检查 alpha",
        primary_results=primary_results,
        recovery_results=None,
        max_chars=8_000,
    )

    assert "Round 1" in context
    assert context.count("def alpha():") == 1
    assert "dep_depth=1" in context


def test_build_context_respects_budget_and_marks_omitted_items() -> None:
    big_text = "x" * 5_000
    primary_results = [
        {
            "step_id": "s1",
            "tool": "read_file_tool",
            "status": "ok",
            "output": big_text,
            "deps": [],
        },
        {
            "step_id": "s2",
            "tool": "grep_tool",
            "status": "ok",
            "output": big_text,
            "deps": ["s1"],
        },
    ]

    context = build_context(
        user_query="超长上下文预算",
        primary_results=primary_results,
        recovery_results=None,
        max_chars=2_100,
    )

    assert len(context) <= 2_100
    assert "[context_builder] omitted" in context


def test_build_context_prioritizes_search_over_analyze() -> None:
    primary_results = [
        {
            "step_id": "s1",
            "tool": "analyze_tool",
            "status": "ok",
            "output": "analysis output",
            "deps": [],
        },
        {
            "step_id": "s2",
            "tool": "search_tool",
            "status": "ok",
            "output": json.dumps(
                [
                    {
                        "chunk_id": "c2",
                        "file_path": "app/b.py",
                        "content": "def beta():\n    pass",
                        "rerank_score": 0.7,
                    }
                ],
                ensure_ascii=False,
            ),
            "deps": [],
        },
    ]

    context = build_context(
        user_query="检查 beta",
        primary_results=primary_results,
        recovery_results=None,
        max_chars=8_000,
    )

    assert context.find("Tool: search_tool") < context.find("Tool: analyze_tool")
