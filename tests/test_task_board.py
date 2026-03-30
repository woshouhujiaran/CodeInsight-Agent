from __future__ import annotations

import pytest

from app.agent.task_board import TaskBoard


def _board() -> TaskBoard:
    return TaskBoard.from_dicts(
        [
            {
                "id": "t1",
                "title": "定位",
                "description": "定位代码入口。",
                "depends_on": [],
                "status": "pending",
                "acceptance": "找到相关文件。",
            },
            {
                "id": "t2",
                "title": "修改",
                "description": "修改实现。",
                "depends_on": ["t1"],
                "status": "pending",
                "acceptance": "改动完成。",
            },
            {
                "id": "t3",
                "title": "验证",
                "description": "验证结果。",
                "depends_on": ["t2"],
                "status": "pending",
                "acceptance": "验证通过。",
            },
        ]
    )


def test_task_board_orders_by_dependencies() -> None:
    board = _board()
    assert [task.id for task in board.ordered_tasks()] == ["t1", "t2", "t3"]


def test_task_board_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        TaskBoard.from_dicts(
            [
                {
                    "id": "t1",
                    "title": "a",
                    "description": "a",
                    "depends_on": [],
                    "status": "pending",
                    "acceptance": "a",
                },
                {
                    "id": "t1",
                    "title": "b",
                    "description": "b",
                    "depends_on": [],
                    "status": "pending",
                    "acceptance": "b",
                },
                {
                    "id": "t2",
                    "title": "c",
                    "description": "c",
                    "depends_on": [],
                    "status": "pending",
                    "acceptance": "c",
                },
            ]
        )


def test_task_board_status_transitions() -> None:
    board = _board()
    board.mark_in_progress("t1")
    board.mark_done("t1", summary="已完成")
    assert board.get("t1").summary == "已完成"
    with pytest.raises(ValueError, match="invalid task transition"):
        board.mark_in_progress("t1")


def test_task_board_failed_summary_and_completed_summaries() -> None:
    board = _board()
    board.mark_in_progress("t1")
    board.mark_done("t1", summary="完成定位")
    board.mark_in_progress("t2")
    board.mark_failed("t2", summary="修改失败")
    assert board.completed_summaries() == ["定位: 完成定位"]
    assert board.get("t2").status == "failed"
