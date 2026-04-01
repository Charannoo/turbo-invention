"""Task __init__.py to make it a package."""

from .definitions import (
    CodeReviewTask,
    Issue,
    GradingResult,
    create_easy_syntax_task,
    create_medium_logic_task,
    create_hard_security_task,
    get_all_tasks,
    get_task_by_id,
)

__all__ = [
    "CodeReviewTask",
    "Issue",
    "GradingResult",
    "create_easy_syntax_task",
    "create_medium_logic_task",
    "create_hard_security_task",
    "get_all_tasks",
    "get_task_by_id",
]
