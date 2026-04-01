"""Code Review Environment for OpenEnv.

A real-world code review environment where AI agents analyze code snippets,
identify bugs, security vulnerabilities, and provide actionable feedback.

Example usage:
    from code_review_env import CodeReviewEnv

    env = CodeReviewEnv()
    result = env.reset()
    print(result.observation.code_snippet)

    action = Action(
        comments=[ReviewComment(...)],
        submit_review=False
    )
    result = env.step(action)
"""

from .models import (
    Observation,
    Action,
    Reward,
    StepResult,
    ReviewComment,
    CodeLocation,
)
from .environment import CodeReviewEnv, create_env
from .tasks import (
    CodeReviewTask,
    Issue,
    GradingResult,
    get_all_tasks,
    get_task_by_id,
)

__version__ = "1.0.0"

__all__ = [
    "Observation",
    "Action", 
    "Reward",
    "StepResult",
    "ReviewComment",
    "CodeLocation",
    "CodeReviewEnv",
    "create_env",
    "CodeReviewTask",
    "Issue",
    "GradingResult",
    "get_all_tasks",
    "get_task_by_id",
]
