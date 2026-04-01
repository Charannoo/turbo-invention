"""Code Review Environment - OpenEnv compliant implementation."""

import random
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from .models import (
    Observation,
    Action,
    Reward,
    StepResult,
    ReviewComment,
)
from .tasks import (
    get_all_tasks,
    get_task_by_id,
    CodeReviewTask,
    GradingResult,
)


PARTIAL_PROGRESS_WEIGHTS = {
    "easy": 0.15,
    "medium": 0.12,
    "hard": 0.08,
}

STEP_PENALTIES = {
    "excessive_comments": -0.05,
    "no_progress": -0.02,
}


@dataclass
class EnvironmentState:
    """Internal state of the code review environment."""
    task: CodeReviewTask
    current_step: int = 0
    cumulative_reward: float = 0.0
    submitted_comments: List[Dict] = field(default_factory=list)
    previously_found_issues: List[str] = field(default_factory=list)
    steps_without_progress: int = 0
    episode_complete: bool = False
    last_partial_score: float = 0.0


class CodeReviewEnv:
    """OpenEnv compliant code review environment.

    This environment simulates a real-world code review task where an AI agent
    must analyze code, identify issues, and provide actionable feedback.

    The agent receives rewards based on:
    - Finding real issues (true positives)
    - Avoiding false positives
    - Completeness of review
    - Partial progress signals throughout the episode
    """

    def __init__(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        max_steps: int = 20,
    ):
        """Initialize the environment.

        Args:
            task_id: Specific task to load. If None, tasks are randomly selected.
            seed: Random seed for reproducibility.
            max_steps: Maximum steps per episode.
        """
        self._task_id = task_id
        self._seed = seed
        self._max_steps = max_steps
        self._state: Optional[EnvironmentState] = None

        if seed is not None:
            random.seed(seed)

    @property
    def task_id(self) -> str:
        """Current task ID."""
        if self._state is None:
            return self._task_id or ""
        return self._state.task.task_id

    @property
    def observation_space(self) -> Dict[str, Any]:
        """Define the observation space structure."""
        return {
            "task_id": str,
            "task_name": str,
            "task_description": str,
            "difficulty": str,
            "code_snippet": str,
            "language": str,
            "file_path": str,
            "target_issues": list,
            "max_comments": int,
            "step": int,
            "max_steps": int,
            "cumulative_reward": float,
            "previous_comments": list,
        }

    @property
    def action_space(self) -> Dict[str, Any]:
        """Define the action space structure."""
        return {
            "comments": list,
            "submit_review": bool,
            "reasoning": str,
        }

    def reset(self, task_id: Optional[str] = None) -> StepResult:
        """Reset the environment to initial state.

        Args:
            task_id: Override task ID. Uses constructor value if not provided.

        Returns:
            StepResult with initial observation.
        """
        target_task_id = task_id or self._task_id

        if target_task_id:
            task = get_task_by_id(target_task_id)
        else:
            all_tasks = get_all_tasks()
            task = random.choice(all_tasks)

        max_steps = min(self._max_steps, task.max_steps)

        self._state = EnvironmentState(
            task=task,
            current_step=0,
            cumulative_reward=0.0,
            submitted_comments=[],
            previously_found_issues=[],
            steps_without_progress=0,
            episode_complete=False,
            last_partial_score=0.0,
        )

        observation = self._create_observation()

        return StepResult(
            observation=observation,
            reward=Reward(value=0.0, message="Episode started"),
            done=False,
            info={"task": task.task_id, "difficulty": task.difficulty}
        )

    def step(self, action: Action) -> StepResult:
        """Execute one step in the environment.

        Args:
            action: The action containing review comments to submit.

        Returns:
            StepResult with observation, reward, done flag, and info.
        """
        if self._state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        if self._state.episode_complete:
            return StepResult(
                observation=self._create_observation(),
                reward=Reward(value=0.0, message="Episode already complete"),
                done=True,
                info={"error": "Episode already finished"}
            )

        self._state.current_step += 1

        new_comments = [c.model_dump() for c in action.comments]
        for comment in new_comments:
            if comment.get("reasoning"):
                del comment["reasoning"]

        self._state.submitted_comments.extend(new_comments)

        partial_result = self._calculate_partial_reward(
            self._state.submitted_comments
        )

        reward_value = partial_result.value
        reward_info = partial_result.partial_scores

        if action.submit_review or self._state.current_step >= self._state.task.max_steps:
            final_result = self._calculate_final_reward(
                self._state.submitted_comments
            )
            reward_value = final_result.value
            reward_info = final_result.partial_scores
            self._state.episode_complete = True

        if partial_result.value > self._state.last_partial_score:
            self._state.steps_without_progress = 0
        else:
            self._state.steps_without_progress += 1

        if self._state.steps_without_progress > 3:
            reward_value += STEP_PENALTIES["no_progress"]

        if len(new_comments) > self._state.task.max_comments:
            reward_value += STEP_PENALTIES["excessive_comments"]

        self._state.cumulative_reward += reward_value
        self._state.last_partial_score = partial_result.value

        observation = self._create_observation()

        return StepResult(
            observation=observation,
            reward=Reward(
                value=reward_value,
                partial_scores=reward_info,
                issues_found=partial_result.issues_found,
                false_positives=partial_result.false_positives,
                message=partial_result.message
            ),
            done=self._state.episode_complete,
            info={
                "step": self._state.current_step,
                "total_comments": len(self._state.submitted_comments),
                "is_final": self._state.episode_complete
            }
        )

    def state(self) -> Dict[str, Any]:
        """Return current environment state.

        Returns:
            Dictionary containing full environment state.
        """
        if self._state is None:
            return {"error": "Environment not initialized"}

        return {
            "task_id": self._state.task.task_id,
            "task_name": self._state.task.name,
            "difficulty": self._state.task.difficulty,
            "current_step": self._state.current_step,
            "max_steps": self._state.task.max_steps,
            "cumulative_reward": self._state.cumulative_reward,
            "submitted_comments": self._state.submitted_comments,
            "episode_complete": self._state.episode_complete,
            "remaining_steps": self._state.task.max_steps - self._state.current_step,
        }

    def _create_observation(self) -> Observation:
        """Create observation from current state."""
        return Observation(
            task_id=self._state.task.task_id,
            task_name=self._state.task.name,
            task_description=self._state.task.description,
            difficulty=self._state.task.difficulty,
            code_snippet=self._state.task.code,
            language=self._state.task.language,
            file_path=self._state.task.file_path,
            target_issues=[
                {"type": issue_type}
                for issue_type in self._state.task.target_issue_types
            ],
            max_comments=self._state.task.max_comments,
            step=self._state.current_step,
            max_steps=self._state.task.max_steps,
            cumulative_reward=self._state.cumulative_reward,
            previous_comments=[
                ReviewComment(**c) 
                for c in self._state.submitted_comments[-10:]
            ],
            info={
                "code_lines": len(self._state.task.code.splitlines()),
                "issue_count_hint": len(self._state.task.issues),
            }
        )

    def _calculate_partial_reward(
        self, 
        comments: List[Dict]
    ) -> Reward:
        """Calculate partial reward for current progress."""
        result = self._state.task.grade(comments, partial_progress_bonus=0.0)

        difficulty_weight = PARTIAL_PROGRESS_WEIGHTS.get(
            self._state.task.difficulty, 0.1
        )

        issues_found_count = len(result.issues_found)
        total_issues = len(self._state.task.issues)

        if total_issues > 0:
            progress_ratio = issues_found_count / total_issues
        else:
            progress_ratio = 1.0

        partial_score = progress_ratio * difficulty_weight

        return Reward(
            value=partial_score,
            partial_scores={
                "precision": result.precision,
                "recall": result.recall,
                "issues_found": issues_found_count,
                "total_issues": total_issues,
                "false_positives": len(result.false_positives),
            },
            issues_found=result.issues_found,
            false_positives=len(result.false_positives),
            message=f"Progress: {issues_found_count}/{total_issues} issues found"
        )

    def _calculate_final_reward(
        self, 
        comments: List[Dict]
    ) -> Reward:
        """Calculate final reward when episode completes."""
        result = self._state.task.grade(comments)

        difficulty_bonus = {
            "easy": 0.0,
            "medium": 0.1,
            "hard": 0.2,
        }.get(self._state.task.difficulty, 0.0)

        steps_bonus = 0.0
        if self._state.current_step <= self._state.task.max_steps * 0.5:
            steps_bonus = 0.1

        final_score = min(1.0, result.score + difficulty_bonus + steps_bonus)

        return Reward(
            value=final_score,
            partial_scores={
                "base_score": result.score,
                "difficulty_bonus": difficulty_bonus,
                "steps_bonus": steps_bonus,
                "precision": result.precision,
                "recall": result.recall,
            },
            issues_found=result.issues_found,
            false_positives=len(result.false_positives),
            message=result.message
        )

    def get_task_list(self) -> List[Dict[str, Any]]:
        """Get list of all available tasks."""
        tasks = get_all_tasks()
        return [
            {
                "task_id": t.task_id,
                "name": t.name,
                "difficulty": t.difficulty,
                "description": t.description,
                "issue_count": len(t.issues),
            }
            for t in tasks
        ]

    def close(self) -> None:
        """Clean up environment resources."""
        self._state = None


def create_env(task_id: Optional[str] = None, seed: Optional[int] = None) -> CodeReviewEnv:
    """Factory function to create environment instances."""
    return CodeReviewEnv(task_id=task_id, seed=seed)
