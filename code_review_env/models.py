from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class CodeLocation(BaseModel):
    line_start: int = Field(description="Starting line number")
    line_end: int = Field(description="Ending line number")
    description: str = Field(default="", description="Description of the code region")


class ReviewComment(BaseModel):
    location: CodeLocation = Field(description="Location in code")
    severity: Literal["info", "warning", "error", "critical"] = Field(
        default="warning",
        description="Severity level of the issue"
    )
    category: Literal[
        "syntax", "logic", "security", "performance", 
        "style", "correctness", "maintainability", "other"
    ] = Field(default="other", description="Category of the issue")
    message: str = Field(description="The review comment/message")
    suggestion: Optional[str] = Field(
        default=None, 
        description="Suggested fix if applicable"
    )


class Observation(BaseModel):
    task_id: str = Field(description="Current task identifier")
    task_name: str = Field(description="Human-readable task name")
    task_description: str = Field(description="Detailed task instructions")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        description="Task difficulty level"
    )
    code_snippet: str = Field(description="The code to review")
    language: str = Field(default="python", description="Programming language")
    file_path: str = Field(description="File path for context")
    target_issues: List[dict] = Field(
        default_factory=list,
        description="Types of issues to find (not the actual issues)"
    )
    max_comments: int = Field(
        default=10, 
        description="Maximum number of review comments allowed"
    )
    step: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=20, description="Maximum steps per episode")
    cumulative_reward: float = Field(default=0.0, description="Total reward so far")
    previous_comments: List[ReviewComment] = Field(
        default_factory=list,
        description="Comments made in previous steps"
    )
    info: dict = Field(
        default_factory=dict,
        description="Additional context information"
    )


class Action(BaseModel):
    comments: List[ReviewComment] = Field(
        default_factory=list,
        description="List of review comments to submit"
    )
    submit_review: bool = Field(
        default=False,
        description="Whether to submit the review and end the episode"
    )
    reasoning: str = Field(
        default="",
        description="Agent's reasoning for this action"
    )


class Reward(BaseModel):
    value: float = Field(description="Reward value")
    partial_scores: dict = Field(
        default_factory=dict,
        description="Breakdown of reward components"
    )
    issues_found: List[str] = Field(
        default_factory=list,
        description="List of issue IDs that were correctly identified"
    )
    false_positives: int = Field(
        default=0,
        description="Number of incorrect findings"
    )
    message: str = Field(
        default="",
        description="Human-readable reward explanation"
    )


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool = Field(description="Whether episode is finished")
    info: dict = Field(
        default_factory=dict,
        description="Additional step information"
    )
