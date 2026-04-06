from typing import List, Optional
from pydantic import BaseModel


class Document(BaseModel):
    """A document to be audited (e.g., privacy policy)."""
    id: str
    title: str
    content: str
    doc_type: str


class DataPractice(BaseModel):
    """A data practice extracted from real operations."""
    id: str
    category: str
    purpose: str
    data_type: str
    shared_with_third_parties: bool = False


class Observation(BaseModel):
    """Observation returned by the environment after reset/step."""
    task_id: str
    task_name: str
    difficulty: str
    step: int
    documents: List[Document]
    data_practices: List[DataPractice]
    compliance_requirements: List[str]
    flagged_issues: List[str]
    echoed_message: str


class Action(BaseModel):
    """Action submitted by the agent — a compliance finding."""
    message: str


class Reward(BaseModel):
    """Reward structure with partial progress signals."""
    value: float
    reason: str
    issues_found: int = 0
    total_issues: int = 0
