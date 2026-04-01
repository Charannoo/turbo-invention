"""Task definitions and graders for the code review environment."""

from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import re


@dataclass
class Issue:
    issue_id: str
    line_start: int
    line_end: int
    severity: str
    category: str
    description: str
    expected_findings: List[str]


@dataclass
class GradingResult:
    score: float
    issues_found: List[str]
    false_positives: List[str]
    missed_issues: List[str]
    precision: float
    recall: float
    message: str


class CodeReviewTask:
    """Base class for code review tasks."""

    def __init__(
        self,
        task_id: str,
        name: str,
        description: str,
        difficulty: str,
        code: str,
        language: str,
        file_path: str,
        issues: List[Issue],
        target_issue_types: List[str],
        max_comments: int = 10,
        max_steps: int = 20
    ):
        self.task_id = task_id
        self.name = name
        self.description = description
        self.difficulty = difficulty
        self.code = code
        self.language = language
        self.file_path = file_path
        self.issues = issues
        self.target_issue_types = target_issue_types
        self.max_comments = max_comments
        self.max_steps = max_steps

    def get_ground_truth(self) -> List[Issue]:
        """Return all issues in the code."""
        return self.issues

    def grade(
        self, 
        submitted_comments: List[Dict], 
        partial_progress_bonus: float = 0.0
    ) -> GradingResult:
        """Grade the submitted review comments."""
        issues_found = []
        false_positives = []
        missed_issues = []
        matched_issues = set()

        for comment in submitted_comments:
            line_start = comment.get("location", {}).get("line_start", 0)
            line_end = comment.get("location", {}).get("line_end", line_start)
            comment_category = comment.get("category", "other")
            message = comment.get("message", "").lower()

            matched = False
            for issue in self.issues:
                if issue.issue_id in matched_issues:
                    continue

                if self._matches_issue(comment, issue, line_start, line_end):
                    issues_found.append(issue.issue_id)
                    matched_issues.add(issue.issue_id)
                    matched = True
                    break

            if not matched:
                false_positives.append(f"Line {line_start}: {message[:50]}")

        for issue in self.issues:
            if issue.issue_id not in matched_issues:
                missed_issues.append(issue.issue_id)

        total_issues = len(self.issues)
        found_count = len(issues_found)
        fp_count = len(false_positives)

        if total_issues > 0:
            recall = found_count / total_issues
        else:
            recall = 1.0

        total_claimed = found_count + fp_count
        if total_claimed > 0:
            precision = found_count / total_claimed
        else:
            precision = 1.0

        base_score = 0.5 * precision + 0.5 * recall

        if found_count == total_issues and fp_count == 0:
            bonus = 0.5
        elif found_count >= total_issues * 0.8 and fp_count == 0:
            bonus = 0.25
        else:
            bonus = 0.0

        score = min(1.0, base_score + bonus + partial_progress_bonus)

        message = f"Found {found_count}/{total_issues} issues"
        if fp_count > 0:
            message += f", {fp_count} false positive(s)"
        message += f". Precision: {precision:.2f}, Recall: {recall:.2f}"

        return GradingResult(
            score=score,
            issues_found=issues_found,
            false_positives=false_positives,
            missed_issues=missed_issues,
            precision=precision,
            recall=recall,
            message=message
        )

    def _matches_issue(
        self, 
        comment: Dict, 
        issue: Issue, 
        line_start: int, 
        line_end: int
    ) -> bool:
        """Check if a comment matches an issue."""
        comment_category = comment.get("category", "other")
        message = comment.get("message", "").lower()

        line_overlap = not (
            line_end < issue.line_start or 
            line_start > issue.line_end
        )

        category_match = comment_category == issue.category

        finding_keywords = [kw.lower() for kw in issue.expected_findings]
        content_match = any(
            kw in message for kw in finding_keywords
        ) or any(
            kw.replace(" ", "") in message.replace(" ", "") 
            for kw in finding_keywords
        )

        return line_overlap and (category_match or content_match)


def create_easy_syntax_task() -> CodeReviewTask:
    """Task 1: Detect syntax errors and obvious issues (Easy)."""
    code = '''
def calculate_average(numbers):
    total = 0
    for num in numbers
        total += num
    return total / len(numbers)

def greet(name):
    print("Hello " + name

result = calculate_average([1, 2, 3])
greet("World")
'''

    issues = [
        Issue(
            issue_id="easy_1",
            line_start=4,
            line_end=4,
            severity="error",
            category="syntax",
            description="Missing colon after for loop",
            expected_findings=["colon", "missing", "syntax", "for"]
        ),
        Issue(
            issue_id="easy_2",
            line_start=8,
            line_end=8,
            severity="error",
            category="syntax",
            description="Missing closing parenthesis",
            expected_findings=["parenthesis", "missing", "closing", ")"]
        ),
        Issue(
            issue_id="easy_3",
            line_start=5,
            line_end=5,
            severity="error",
            category="correctness",
            description="Division by zero if list is empty",
            expected_findings=["zero", "division", "empty", "divide"]
        ),
    ]

    return CodeReviewTask(
        task_id="syntax_basics",
        name="Syntax and Basic Issues",
        description="Review the code and identify syntax errors and basic correctness issues. "
                    "Look for missing punctuation, obvious bugs, and runtime errors.",
        difficulty="easy",
        code=code.strip(),
        language="python",
        file_path="src/calculations.py",
        issues=issues,
        target_issue_types=["syntax", "correctness"],
        max_comments=5,
        max_steps=10
    )


def create_medium_logic_task() -> CodeReviewTask:
    """Task 2: Detect logic errors and bugs (Medium)."""
    code = '''
def find_max(numbers):
    max_val = 0
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val

def binary_search(arr, target):
    left, right = 0, len(arr)
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def process_data(items):
    results = []
    for i, item in enumerate(items):
        if item > 10:
            results.append(item * 2)
        else:
            continue
    return results

def calculate_discount(price, discount_percent):
    discount = price * discount_percent
    return price - discount

data = [5, 12, -3, 8, 0]
print(find_max(data))
print(binary_search(data, 8))
print(process_data(data))
'''

    issues = [
        Issue(
            issue_id="medium_1",
            line_start=3,
            line_end=3,
            severity="error",
            category="logic",
            description="max_val initialized to 0 fails for all-negative numbers",
            expected_findings=["negative", "initialization", "zero", "edge case"]
        ),
        Issue(
            issue_id="medium_2",
            line_start=10,
            line_end=10,
            severity="error",
            category="logic",
            description="Off-by-one: right should be len(arr) - 1",
            expected_findings=["off-by-one", "bounds", "index", "len"]
        ),
        Issue(
            issue_id="medium_3",
            line_start=25,
            line_end=25,
            severity="warning",
            category="logic",
            description="continue in loop prevents adding items <= 10",
            expected_findings=["continue", "logic", "skip", "items"]
        ),
        Issue(
            issue_id="medium_4",
            line_start=28,
            line_end=29,
            severity="error",
            category="correctness",
            description="Discount calculation is wrong: 10% becomes 0.10, not 10",
            expected_findings=["discount", "percent", "decimal", "10", "calculation"]
        ),
    ]

    return CodeReviewTask(
        task_id="logic_bugs",
        name="Logic and Algorithmic Issues",
        description="Review the code for logic errors and algorithmic bugs. "
                    "Look for off-by-one errors, incorrect edge case handling, "
                    "and flawed business logic.",
        difficulty="medium",
        code=code.strip(),
        language="python",
        file_path="src/algorithms.py",
        issues=issues,
        target_issue_types=["logic", "correctness", "algorithm"],
        max_comments=8,
        max_steps=15
    )


def create_hard_security_task() -> CodeReviewTask:
    """Task 3: Detect security vulnerabilities (Hard)."""
    code = '''
import hashlib
import sqlite3
from typing import List

def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    return result is not None

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()

def get_user_profile(user_id):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    return result

def process_template(template_path, **kwargs):
    with open(template_path, 'r') as f:
        template = f.read()
    for key, value in kwargs.items():
        template = template.replace(f'{{{key}}}', value)
    return template

def download_file(url, filename):
    import os
    filepath = os.path.join('downloads', filename)
    os.system(f'wget -O {filepath} {url}')
    return filepath

def generate_token(user_id):
    import time
    import base64
    timestamp = str(int(time.time()))
    data = f"{user_id}:{timestamp}"
    return base64.b64encode(data.encode()).decode()

def validate_email(email):
    import re
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(pattern, email) is not None
'''

    issues = [
        Issue(
            issue_id="hard_1",
            line_start=9,
            line_end=9,
            severity="critical",
            category="security",
            description="SQL Injection vulnerability - user input directly in query",
            expected_findings=["sql", "injection", "parameterized", "query", "f-string"]
        ),
        Issue(
            issue_id="hard_2",
            line_start=13,
            line_end=13,
            severity="critical",
            category="security",
            description="MD5 is cryptographically broken for password hashing",
            expected_findings=["md5", "hash", "password", "cryptographic", "security"]
        ),
        Issue(
            issue_id="hard_3",
            line_start=22,
            line_end=22,
            severity="critical",
            category="security",
            description="SQL Injection via string formatting in query",
            expected_findings=["sql", "injection", "user_id", "formatting"]
        ),
        Issue(
            issue_id="hard_4",
            line_start=27,
            line_end=32,
            severity="critical",
            category="security",
            description="Server-Side Template Injection (SSTI) vulnerability",
            expected_findings=["template", "injection", "ssti", "replace", "unsafe"]
        ),
        Issue(
            issue_id="hard_5",
            line_start=36,
            line_end=37,
            severity="critical",
            category="security",
            description="Command injection vulnerability via os.system",
            expected_findings=["command", "injection", "os.system", "shell", "wget"]
        ),
        Issue(
            issue_id="hard_6",
            line_start=40,
            line_end=43,
            severity="warning",
            category="security",
            description="JWT-like token without signature is easily forgeable",
            expected_findings=["token", "signature", "forgeable", "jwt", "unsigned"]
        ),
    ]

    return CodeReviewTask(
        task_id="security_review",
        name="Security Vulnerability Detection",
        description="This is a critical security review task. You must identify all security "
                    "vulnerabilities in the code including SQL injection, command injection, "
                    "insecure cryptography, and other security issues. Each vulnerability "
                    "could lead to system compromise.",
        difficulty="hard",
        code=code.strip(),
        language="python",
        file_path="src/security_module.py",
        issues=issues,
        target_issue_types=["security", "injection", "cryptography"],
        max_comments=10,
        max_steps=20
    )


def get_all_tasks() -> List[CodeReviewTask]:
    """Return all available tasks."""
    return [
        create_easy_syntax_task(),
        create_medium_logic_task(),
        create_hard_security_task(),
    ]


def get_task_by_id(task_id: str) -> CodeReviewTask:
    """Get a specific task by ID."""
    for task in get_all_tasks():
        if task.task_id == task_id:
            return task
    raise ValueError(f"Task not found: {task_id}")
