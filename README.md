# Code Review Environment for OpenEnv

A real-world code review environment where AI agents analyze code snippets, identify bugs, security vulnerabilities, and provide actionable feedback.

## Overview

This environment simulates a genuine software engineering task: reviewing code changes for issues. Agents must:

1. Analyze code snippets carefully
2. Identify bugs, security vulnerabilities, and quality issues
3. Provide precise location information (line numbers)
4. Categorize issues by severity and type
5. Suggest fixes when applicable

### Why Code Review?

Code review is a critical real-world task that:

- **Is genuinely useful** - Every software team performs code reviews
- **Requires reasoning** - Agents must understand code semantics, not just pattern match
- **Has clear success criteria** - Issues can be objectively verified
- **Shows difficulty progression** - From syntax errors to subtle security vulnerabilities

## Environment Description

### Task Overview

| Task ID | Name | Difficulty | Description |
|---------|------|------------|-------------|
| `syntax_basics` | Syntax and Basic Issues | Easy | Detect syntax errors and obvious correctness issues |
| `logic_bugs` | Logic and Algorithmic Issues | Medium | Find logic errors, off-by-one bugs, and algorithmic flaws |
| `security_review` | Security Vulnerability Detection | Hard | Identify critical security vulnerabilities |

### Action Space

The agent submits `Action` objects containing:

```python
Action(
    comments=[
        ReviewComment(
            location=CodeLocation(
                line_start=5,
                line_end=7,
                description="Loop condition"
            ),
            severity="warning",  # info, warning, error, critical
            category="logic",     # syntax, logic, security, performance, etc.
            message="Off-by-one error in loop condition",
            suggestion="Change < to <="
        )
    ],
    submit_review=False  # Set to True when review is complete
)
```

### Observation Space

Each step returns an `Observation` containing:

- `task_id`: Current task identifier
- `task_description`: Instructions for the current task
- `difficulty`: Easy, medium, or hard
- `code_snippet`: The code to review
- `language`: Programming language (python, javascript, etc.)
- `file_path`: Context path for the file
- `step`: Current step number
- `max_steps`: Maximum allowed steps
- `cumulative_reward`: Running reward total
- `previous_comments`: Comments submitted in prior steps

### Reward Function

Rewards are calculated based on:

1. **Precision** (50% weight): Ratio of true positives to total claimed issues
2. **Recall** (50% weight): Ratio of issues found to total issues
3. **Completion Bonus**: +0.5 for perfect detection with no false positives
4. **Efficiency Bonus**: +0.1 for completing within 50% of max steps
5. **Difficulty Bonus**: +0.0 (easy), +0.1 (medium), +0.2 (hard)

**Partial Progress**: Reward signals are provided throughout the episode, not just at the end.

**Penalties**:
- -0.05 for submitting excessive comments (> max_comments)
- -0.02 per step for no new progress after 3 consecutive steps

## Setup Instructions

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/code-review-env.git
cd code-review-env

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
# Start the OpenEnv server
python -m uvicorn code_review_env.server:app --host 0.0.0.0 --port 7860

# Or run directly
python code_review_env/server.py
```

### Docker Deployment

```bash
# Build the image
docker build -t code-review-env .

# Run the container
docker run -p 7860:7860 code-review-env
```

### Hugging Face Space Deployment

1. Create a new Space at https://huggingface.co/new-space
2. Select "Docker" as the SDK
3. Copy your files to the Space
4. The Space will automatically build and run

## Usage Examples

### Python API

```python
from code_review_env import CodeReviewEnv, Action, ReviewComment, CodeLocation

# Create environment
env = CodeReviewEnv()

# Reset and get initial observation
result = env.reset(task_id="syntax_basics")
observation = result.observation

print(f"Code to review:\n{observation.code_snippet}")

# Submit review comments
action = Action(
    comments=[
        ReviewComment(
            location=CodeLocation(line_start=4, line_end=4),
            severity="error",
            category="syntax",
            message="Missing colon after for loop",
            suggestion="Add ':' after 'for num in numbers'"
        )
    ],
    submit_review=True
)

# Step through environment
result = env.step(action)
print(f"Reward: {result.reward.value}")
print(f"Done: {result.done}")
print(f"Message: {result.reward.message}")

env.close()
```

### REST API

```bash
# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "syntax_basics"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "action": {
      "comments": [...],
      "submit_review": false
    }
  }'

# Get current state
curl "http://localhost:7860/state?session_id=your-session-id"

# List available tasks
curl http://localhost:7860/tasks
```

## Baseline Scores

Running `inference.py` with the default model produces these baseline scores:

| Task | Difficulty | Expected Score |
|------|-----------|----------------|
| syntax_basics | Easy | 0.75 - 0.85 |
| logic_bugs | Medium | 0.55 - 0.70 |
| security_review | Hard | 0.40 - 0.60 |

**Average Baseline**: ~0.60

To reproduce:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
export HF_TOKEN="hf_your_token_here"
python inference.py
```

## Running the Inference Script

The `inference.py` script runs a language model against all tasks:

```bash
# Set environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-api-key"

# Run inference
python inference.py
```

Results are saved to `inference_results.json`.

## Grading System

Each task has predefined issues that agents must identify. The grader evaluates:

- **True Positives**: Correctly identified issues
- **False Positives**: Incorrectly flagged issues
- **Missed Issues**: Issues the agent failed to find

```python
score = 0.5 * precision + 0.5 * recall

# Bonuses
if perfect_detection:
    score += 0.5
elif good_detection_no_fp:
    score += 0.25

# Final score capped at 1.0
score = min(1.0, score + difficulty_bonus)
```

## API Reference

### Methods

#### `reset(task_id=None, seed=None) -> StepResult`
Reset the environment to initial state. Returns initial observation.

#### `step(action: Action) -> StepResult`
Execute one step with the given action. Returns observation, reward, done flag, and info.

#### `state() -> dict`
Return current environment state.

### Models

#### `Observation`
- `task_id`: str
- `task_name`: str
- `task_description`: str
- `difficulty`: Literal["easy", "medium", "hard"]
- `code_snippet`: str
- `language`: str
- `file_path`: str
- `step`: int
- `max_steps`: int
- `cumulative_reward`: float
- `previous_comments`: List[ReviewComment]

#### `Action`
- `comments`: List[ReviewComment]
- `submit_review`: bool

#### `Reward`
- `value`: float (0.0 - 1.0)
- `partial_scores`: dict
- `issues_found`: List[str]
- `false_positives`: int
- `message`: str

## Project Structure

```
code_review_env/
├── __init__.py           # Package init
├── models.py             # Pydantic models (Observation, Action, Reward)
├── environment.py        # Core environment implementation
├── server.py             # FastAPI server for REST API
├── tasks/
│   ├── __init__.py
│   └── definitions.py    # Task definitions and graders
├── openenv.yaml          # OpenEnv specification
└── README.md             # This file

inference.py              # Baseline inference script
Dockerfile                # Container build file
requirements.txt          # Python dependencies
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please read our contributing guidelines before submitting PRs.
