#!/usr/bin/env python3
"""
Baseline Inference Script for Code Review Environment.

This script runs a language model against the code review environment
and produces reproducible baseline scores.

Usage:
    python inference.py

Environment Variables:
    API_BASE_URL: The API endpoint for the LLM (default: HF Inference API)
    MODEL_NAME: The model identifier to use
    HF_TOKEN: Hugging Face API key
    OPENAI_API_KEY: Alternative API key

Example:
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
    export HF_TOKEN="hf_xxxxx"
    python inference.py
"""

import os
import re
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from openai import OpenAI

from code_review_env import (
    CodeReviewEnv,
    Action,
    ReviewComment,
    CodeLocation,
    get_all_tasks,
)


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")

MAX_STEPS_PER_TASK = 15
MAX_COMMENTS_PER_STEP = 5
TEMPERATURE = 0.3
MAX_TOKENS = 2048

SYSTEM_PROMPT = """You are an expert code reviewer. Your task is to analyze code and identify issues.
For each issue you find, provide:
1. The location (line numbers)
2. The severity (info, warning, error, critical)
3. The category (syntax, logic, security, performance, style, correctness, maintainability)
4. A clear description of the issue
5. A suggested fix (if applicable)

Be thorough but precise. Avoid false positives - only report genuine issues.
When you have finished reviewing, set submit_review to true."""


@dataclass
class InferenceResult:
    task_id: str
    difficulty: str
    final_score: float
    issues_found: List[str]
    false_positives: int
    steps_taken: int
    total_reward: float
    reasoning_log: List[str] = field(default_factory=list)


def parse_comments_from_response(response_text: str) -> List[ReviewComment]:
    """Parse review comments from model response."""
    comments = []

    code_block_match = re.search(
        r'```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```',
        response_text,
        re.MULTILINE
    )

    if code_block_match:
        try:
            parsed = json.loads(code_block_match.group(1))
            if isinstance(parsed, dict) and "comments" in parsed:
                parsed = parsed["comments"]

            items = parsed if isinstance(parsed, list) else [parsed]

            for item in items:
                if isinstance(item, dict):
                    location = item.get("location", {})
                    comments.append(ReviewComment(
                        location=CodeLocation(
                            line_start=location.get("line_start", 1),
                            line_end=location.get("line_end", location.get("line_start", 1)),
                            description=location.get("description", "")
                        ),
                        severity=item.get("severity", "warning"),
                        category=item.get("category", "other"),
                        message=item.get("message", ""),
                        suggestion=item.get("suggestion")
                    ))
            return comments
        except json.JSONDecodeError:
            pass

    comment_blocks = re.split(
        r'(?=\d+\.\s*(?:Line |Location|Severity))',
        response_text
    )

    for block in comment_blocks:
        if not block.strip():
            continue

        location_match = re.search(
            r'[Ll]ine[s]?\s*(\d+)(?:\s*[-–]\s*(\d+))?',
            block
        )
        if not location_match:
            continue

        line_start = int(location_match.group(1))
        line_end = int(location_match.group(2)) if location_match.group(2) else line_start

        severity_match = re.search(
            r'[Ss]everity[:\s]+(\w+)',
            block
        )
        severity = severity_match.group(1).lower() if severity_match else "warning"
        if severity not in ["info", "warning", "error", "critical"]:
            severity = "warning"

        category_match = re.search(
            r'[Cc]ategory[:\s]+(\w+)',
            block
        )
        category = category_match.group(1).lower() if category_match else "other"
        if category not in ["syntax", "logic", "security", "performance",
                           "style", "correctness", "maintainability", "other"]:
            category = "other"

        message_lines = []
        in_message = False
        for line in block.split('\n'):
            if re.match(r'(?:[Mm]essage|[Dd]escription|Issue)[:\s]', line):
                in_message = True
                message_lines.append(re.sub(r'^[Mm]essage\s*[:\-]\s*', '', line))
            elif in_message and line.strip() and not line.startswith(' ' * 4):
                if re.match(r'^[A-Z][a-z]+[:\s]', line):
                    break
                message_lines.append(line)

        message = ' '.join(message_lines).strip()

        suggestion_match = re.search(
            r'[Ss]uggest(?:ion|ed\s*fix)[:\s]+(.+?)(?:\n\n|\n[A-Z]|$)',
            block,
            re.DOTALL
        )
        suggestion = suggestion_match.group(1).strip() if suggestion_match else None

        if message:
            comments.append(ReviewComment(
                location=CodeLocation(
                    line_start=line_start,
                    line_end=line_end,
                    description=f"Lines {line_start}-{line_end}"
                ),
                severity=severity,
                category=category,
                message=message,
                suggestion=suggestion
            ))

    return comments[:MAX_COMMENTS_PER_STEP]


def build_user_prompt(observation, reasoning: str = "") -> str:
    """Build user prompt for the model."""
    code_lines = observation.code_snippet.split('\n')

    numbered_code = '\n'.join(
        f"{i+1:3d} | {line}" 
        for i, line in enumerate(code_lines)
    )

    prompt = f"""Task: {observation.task_description}

File: {observation.file_path}
Language: {observation.language}

Code to review:
```
{numbered_code}
```

"""

    if observation.previous_comments:
        prompt += "Previously submitted comments:\n"
        for i, comment in enumerate(observation.previous_comments, 1):
            prompt += f"{i}. Line {comment.location.line_start}: [{comment.severity}] {comment.message[:100]}\n"
        prompt += "\n"

    if reasoning:
        prompt += f"Your reasoning: {reasoning}\n\n"

    prompt += """Provide your review comments in the following JSON format:
```json
{
  "comments": [
    {
      "location": {"line_start": 1, "line_end": 1, "description": "..."},
      "severity": "warning",
      "category": "logic",
      "message": "Description of the issue",
      "suggestion": "Suggested fix (optional)"
    }
  ],
  "submit_review": false,
  "reasoning": "Your analysis for this step"
}
```

If you believe the review is complete, set submit_review to true.
If you find no issues, return empty comments array with submit_review true."""

    return prompt


def call_model(
    client: OpenAI,
    messages: List[Dict],
    max_tokens: int = MAX_TOKENS
) -> str:
    """Call the language model."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=max_tokens,
            stream=False,
        )
        return completion.choices[0].message.content or ""
    except Exception as e:
        print(f"  Model call failed: {e}")
        return ""


def run_task(
    client: OpenAI,
    task_id: str,
    verbose: bool = False
) -> InferenceResult:
    """Run inference on a single task."""
    env = CodeReviewEnv(task_id=task_id, seed=42)
    result = env.reset(task_id=task_id)
    observation = result.observation

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {observation.task_name} ({observation.difficulty})")
        print(f"{'='*60}")
        print(f"Code to review ({len(observation.code_snippet.splitlines())} lines):")
        print(observation.code_snippet[:500])
        if len(observation.code_snippet) > 500:
            print("... (truncated)")
        print()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    total_reward = 0.0
    reasoning_log = []

    for step in range(1, MAX_STEPS_PER_TASK + 1):
        user_prompt = build_user_prompt(observation)
        messages.append({"role": "user", "content": user_prompt})

        if verbose:
            print(f"Step {step}/{MAX_STEPS_PER_TASK}: Calling model...")

        response_text = call_model(client, messages)
        messages.append({"role": "assistant", "content": response_text})

        comments = parse_comments_from_response(response_text)

        submit_review = "submit_review" in response_text.lower() and (
            re.search(r'"submit_review"\s*:\s*true', response_text, re.IGNORECASE) or
            re.search(r'submit_review\s*[:=]\s*true', response_text, re.IGNORECASE)
        )

        action = Action(
            comments=comments,
            submit_review=submit_review,
            reasoning=""
        )

        step_result = env.step(action)

        total_reward += step_result.reward.value
        observation = step_result.observation

        if verbose:
            print(f"  Step reward: {step_result.reward.value:.4f}")
            print(f"  Cumulative reward: {total_reward:.4f}")
            print(f"  Comments this step: {len(comments)}")
            print(f"  Done: {step_result.done}")

        if step_result.done:
            if verbose:
                print(f"\nEpisode complete!")
                print(f"Final message: {step_result.reward.message}")
            break

    final_result = InferenceResult(
        task_id=task_id,
        difficulty=observation.difficulty,
        final_score=step_result.reward.value,
        issues_found=step_result.reward.issues_found,
        false_positives=step_result.reward.false_positives,
        steps_taken=step,
        total_reward=total_reward,
        reasoning_log=reasoning_log
    )

    env.close()
    return final_result


def main():
    """Main inference loop."""
    print("=" * 60)
    print("Code Review Environment - Baseline Inference")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"API: {API_BASE_URL}")
    print()

    if not API_KEY:
        print("ERROR: No API key provided. Set HF_TOKEN or OPENAI_API_KEY.")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks = get_all_tasks()
    results: List[InferenceResult] = []

    for task in tasks:
        print(f"\nRunning task: {task.name} ({task.difficulty})...")

        try:
            result = run_task(client, task.task_id, verbose=False)
            results.append(result)

            print(f"  Score: {result.final_score:.4f}")
            print(f"  Issues found: {len(result.issues_found)}/{len(task.issues)}")
            print(f"  False positives: {result.false_positives}")
            print(f"  Steps taken: {result.steps_taken}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(InferenceResult(
                task_id=task.task_id,
                difficulty=task.difficulty,
                final_score=0.0,
                issues_found=[],
                false_positives=0,
                steps_taken=0,
                total_reward=0.0
            ))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_score = 0.0
    for result in results:
        difficulty_marker = {
            "easy": "[E]",
            "medium": "[M]",
            "hard": "[H]"
        }.get(result.difficulty, "?")

        print(f"{difficulty_marker} {result.task_id:20s} Score: {result.final_score:.4f}")
        total_score += result.final_score

    avg_score = total_score / len(results) if results else 0.0
    print("-" * 40)
    print(f"Average Score: {avg_score:.4f}")
    print()

    output_path = "inference_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "api": API_BASE_URL,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "average_score": avg_score,
                "results": [
                    {
                        "task_id": r.task_id,
                        "difficulty": r.difficulty,
                        "final_score": r.final_score,
                        "issues_found": r.issues_found,
                        "false_positives": r.false_positives,
                        "steps_taken": r.steps_taken,
                        "total_reward": r.total_reward
                    }
                    for r in results
                ]
            },
            f,
            indent=2
        )

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
