#!/usr/bin/env python3
"""
Baseline Inference Script for GDPR Auditor Environment.

Uses OpenAI Client + HTTP calls to the server to run a model against the environment.

Usage:
    python inference.py

Environment Variables:
    API_BASE_URL:   The API endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME:     The model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN:       Your Hugging Face / API key (required)
    SERVER_URL:     The environment server URL (default: http://localhost:7860)

Expected format for STDOUT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import re
import textwrap
import requests
from typing import List, Optional

from openai import OpenAI


API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:7860")

MAX_STEPS = 8
MAX_TOKENS = 512
TEMPERATURE = 0.7
SUCCESS_SCORE_THRESHOLD = 0.5
BENCHMARK = "gdpr_auditor_env"

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert GDPR Compliance Auditor. Your job is to review privacy policies
    and data practices to identify compliance violations.
    
    IMPORTANT: This is a multi-step auditing task. You must systematically review
    each section of the policy and data practice to find ALL issues.
    
    Task Types:
    - Easy (Clause Existence): Check if mandatory GDPR clauses are present
    - Medium (Purpose Mapping): Match data practices to stated purposes
    - Hard (Dark Patterns): Find contradictions between policy statements and actual practices
    
    Output Format:
    Your finding should clearly identify:
    - The issue type (missing clause, contradiction, etc.)
    - The specific violation
    - The severity (if mentioned)
    
    Examples of findings:
    - "Missing Right to be Forgotten clause"
    - "Health data shared with advertisers - contradicts policy"
    - "Section 1 says never shared but Section 3 lists third-party partners"
    
    Output ONLY your finding, nothing else."""
).strip()


def log_start(task, env, model):
    print(f"[START] Task: {task} | Env: {env} | Model: {model}", flush=True)


def log_step(step, action, reward, done, error):
    print(f"[STEP] Step: {step} | Action: {action} | Reward: {reward} | Done: {done} | Error: {error}", flush=True)


def log_end(success, steps, score, rewards):
    print(f"[END] Success: {success} | Steps: {steps} | Score: {score} | Rewards: {rewards}", flush=True)


def reset_env(task: str) -> dict:
    """Reset the environment via HTTP."""
    resp = requests.get(f"{SERVER_URL}/reset", params={"task": task})
    resp.raise_for_status()
    return resp.json()


def step_env(message: str) -> dict:
    """Send action to environment via HTTP."""
    resp = requests.post(f"{SERVER_URL}/step", json={"message": message})
    resp.raise_for_status()
    return resp.json()


def build_user_prompt(obs_data: dict, step_num: int) -> str:
    documents = obs_data.get("documents", [])
    data_practices = obs_data.get("data_practices", [])
    requirements = obs_data.get("compliance_requirements", [])
    flagged = obs_data.get("flagged_issues", [])
    
    policy_text = documents[0].get("content", "") if documents else "No policy found"
    
    prompt = f"""Privacy Policy Audit (Step {step_num}):

--- POLICY CONTENT ---
{policy_text[:1500]}

--- DATA PRACTICES ---
"""
    for dp in data_practices:
        prompt += f"- {dp.get('category')}: {dp.get('purpose')} (shared: {dp.get('shared_with_third_parties')})\n"
    
    prompt += f"""
--- COMPLIANCE REQUIREMENTS ---
"""
    for req in requirements:
        prompt += f"- {req}\n"
    
    if flagged:
        prompt += f"\n--- ALREADY FLAGGED ---\n"
        for f in flagged:
            prompt += f"- {f}\n"
    
    prompt += f"""
Task: {obs_data.get('task_name')} ({obs_data.get('difficulty')})

Provide your next compliance finding:"""
    
    return prompt


def call_model(client: OpenAI, user_prompt: str, history: List[dict]) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                return line
        return text if text else "No finding submitted"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "No finding submitted"


TASKS = {
    "easy": {"task_id": "easy_clause_existence", "name": "Clause Existence Check", "difficulty": "easy"},
    "medium": {"task_id": "medium_purpose_mapping", "name": "Purpose Mapping", "difficulty": "medium"},
    "hard": {"task_id": "hard_dark_patterns", "name": "Dark Pattern Detection", "difficulty": "hard"},
    "elite": {"task_id": "elite_multi_doc_reasoning", "name": "Multi-Document Reasoning", "difficulty": "elite"},
}


def run_task(client: OpenAI, task_key: str, verbose: bool = False) -> dict:
    """Run inference on a single task via HTTP."""
    task = TASKS[task_key]
    task_name = task["name"]

    history: List[dict] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = reset_env(task_key)
        obs_data = result.get("observation", {})
        
        done = result.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            user_prompt = build_user_prompt(obs_data, step)
            response_text = call_model(client, user_prompt, history)
            history.append({"role": "assistant", "content": response_text})

            action_str = response_text[:60] + "..." if len(response_text) > 60 else response_text
            action_log_str = action_str.replace('\n', ' ').replace('\r', '')

            try:
                result = step_env(response_text)
                
                reward_data = result.get("reward", {})
                if not reward_data:
                    print(f"[DEBUG] No reward in response: {result}", flush=True)
                if isinstance(reward_data, dict):
                    reward = reward_data.get("value", 0.0)
                else:
                    reward = float(reward_data)
                done = result.get("done", False)
                error_msg = None
                obs_data = result.get("observation", {})
                
            except Exception as exc:
                error_msg = str(exc).replace('\n', ' ').replace('\r', '')
                reward = 0.0
                done = True
                obs_data = {}

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_log_str, reward=reward, done=done, error=error_msg)

            if done:
                break

        score = max(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        error_msg = str(exc).replace('\n', ' ').replace('\r', '')
        print(f"[DEBUG] Task execution error: {error_msg}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task["task_id"],
        "task_name": task_name,
        "score": score,
        "success": success,
        "steps": steps_taken,
        "rewards": rewards,
    }


def main():
    print("=" * 60)
    print("GDPR Auditor - Baseline Inference")
    print("=" * 60)
    print(f"API URL : {API_BASE_URL}")
    print(f"Model  : {MODEL_NAME}")
    print(f"Server : {SERVER_URL}")
    print()

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set")
        return

    if not API_BASE_URL:
        print("ERROR: API_BASE_URL not set")
        return

    if not MODEL_NAME:
        print("ERROR: MODEL_NAME not set")
        return

    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
        )
    except Exception as e:
        print(f"ERROR: Failed to create client: {e}")
        return

    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        if resp.status_code != 200:
            print(f"ERROR: Server returned {resp.status_code}")
            return
        print("Server connection: OK")
    except Exception as e:
        print(f"ERROR: Cannot connect to server: {e}")
        return

    results = []
    
    for task_key in ["easy", "medium", "hard", "elite"]:
        task_name = TASKS[task_key]["name"]
        print(f"\nRunning task: {task_name} ({task_key})...")
        
        result = run_task(client, task_key)
        results.append(result)
        print(f"  score={result['score']:.4f}  steps={result['steps']}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for r in results:
        print(f"[{r['task_id'][:1].upper()}] {r['task_id']:<30} score={r['score']:.4f}")
    
    print("-" * 60)
    
    avg = sum(r["score"] for r in results) / len(results)
    print(f"Average score: {avg:.4f}")

    output = {
        "model": MODEL_NAME,
        "api_url": API_BASE_URL,
        "timestamp": str(__import__("datetime").datetime.now()),
        "average_score": avg,
        "results": results,
    }
    
    with open("inference_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: inference_results.json")


if __name__ == "__main__":
    main()
