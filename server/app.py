import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from env.core import GDPRAuditorEnvironment
from models import Action as ActionModel

app = FastAPI(title="GDPR Auditor")
env = GDPRAuditorEnvironment()

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>GDPR Compliance Auditor</title>
    <meta name="description" content="OpenEnv environment for AI-powered GDPR compliance auditing">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 30px;
        }
        .container { max-width: 960px; margin: 0 auto; }
        h1 {
            font-size: 2rem;
            background: linear-gradient(90deg, #7f5af0, #2cb67d);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        .subtitle { color: #94a1b2; margin-bottom: 24px; font-size: 0.95rem; }
        .card {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
        }
        .card h3 { color: #fffffe; margin-bottom: 12px; font-size: 1rem; }
        select, textarea {
            width: 100%;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 8px;
            color: #fffffe;
            padding: 10px 14px;
            font-family: inherit;
            font-size: 0.9rem;
            outline: none;
            transition: border-color 0.2s;
        }
        select:focus, textarea:focus { border-color: #7f5af0; }
        textarea { height: 120px; resize: vertical; margin-bottom: 12px; }
        .btn {
            background: linear-gradient(135deg, #7f5af0, #6246d8);
            color: #fffffe;
            border: none;
            padding: 10px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.9rem;
            transition: transform 0.15s, box-shadow 0.15s;
            margin-top: 10px;
        }
        .btn:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(127,90,240,0.35); }
        .btn:active { transform: translateY(0); }
        #output {
            background: rgba(0,0,0,0.4);
            color: #2cb67d;
            padding: 16px;
            border-radius: 8px;
            white-space: pre-wrap;
            font-family: 'Cascadia Code', 'Fira Code', monospace;
            font-size: 0.85rem;
            max-height: 420px;
            overflow-y: auto;
            line-height: 1.5;
        }
        .row { display: flex; gap: 10px; align-items: center; }
        .row select { flex: 1; }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .badge-easy { background: #2cb67d33; color: #2cb67d; }
        .badge-medium { background: #e1a94033; color: #e1a940; }
        .badge-hard { background: #e0463633; color: #e04636; }
        .badge-elite { background: #7f5af033; color: #7f5af0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔒 GDPR Compliance Auditor</h1>
        <p class="subtitle">OpenEnv environment — Audit privacy policies for GDPR/CCPA compliance violations</p>

        <div class="card">
            <h3>Reset Environment</h3>
            <div class="row">
                <select id="task">
                    <option value="easy">Easy — Clause Existence Check</option>
                    <option value="medium">Medium — Purpose Mapping</option>
                    <option value="hard">Hard — Dark Pattern Detection</option>
                    <option value="elite">Elite — Multi-Document Reasoning</option>
                </select>
                <button class="btn" onclick="resetEnv()">Reset</button>
            </div>
        </div>

        <div class="card">
            <h3>Submit Finding</h3>
            <textarea id="action" placeholder="Describe your compliance finding...&#10;Examples:&#10;- Missing Right to be Forgotten clause&#10;- Health data shared with advertisers&#10;- Policy contradicts cookie section"></textarea>
            <button class="btn" onclick="submitStep()">Submit Finding</button>
        </div>

        <div class="card">
            <h3>Output</h3>
            <div id="output">Select a task and click Reset to begin...</div>
        </div>
    </div>

    <script>
        async function resetEnv() {
            const task = document.getElementById('task').value;
            try {
                const res = await fetch('/reset?task=' + task);
                const data = await res.json();
                const obs = data.observation;
                let output = '=== Task: ' + obs.task_name + ' ===\\n';
                output += 'Difficulty: ' + obs.difficulty + '\\n\\n';
                output += '--- PRIVACY POLICY ---\\n';
                if (obs.documents && obs.documents.length > 0) {
                    output += obs.documents[0].content + '\\n\\n';
                }
                output += '--- DATA PRACTICES ---\\n';
                if (obs.data_practices) {
                    obs.data_practices.forEach((dp, i) => {
                        output += (i+1) + '. ' + dp.category + ': ' + dp.purpose + ' (shared: ' + dp.shared_with_third_parties + ')\\n';
                    });
                }
                output += '\\n--- COMPLIANCE REQUIREMENTS ---\\n';
                if (obs.compliance_requirements) {
                    obs.compliance_requirements.forEach(req => {
                        output += '- ' + req + '\\n';
                    });
                }
                document.getElementById('output').textContent = output;
            } catch (e) {
                document.getElementById('output').textContent = 'Error: ' + e.message;
            }
        }

        async function submitStep() {
            const action = document.getElementById('action').value;
            if (!action.trim()) return;
            try {
                const res = await fetch('/step', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: action})
                });
                const data = await res.json();
                let output = document.getElementById('output').textContent;
                output += '\\n\\n--- STEP ' + data.observation.step + ' ---\\n';
                output += 'Reward: ' + data.reward.value.toFixed(3) + '\\n';
                output += 'Reason: ' + data.reward.reason + '\\n';
                output += 'Issues Found: ' + data.reward.issues_found + '/' + data.reward.total_issues + '\\n';
                output += 'Done: ' + data.done + '\\n';
                document.getElementById('output').textContent = output;
                document.getElementById('action').value = '';
            } catch (e) {
                document.getElementById('output').textContent += '\\nError: ' + e.message;
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the interactive web UI."""
    return HTML_CONTENT


@app.get("/json")
async def api_info():
    """API information and available endpoints."""
    return {
        "name": "GDPR Compliance Auditor",
        "version": "1.0.0",
        "description": "OpenEnv environment for AI-powered GDPR/CCPA compliance auditing",
        "tasks": ["easy", "medium", "hard", "elite"],
        "endpoints": {
            "health": "GET /health",
            "reset": "GET /reset?task=easy|medium|hard|elite",
            "step": "POST /step {message: string}",
            "state": "GET /state",
        },
    }


class ActionRequest(BaseModel):
    message: str


class ResetRequest(BaseModel):
    task: Optional[str] = "easy"


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/reset")
async def reset_get(task: str = "easy"):
    """Reset the environment for a given task (GET)."""
    try:
        obs = env.reset(task_id=task)
        return {"observation": obs.model_dump(), "done": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_post(request: Request):
    """Reset the environment for a given task (POST)."""
    try:
        body = await request.json()
        task = body.get("task", "easy") if body else "easy"
    except Exception:
        task = "easy"
    try:
        obs = env.reset(task_id=task)
        return {"observation": obs.model_dump(), "done": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(action: ActionRequest):
    """Submit an action (compliance finding) to the environment."""
    try:
        action_obj = ActionModel(message=action.message)
        obs, reward, done, info = env.step(action_obj)

        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state():
    """Get the current environment state."""
    return env.state()


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
