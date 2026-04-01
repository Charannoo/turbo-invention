"""OpenEnv Server for Code Review Environment."""

import os
import json
import uuid
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from code_review_env.environment import CodeReviewEnv
from code_review_env.models import Observation, Action, Reward, StepResult


sessions: Dict[str, CodeReviewEnv] = {}


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    session_id: str
    action: Action


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    sessions.clear()


app = FastAPI(
    title="Code Review Environment",
    description="OpenEnv compliant code review environment for AI agents",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "environment": "code-review",
        "version": "1.0.0"
    }


@app.post("/reset")
async def reset(request: ResetRequest):
    """Reset the environment and start a new episode."""
    try:
        session_id = str(uuid.uuid4())
        env = CodeReviewEnv(task_id=request.task_id, seed=request.seed)
        result = env.reset(task_id=request.task_id)

        sessions[session_id] = env

        return {
            "session_id": session_id,
            "observation": result.observation.model_dump(),
            "reward": result.reward.model_dump(),
            "done": result.done,
            "info": result.info,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(request: StepRequest):
    """Execute one step in the environment."""
    env = sessions.get(request.session_id)
    if env is None:
        raise HTTPException(
            status_code=404, 
            detail=f"Session not found: {request.session_id}"
        )

    try:
        result = env.step(request.action)

        if result.done:
            del sessions[request.session_id]

        return {
            "observation": result.observation.model_dump(),
            "reward": result.reward.model_dump(),
            "done": result.done,
            "info": result.info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def get_state(session_id: str):
    """Get current environment state."""
    env = sessions.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )

    return env.state()


@app.get("/tasks")
async def list_tasks():
    """List all available tasks."""
    env = CodeReviewEnv()
    return {"tasks": env.get_task_list()}


@app.get("/observation-space")
async def observation_space():
    """Get observation space definition."""
    env = CodeReviewEnv()
    return env.observation_space


@app.get("/action-space")
async def action_space():
    """Get action space definition."""
    env = CodeReviewEnv()
    return env.action_space


def start_server(host: str = "0.0.0.0", port: int = 7860):
    """Start the OpenEnv server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    start_server(port=port)
