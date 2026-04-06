---
title: GDPR Auditor
emoji: 📋
colorFrom: purple
colorTo: red
sdk: docker
app_port: 7860
---

# 🔒 GDPR Compliance Auditor — OpenEnv Environment

**GDPR Auditor** is an OpenEnv-compatible RL environment where AI agents act as autonomous compliance officers, auditing privacy policies for GDPR/CCPA violations, detecting dark patterns, and identifying policy contradictions.

---

## The Problem It Solves

Every company needs compliance auditing to avoid massive fines:
- GDPR fines up to **€20 million** or **4% of global revenue**
- CCPA fines up to **$7,500 per violation**
- Average human compliance auditor cost: **$100,000+/year**

### The Agent's Job

1. Review privacy policy documents (single or multi-document)
2. Map data practices to stated purposes
3. Identify contradictions, missing clauses, and dark patterns
4. Report compliance violations with severity levels

---

## Tasks & Grading

| Task | Difficulty | Description | Hidden Issues |
|------|------------|-------------|---------------|
| `easy_clause_existence` | Easy | Verify mandatory GDPR clauses are present | 2 |
| `medium_purpose_mapping` | Medium | Match practices to purposes, find mismatches | 3 |
| `hard_dark_patterns` | Hard | Find contradictions within a single document | 5 |
| `elite_multi_doc_reasoning` | Elite | Cross-document contradiction detection | 6 |

### Reward Function

```
R = base_score + severity_bonus + multi_doc_bonus + exploration_bonus
```

- **Base Score**: `issues_found / total_issues`
- **Severity Bonus**: +0.25 for critical findings, +0.15 for high
- **Multi-Document Bonus**: +0.2 for elite task (cross-doc findings)
- **Exploration Bonus**: +0.02 per step (max 0.1)

All rewards are clamped to `[0.0, 1.0]`.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check → `{"status": "ok"}` |
| `/reset?task=easy` | GET | Reset environment for a task |
| `/step` | POST | Submit a finding → `{"message": "..."}` |
| `/state` | GET | Get current episode state |

### Example Usage

```bash
# Reset environment
curl "http://localhost:7860/reset?task=easy"

# Submit a compliance finding
curl -X POST "http://localhost:7860/step" \
  -H "Content-Type: application/json" \
  -d '{"message": "Missing Right to be Forgotten clause"}'

# Get current state
curl "http://localhost:7860/state"
```

---

## Action / Observation Spaces

### Observation (returned by reset/step)
```json
{
  "task_id": "easy_clause_existence",
  "task_name": "Clause Existence Check",
  "difficulty": "easy",
  "step": 0,
  "documents": [{"id": "...", "title": "...", "content": "...", "doc_type": "policy"}],
  "data_practices": [{"id": "...", "category": "...", "purpose": "...", "data_type": "...", "shared_with_third_parties": false}],
  "compliance_requirements": ["Right to be Forgotten", "Data Portability", "Contact Information"],
  "flagged_issues": [],
  "echoed_message": "Review the privacy policy..."
}
```

### Action (sent to /step)
```json
{"message": "Missing Right to be Forgotten clause"}
```

### Reward (returned from /step)
```json
{
  "value": 0.52,
  "reason": "Found 1/2 issues",
  "issues_found": 1,
  "total_issues": 2
}
```

---

## Setup & Local Development

### Prerequisites
- Python 3.10+
- `uv` or `pip`

### Install & Run
```bash
# Install dependencies
pip install -e .

# Start the server
python main.py
# → Server at http://localhost:7860
```

### Run Inference
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token-here"
export SERVER_URL="http://localhost:7860"

python inference.py
```

### Docker
```bash
docker build -t gdpr-auditor .
docker run -p 7860:7860 gdpr-auditor
```

---

## Project Structure

```
├── models.py          # Pydantic typed models (Observation, Action, Reward)
├── env/
│   ├── __init__.py
│   └── core.py        # GDPRAuditorEnvironment with 4 tasks + graders
├── main.py            # FastAPI server with all endpoints
├── inference.py       # Baseline inference script (OpenAI client)
├── openenv.yaml       # OpenEnv manifest with task definitions
├── pyproject.toml     # Dependencies
├── Dockerfile         # Container configuration
└── README.md          # This file
```

---

## Environment Variables

Copy `.env.example` to `.env` and add your Hugging Face token:

```bash
cp .env.example .env
# Then edit .env with your HF_TOKEN
```

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | Hugging Face / API key | (required) |
| `SERVER_URL` | Environment server URL | `http://localhost:7860` |

---

## License

MIT
