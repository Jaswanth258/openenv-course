# Customer Support Triage — OpenEnv Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

> **Real-world task:** LLM agents act as AI customer support managers — classifying tickets, drafting knowledge-base responses, and making escalation decisions on realistic enterprise scenarios.

---

## Why This Environment?

Customer support is a [multi-billion-dollar](https://www.grandviewresearch.com/industry-analysis/customer-service-software-market) industry where AI agents can provide immediate value. Every company processes support tickets, yet no standardised OpenEnv environment exists for this domain.  
This environment fills that gap with realistic scenarios, deterministic graders, and meaningful partial-reward signals for RL training.

---

## Action / Observation Space

### `SupportAction`

| Field | Type | Tasks | Description |
|---|---|---|---|
| `category` | `str` | 1 | `billing \| technical \| account \| general \| hr \| legal` |
| `priority` | `str` | 1 | `low \| medium \| high \| urgent` |
| `response_text` | `str` | 2 | Full customer-facing response draft |
| `resolution_status` | `str` | 2, 3 | `resolved \| needs_followup \| escalate` |
| `escalation_team` | `str` | 3 | Comma-separated: `security \| legal \| management \| billing \| technical` |
| `escalation_reason` | `str` | 3 | Specific justification (≥ 30 chars) |
| `identified_issues` | `List[str]` | 3 | e.g. `["account_compromise", "data_breach"]` |

### `SupportObservation`

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | `task_1_easy \| task_2_medium \| task_3_hard` |
| `task_difficulty` | `str` | `easy \| medium \| hard` |
| `task_description` | `str` | Explicit instructions for the agent |
| `email_subject` | `str` | Ticket subject line |
| `email_body` | `str` | Full ticket body |
| `email_from` | `str` | Sender address |
| `context` | `str \| None` | KB articles, account info |
| `required_fields` | `List[str]` | Fields the agent must fill |
| `feedback` | `str` | Grader feedback from previous step |
| `score_so_far` | `float` | Cumulative episode score |
| `done` | `bool` | Episode completed flag |
| `reward` | `float \| None` | Step reward (set when `done=True`) |

---

## Tasks

### Task 1 — Ticket Classification (Easy)

**Scenario:** A customer emails about a double billing charge.  
**Agent must:** set the correct `category` and `priority`.  
**Grader:**
- `category == "billing"` → +0.50
- `priority == "high"` → +0.50

**Expected score for GPT-4-class model:** ~1.0  
**Expected score for random baseline:** ~0.12

---

### Task 2 — Response Drafting (Medium)

**Scenario:** CTO reports 100% API 500 errors blocking a product release. A KB article is provided.  
**Agent must:** draft a response covering KB troubleshooting steps and set `resolution_status`.  
**Grader (5 KB points × 0.14 each = 0.70; status = 0.30):**
- Each KB keyword group covered: +0.14
- `resolution_status` correct (`needs_followup` or `escalate`): +0.30

**Expected score for GPT-4-class model:** ~0.60–0.80  
**Expected score for random baseline:** ~0.10

---

### Task 3 — Security Escalation Decision (Hard)

**Scenario:** VP of Operations at an enterprise company reports: unauthorized account access, $5,590 in fraudulent transactions, suspected data breach via phishing, and a legal threat — all within 24 hours.  
**Agent must:** set `resolution_status = escalate`, name the security team, identify 4+ issue types, and write a substantive escalation reason.  
**Grader (multi-component):**

| Component | Points |
|---|---|
| `resolution_status == "escalate"` | 0.25 |
| `escalation_team` includes `"security"` | 0.20 |
| Issue: `account_compromise` identified | 0.10 |
| Issue: `unauthorized_transactions` identified | 0.10 |
| Issue: `data_breach` identified | 0.10 |
| Issue: `legal_threat` identified | 0.10 |
| Escalation reason substantive (≥30 chars, 2+ key terms) | 0.15 |

**Expected score for GPT-4-class model:** ~0.60–0.80  
**Expected score for random baseline:** ~0.15

---

## Reward Design

Rewards are **fractional, not binary**. Each graded dimension contributes independently, providing gradient signal even when the agent is partially correct. This enables meaningful RL training signals from the very first episode.

- Task 1: 0.0 → 0.5 → 1.0 (step-wise per field)
- Task 2: 0.00 → 1.00 in increments of ~0.14 per KB point
- Task 3: 0.00 → 1.00 across 7 independent scoring dimensions

---

## Setup

### Quick Start (Remote HF Space)

```python
from support_triage_env import SupportAction, SupportTriageEnv

async with SupportTriageEnv(base_url="https://your-space.hf.space") as env:
    result = await env.reset()
    obs = result.observation
    print(obs.email_subject)   # "Double charge on my account…"
    print(obs.required_fields) # ['category', 'priority']

    result = await env.step(SupportAction(category="billing", priority="high"))
    print(result.reward)       # 1.0
```

### Local Docker

```bash
# Option 1: Pull from HF Spaces registry
docker pull registry.hf.space/<your-username>-support-triage-env:latest
docker run -d -p 8000:8000 registry.hf.space/<your-username>-support-triage-env:latest

# Option 2: Build from source
git clone https://github.com/<your-username>/support-triage-env
cd support-triage-env
docker build -t support-triage-env:latest .
docker run -d -p 8000:8000 support-triage-env:latest

# Verify
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" -d '{}'
```

### Local Development (no Docker)

```bash
pip install openenv-core>=0.2.3
git clone https://github.com/<your-username>/support-triage-env
cd support-triage-env
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Install Client Only

```bash
pip install git+https://huggingface.co/spaces/<your-username>/support-triage-env
```

---

## Baseline Inference

```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export LOCAL_IMAGE_NAME=support-triage-env:latest

python inference.py
```

Expected output:
```
[START] task=task_1_easy env=support-triage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"category":"billing","priority":"high"} reward=1.00 done=true error=null
[END] success=true steps=1 score=1.000 rewards=1.00
[START] task=task_2_medium env=support-triage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"response_text":"...","resolution_status":"needs_followup"} reward=0.72 done=true error=null
[END] success=true steps=1 score=0.720 rewards=0.72
[START] task=task_3_hard env=support-triage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"resolution_status":"escalate","escalation_team":"security,legal",...} reward=0.70 done=true error=null
[END] success=true steps=1 score=0.700 rewards=0.70

[SUMMARY] avg_score=0.807
```

### Baseline Scores (Qwen2.5-72B-Instruct)

| Task | Expected Score | Difficulty |
|------|---------------|------------|
| task_1_easy | ~1.00 | Easy |
| task_2_medium | ~0.60–0.80 | Medium |
| task_3_hard | ~0.50–0.70 | Hard |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode (JSON body: `{"task_id": "task_1_easy"}`) |
| `/step` | POST | Submit action |
| `/state` | GET | Get episode metadata |
| `/health` | GET | Health check |
| `/ws` | WebSocket | Persistent session (used by client) |
| `/docs` | GET | Interactive OpenAPI docs |
| `/web` | GET | Web UI for manual testing |

---

## Deploy to HF Spaces

```bash
cd support-triage-env
openenv push --repo-id your-username/support-triage-env
```

Then set HF Space variables:
- `WORKERS=2`
- `MAX_CONCURRENT_ENVS=100`

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKERS` | 1 | Uvicorn worker processes |
| `PORT` | 8000 | Server port |
| `HOST` | 0.0.0.0 | Bind address |
| `MAX_CONCURRENT_ENVS` | 100 | Max WebSocket sessions |

---

## Validation

```bash
# Pre-submission check
pip install openenv-core
openenv validate

# Full validation (HF Space must be deployed)
./validate-submission.sh https://your-username-support-triage-env.hf.space .
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE)
