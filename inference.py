"""
Baseline inference script for the Customer Support Triage environment.

MANDATORY ENVIRONMENT VARIABLES:
  API_BASE_URL      LLM inference endpoint  (default: HF router)
  MODEL_NAME        Model identifier         (default: Qwen2.5-72B-Instruct)
  HF_TOKEN          Hugging Face / API key
  LOCAL_IMAGE_NAME  Docker image name for the environment

STDOUT FORMAT (strictly enforced for evaluation):
  [START] task=<task_id> env=support-triage model=<model>
  [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Runs all 3 tasks sequentially. Total runtime target: < 5 minutes.
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

# ── Load env vars ──────────────────────────────────────────────────────────────
API_KEY        = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL   = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME     = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
IMAGE_NAME     = os.getenv("LOCAL_IMAGE_NAME", "")
BENCHMARK      = "support-triage"
SUCCESS_THRESHOLD = 0.5   # score >= 0.5 counts as success

# ── Logging helpers (mandatory format) ────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={err_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert customer support manager AI.
    You will receive a support ticket and must respond with a JSON object.

    The required fields change per task — they are listed in the task_description
    and also in the `required_fields` array of the observation.

    TASK 1 (classification):
      { "category": "<billing|technical|account|general|hr|legal>",
        "priority": "<low|medium|high|urgent>" }

    TASK 2 (response drafting):
      { "response_text": "<full customer-facing response>",
        "resolution_status": "<resolved|needs_followup|escalate>" }

    TASK 3 (escalation decision):
      { "resolution_status": "<resolved|needs_followup|escalate>",
        "escalation_team": "<security,legal,management,...>",
        "identified_issues": ["account_compromise","data_breach","unauthorized_transactions","legal_threat"],
        "escalation_reason": "<specific reason of at least 30 characters>" }

    Rules:
    - Respond with ONLY the JSON object. No markdown, no explanation.
    - Use only the field names shown above.
    - For Task 3, set resolution_status to 'escalate' if the ticket involves
      account compromise, data breach, significant financial fraud, or legal threats.
""").strip()

# ── LLM call ───────────────────────────────────────────────────────────────────

def build_user_prompt(obs) -> str:
    lines = [
        f"Task: {obs.task_id}  |  Difficulty: {obs.task_difficulty}",
        f"Instructions: {obs.task_description}",
        "",
        f"From: {obs.email_from}",
        f"Subject: {obs.email_subject}",
        f"Body:\n{obs.email_body}",
    ]
    if obs.context:
        lines += ["", obs.context]
    lines += [
        "",
        f"Required fields (fill ALL of these): {obs.required_fields}",
    ]
    if obs.feedback:
        lines += ["", f"Previous feedback: {obs.feedback}"]
    return "\n".join(lines)


def call_llm(client: OpenAI, obs) -> dict:
    """Call the LLM and parse its JSON response into an action dict."""
    user_prompt = build_user_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=512,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        # Return a safe default action so the episode can still complete
        return {"category": "general", "priority": "medium",
                "resolution_status": "needs_followup",
                "response_text": "Thank you for contacting us. We are looking into this.",
                "escalation_team": "technical",
                "escalation_reason": "Requires further investigation by the technical team.",
                "identified_issues": ["account_compromise"]}

# ── Episode runner ─────────────────────────────────────────────────────────────

async def run_episode(env, client: OpenAI, task_id: str) -> dict:
    """Run one full episode for the given task. Returns metrics dict."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg: Optional[str] = None

    try:
        # Reset gets the next task in the cycle; the obs tells us the task_id
        result = await env.reset()
        obs = result.observation

        for step in range(1, obs.max_attempts + 2):  # +1 guard
            if result.done:
                break

            action_dict = call_llm(client, obs)
            action_str = json.dumps(action_dict, separators=(",", ":"))

            # Import here to avoid circular import at module level
            from models import SupportAction
            action = SupportAction(**{
                k: v for k, v in action_dict.items()
                if k in SupportAction.model_fields
            })

            result = await env.step(action)
            obs = result.observation
            reward = float(result.reward or 0.0)
            done = result.done

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        score = rewards[-1] if rewards else 0.0
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        error_msg = str(exc)
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        if not rewards:
            rewards = [0.0]
        score = 0.0

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score, "success": success, "steps": steps_taken}


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Import client here (works whether package is installed or run from repo root)
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from client import SupportTriageEnv

    task_ids = ["task_1_easy", "task_2_medium", "task_3_hard"]
    all_results = []

    # Start one Docker container; the env cycles through the 3 tasks on each reset()
    env = await SupportTriageEnv.from_docker_image(IMAGE_NAME)

    try:
        for task_id in task_ids:
            result = await run_episode(env, client, task_id)
            all_results.append(result)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

    # Summary
    avg_score = sum(r["score"] for r in all_results) / len(all_results)
    print(f"\n[SUMMARY] avg_score={avg_score:.3f}", flush=True)
    for r in all_results:
        print(f"  {r['task_id']}: score={r['score']:.3f} success={r['success']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
