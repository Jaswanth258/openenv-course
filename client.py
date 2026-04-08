"""
HTTP/WebSocket client for the Customer Support Triage environment.
Import this in your training code or inference script.
"""
from typing import Optional

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from models import SupportAction, SupportObservation, SupportState


class SupportTriageEnv(EnvClient[SupportAction, SupportObservation, SupportState]):
    """
    Client for the Customer Support Triage environment.

    Async usage (recommended):
        async with SupportTriageEnv(base_url="http://localhost:8000") as env:
            result = await env.reset()
            result = await env.step(SupportAction(category="billing", priority="high"))

    Sync usage:
        with SupportTriageEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset()

    From Docker image:
        env = await SupportTriageEnv.from_docker_image(image_name)
    """

    # ── Serialisation ─────────────────────────────────────────────────────────

    def _step_payload(self, action: SupportAction) -> dict:
        """Convert action to JSON payload for the server."""
        return action.model_dump(exclude_none=True)

    # ── Deserialisation ───────────────────────────────────────────────────────

    def _parse_result(self, payload: dict) -> StepResult:
        """Parse WebSocket response into typed StepResult."""
        # The server may nest obs fields under "observation" or send them flat
        obs_data: dict = payload.get("observation") or payload

        done: bool = payload.get("done", False)
        reward: Optional[float] = payload.get("reward")

        observation = SupportObservation(
            done=done,
            reward=reward,
            task_id=obs_data.get("task_id", ""),
            task_difficulty=obs_data.get("task_difficulty", ""),
            task_description=obs_data.get("task_description", ""),
            email_subject=obs_data.get("email_subject", ""),
            email_body=obs_data.get("email_body", ""),
            email_from=obs_data.get("email_from", ""),
            context=obs_data.get("context"),
            required_fields=obs_data.get("required_fields", []),
            attempt=obs_data.get("attempt", 0),
            max_attempts=obs_data.get("max_attempts", 1),
            feedback=obs_data.get("feedback", ""),
            score_so_far=obs_data.get("score_so_far", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: dict) -> SupportState:
        """Parse state payload into typed SupportState."""
        return SupportState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            task_difficulty=payload.get("task_difficulty", ""),
            score=payload.get("score", 0.0),
            attempt=payload.get("attempt", 0),
        )
