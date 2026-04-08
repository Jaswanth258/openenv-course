"""
Typed data models for the Customer Support Triage environment.
"""
from typing import List, Optional

from openenv.core.env_server import Action, Observation, State


class SupportAction(Action):
    """
    Action taken by the agent on a support ticket.
    Which fields are required depends on the task (see `required_fields` in the observation).

    Task 1 (easy):   fill `category` and `priority`
    Task 2 (medium): fill `response_text` and `resolution_status`
    Task 3 (hard):   fill `resolution_status`, `escalation_team`,
                          `identified_issues`, and `escalation_reason`
    """

    # ── Task 1: Classification ──────────────────────────────────────────────
    category: Optional[str] = None
    """Ticket category: billing | technical | account | general | hr | legal"""

    priority: Optional[str] = None
    """Ticket urgency: low | medium | high | urgent"""

    # ── Task 2: Response Drafting ───────────────────────────────────────────
    response_text: Optional[str] = None
    """Full drafted response to send to the customer."""

    resolution_status: Optional[str] = None
    """How the episode is resolved: resolved | needs_followup | escalate"""

    # ── Task 3: Escalation ──────────────────────────────────────────────────
    escalation_team: Optional[str] = None
    """Team(s) to escalate to (comma-separated): security | legal | management | billing | technical"""

    escalation_reason: Optional[str] = None
    """Clear, actionable escalation justification (at least 30 characters)."""

    identified_issues: Optional[List[str]] = None
    """
    Issue tags found in the ticket, e.g.:
      ['account_compromise', 'data_breach', 'unauthorized_transactions',
       'legal_threat', 'enterprise_account']
    """


class SupportObservation(Observation):
    """
    What the agent sees at each step.
    `done` and `reward` are inherited from the base Observation class.
    """

    task_id: str
    """Current task identifier (task_1_easy | task_2_medium | task_3_hard)."""

    task_difficulty: str
    """Difficulty level: easy | medium | hard"""

    task_description: str
    """Explicit description of what the agent must do and which fields to fill."""

    email_subject: str
    email_body: str
    email_from: str

    context: Optional[str] = None
    """Extra context: Knowledge Base articles, account info, etc."""

    required_fields: List[str]
    """Fields the agent MUST provide in its SupportAction."""

    attempt: int = 0
    """How many actions have been taken so far (0 on first observation)."""

    max_attempts: int = 1
    """Maximum allowed actions for this episode."""

    feedback: str = ""
    """Grader feedback from the previous step (empty at episode start)."""

    score_so_far: float = 0.0
    """Cumulative score for the current episode."""


class SupportState(State):
    """
    Episode metadata returned by state().
    `episode_id` and `step_count` are inherited from the base State class.
    """

    task_id: str = ""
    task_difficulty: str = ""
    score: float = 0.0
    attempt: int = 0
