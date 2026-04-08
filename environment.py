"""
Core environment logic and graders for the Customer Support Triage environment.
"""
import uuid
from typing import Optional, Tuple

from openenv.core.env_server import Environment

from models import SupportAction, SupportObservation, SupportState

# ── Task Definitions ──────────────────────────────────────────────────────────

TASKS: dict = {
    # ------------------------------------------------------------------
    # TASK 1 — Easy: Ticket Classification
    # ------------------------------------------------------------------
    "task_1_easy": {
        "difficulty": "easy",
        "description": (
            "Classify this support ticket. Provide the correct CATEGORY "
            "(billing | technical | account | general | hr | legal) "
            "and PRIORITY (low | medium | high | urgent)."
        ),
        "email_from": "john.smith@example.com",
        "email_subject": "Double charge on my account — Invoice #45892",
        "email_body": (
            "Hello,\n\n"
            "I noticed two identical charges of $49.99 on my credit card "
            "statement this month for my subscription renewal.\n\n"
            "The charges appeared on November 3rd (Order #ORD-78923) and "
            "November 4th (Order #ORD-78931). My subscription auto-renews "
            "once a month, so I should only have been charged once.\n\n"
            "Please investigate and refund the duplicate charge as soon as possible.\n\n"
            "Thank you,\nJohn Smith"
        ),
        "context": None,
        "required_fields": ["category", "priority"],
        "max_attempts": 1,
        "expected": {"category": "billing", "priority": "high"},
    },

    # ------------------------------------------------------------------
    # TASK 2 — Medium: Response Drafting
    # ------------------------------------------------------------------
    "task_2_medium": {
        "difficulty": "medium",
        "description": (
            "Draft a helpful troubleshooting response using the Knowledge Base "
            "article provided in context. Fill RESPONSE_TEXT with specific steps "
            "and set RESOLUTION_STATUS (resolved | needs_followup | escalate)."
        ),
        "email_from": "sarah.chen@techstartup.io",
        "email_subject": "API integration returning 500 errors — blocking our release",
        "email_body": (
            "Hi Support Team,\n\n"
            "Our API integration has been returning 500 errors since this morning. "
            "We are using your REST API v2. The errors started at approximately "
            "9:15 AM EST. Nothing changed on our end.\n\n"
            "Error details:\n"
            "  Endpoint: POST /api/v2/data/upload\n"
            "  Response: {\"error\": \"Internal Server Error\", \"code\": 500}\n"
            "  Frequency: 100% of requests failing\n\n"
            "This is blocking our product release scheduled for today.\n\n"
            "Sarah Chen — CTO, TechStartup"
        ),
        "context": (
            "=== KB: API 500 Error Troubleshooting Guide ===\n"
            "1. Check our Status Page (status.platform.com) — there may be an active incident.\n"
            "2. Verify your API key is valid and not expired via Settings > API Keys.\n"
            "3. API v2.1 introduced a breaking change: requests MUST include "
            "   Content-Type: application/json. A missing header causes 500 errors.\n"
            "4. For debugging, add header X-Debug-Mode: true to get a detailed error payload.\n"
            "5. If the issue persists, email logs (timestamps + masked API key) "
            "   to api-support@platform.com.\n"
            "6. For release-blocking issues, contact api-escalations@platform.com "
            "   for priority handling.\n"
        ),
        "required_fields": ["response_text", "resolution_status"],
        "max_attempts": 1,
        "expected": {
            "resolution_status_ok": ["needs_followup", "escalate"],
            "kb_keywords": [
                (["status.platform.com", "status page", "active incident"], "Check status page"),
                (["api key", "expired", "settings > api keys", "key validity"], "Verify API key"),
                (["content-type", "application/json", "v2.1", "breaking change"], "Content-Type header"),
                (["x-debug-mode", "debug mode", "debug header", "x-debug"], "Enable debug mode"),
                (["api-support@platform.com", "api-escalations@platform.com", "email logs"], "Contact support email"),
            ],
        },
    },

    # ------------------------------------------------------------------
    # TASK 3 — Hard: Security Escalation Decision
    # ------------------------------------------------------------------
    "task_3_hard": {
        "difficulty": "hard",
        "description": (
            "This is a high-stakes enterprise security incident. Analyze the ticket carefully "
            "and fill in ALL four fields:\n"
            "  RESOLUTION_STATUS — set to 'escalate' if warranted.\n"
            "  ESCALATION_TEAM   — comma-separated: security | legal | management | billing | technical.\n"
            "  IDENTIFIED_ISSUES — array of issue tags from: "
            "['account_compromise', 'data_breach', 'unauthorized_transactions', "
            "'legal_threat', 'enterprise_account'].\n"
            "  ESCALATION_REASON — specific, actionable justification (≥30 characters)."
        ),
        "email_from": "marcus.williams@enterprise-corp.com",
        "email_subject": "URGENT: Unauthorized account access and suspicious transactions",
        "email_body": (
            "I need immediate assistance. Our company account "
            "(Enterprise Plan, Account #ENT-2847) has been compromised:\n\n"
            "1. Someone changed our admin email to an unknown address last night.\n"
            "2. Three unauthorized transactions in the past 24 h: "
            "$1,200 + $3,500 + $890 = $5,590 we did NOT authorize.\n"
            "3. Our CFO received a phishing email referencing our exact account ID — "
            "   suggesting our customer data was leaked or sold.\n"
            "4. IT detected login attempts from Eastern European IPs inconsistent "
            "   with our US-based operations.\n\n"
            "We have 200 employees on this platform. Our legal team will pursue action "
            "if this is not resolved within 24 hours.\n\n"
            "Marcus Williams — VP of Operations, Enterprise Corp"
        ),
        "context": (
            "=== Account Info ===\n"
            "Account #ENT-2847 — Enterprise Plan, 200 seats\n"
            "SLA Tier: Enterprise (4-hour response, dedicated account manager)\n"
            "MRR: $4,800/month\n"
        ),
        "required_fields": [
            "resolution_status",
            "escalation_team",
            "identified_issues",
            "escalation_reason",
        ],
        "max_attempts": 1,
        "expected": {
            "resolution_status": "escalate",
            "must_include_team": "security",
        },
    },
}

TASK_CYCLE = ["task_1_easy", "task_2_medium", "task_3_hard"]


# ── Environment ───────────────────────────────────────────────────────────────

class SupportTriageEnvironment(Environment):
    """
    Customer Support Ticket Resolution Environment.

    Agents act as AI-powered support managers across three realistic tasks
    ranging from simple classification (easy) to complex escalation decisions (hard).
    Reward is fractional (0.0–1.0) and reflects partial credit across multiple
    grading dimensions, providing meaningful gradient signal throughout training.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._cycle_index: int = 0
        self._task_id: str = TASK_CYCLE[0]
        self._task: dict = TASKS[self._task_id]
        self._done: bool = False
        self._score: float = 0.0
        self._attempt: int = 0
        self._feedback: str = ""
        self._state = SupportState()

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> SupportObservation:
        """Start a new episode. Cycles task_1→task_2→task_3 if task_id is omitted."""
        if task_id and task_id in TASKS:
            self._task_id = task_id
        else:
            self._task_id = TASK_CYCLE[self._cycle_index % len(TASK_CYCLE)]
            self._cycle_index += 1

        self._task = TASKS[self._task_id]
        self._done = False
        self._score = 0.0
        self._attempt = 0
        self._feedback = ""
        ep_id = episode_id or str(uuid.uuid4())

        self._state = SupportState(
            episode_id=ep_id,
            step_count=0,
            task_id=self._task_id,
            task_difficulty=self._task["difficulty"],
            score=0.0,
            attempt=0,
        )
        return self._build_obs()

    def step(
        self,
        action: SupportAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> SupportObservation:
        """Execute an agent action and return graded observation."""
        if self._done:
            return self._build_obs(feedback="Episode finished. Call reset() to start a new one.")

        self._attempt += 1
        self._state.step_count += 1

        score, feedback = self._grade(action)
        self._score = score
        self._feedback = feedback
        self._done = True  # all tasks are single-turn

        self._state.score = score
        self._state.attempt = self._attempt
        return self._build_obs(feedback=feedback)

    @property
    def state(self) -> SupportState:
        return self._state

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_obs(self, feedback: str = "") -> SupportObservation:
        t = self._task
        return SupportObservation(
            done=self._done,
            reward=round(self._score, 4) if self._done else None,
            task_id=self._task_id,
            task_difficulty=t["difficulty"],
            task_description=t["description"],
            email_subject=t["email_subject"],
            email_body=t["email_body"],
            email_from=t["email_from"],
            context=t.get("context"),
            required_fields=t["required_fields"],
            attempt=self._attempt,
            max_attempts=t["max_attempts"],
            feedback=feedback or self._feedback,
            score_so_far=round(self._score, 4),
        )

    def _grade(self, action: SupportAction) -> Tuple[float, str]:
        if self._task_id == "task_1_easy":
            return self._grade_task_1(action)
        if self._task_id == "task_2_medium":
            return self._grade_task_2(action)
        if self._task_id == "task_3_hard":
            return self._grade_task_3(action)
        return 0.0, "Unknown task."

    # ── Graders ───────────────────────────────────────────────────────────────

    def _grade_task_1(self, action: SupportAction) -> Tuple[float, str]:
        """
        Score = category_correct (0.50) + priority_correct (0.50)
        Partial credit: 0.0 | 0.5 | 1.0
        """
        score = 0.0
        parts = []
        exp = self._task["expected"]

        cat = (action.category or "").lower().strip()
        if cat == exp["category"]:
            score += 0.50
            parts.append(f"✓ category='{cat}' (+0.50)")
        else:
            parts.append(f"✗ category: expected='{exp['category']}', got='{cat}' (+0.00)")

        pri = (action.priority or "").lower().strip()
        if pri == exp["priority"]:
            score += 0.50
            parts.append(f"✓ priority='{pri}' (+0.50)")
        else:
            parts.append(f"✗ priority: expected='{exp['priority']}', got='{pri}' (+0.00)")

        return round(score, 2), " | ".join(parts)

    def _grade_task_2(self, action: SupportAction) -> Tuple[float, str]:
        """
        Score = KB coverage (0.70, 5 keyword groups × 0.14) + status (0.30)
        Partial credit: each KB point covered adds ~0.14
        """
        score = 0.0
        parts = []
        exp = self._task["expected"]
        response = (action.response_text or "").lower()
        kw_groups = exp["kb_keywords"]
        per_group = round(0.70 / len(kw_groups), 4)

        for kws, label in kw_groups:
            if any(kw.lower() in response for kw in kws):
                score += per_group
                parts.append(f"✓ '{label}' covered (+{per_group:.2f})")
            else:
                parts.append(f"✗ '{label}' missing (+0.00)")

        status = (action.resolution_status or "").lower().strip()
        if status in exp["resolution_status_ok"]:
            score += 0.30
            parts.append(f"✓ resolution_status='{status}' (+0.30)")
        elif status == "resolved":
            parts.append("✗ 'resolved' is incorrect — issue needs follow-up (+0.00)")
        else:
            parts.append(f"✗ resolution_status='{status}' invalid (+0.00)")

        return round(min(score, 1.0), 2), " | ".join(parts)

    def _grade_task_3(self, action: SupportAction) -> Tuple[float, str]:
        """
        Score breakdown (total 1.0):
          escalate decision    0.25
          security in team     0.20
          4 issue tags × 0.10  0.40
          reason quality       0.15
        Partial credit at every dimension.
        """
        score = 0.0
        parts = []

        # ① Escalation decision
        status = (action.resolution_status or "").lower().strip()
        if status == "escalate":
            score += 0.25
            parts.append("✓ resolution_status='escalate' (+0.25)")
        else:
            parts.append(f"✗ should be 'escalate', got='{status}' (+0.00)")

        # ② Escalation team includes security
        team = (action.escalation_team or "").lower()
        if "security" in team:
            score += 0.20
            parts.append(f"✓ escalation_team includes 'security' (+0.20)")
        else:
            parts.append(f"✗ 'security' missing from escalation_team: got='{action.escalation_team}' (+0.00)")

        # ③ Issue identification (0.10 per issue, 4 issues max = 0.40)
        issues_list = [str(i).lower() for i in (action.identified_issues or [])]
        reason_text = (action.escalation_reason or "").lower()
        combined = " ".join(issues_list) + " " + reason_text

        issue_checks = [
            (["account_compromise", "account compromise", "compromised", "unauthorized access"], "account_compromise"),
            (["unauthorized_transaction", "unauthorized transaction", "financial fraud", "unauthorized charge"], "unauthorized_transactions"),
            (["data_breach", "data breach", "data leak", "phishing", "data leaked"], "data_breach"),
            (["legal_threat", "legal threat", "legal action", "lawsuit", "pursue action"], "legal_threat"),
        ]
        for kws, label in issue_checks:
            if any(kw in combined for kw in kws):
                score += 0.10
                parts.append(f"✓ issue '{label}' identified (+0.10)")
            else:
                parts.append(f"✗ issue '{label}' missed (+0.00)")

        # ④ Escalation reason quality
        reason = (action.escalation_reason or "")
        quality_kws = ["breach", "compromise", "security", "unauthorized", "enterprise", "legal", "fraud", "data"]
        hits = sum(1 for kw in quality_kws if kw in reason.lower())
        if len(reason) >= 30 and hits >= 2:
            score += 0.15
            parts.append(f"✓ escalation reason substantive ({hits} key terms) (+0.15)")
        elif len(reason) >= 10:
            score += 0.05
            parts.append(f"~ escalation reason brief, needs more detail (+0.05)")
        else:
            parts.append("✗ escalation reason missing or too short (+0.00)")

        return round(min(score, 1.0), 2), " | ".join(parts)
