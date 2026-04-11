"""
Unit tests for the Customer Support Triage environment graders.

Verifies:
  - Deterministic scoring across all 3 tasks
  - Partial credit produces varied, non-trivial scores
  - Perfect actions score near 1.0, empty actions score near 0.0
  - Score clamping keeps all results in (0, 1) exclusive
  - reset() produces clean state
  - Episode boundaries are respected (double-step returns same result)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import SupportTriageEnvironment
from models import SupportAction


def test_task1_perfect():
    """Perfect classification should score near maximum."""
    env = SupportTriageEnvironment()
    env.reset(task_id="task_1_easy")
    obs = env.step(SupportAction(category="billing", priority="high"))
    assert 0.95 <= obs.reward <= 0.99, f"Expected ~0.99, got {obs.reward}"
    assert obs.done is True


def test_task1_partial():
    """Getting only category right should give partial credit (~0.50)."""
    env = SupportTriageEnvironment()
    env.reset(task_id="task_1_easy")
    obs = env.step(SupportAction(category="billing", priority="low"))
    assert 0.49 <= obs.reward <= 0.51, f"Expected ~0.50, got {obs.reward}"


def test_task1_wrong():
    """Completely wrong classification should score near minimum."""
    env = SupportTriageEnvironment()
    env.reset(task_id="task_1_easy")
    obs = env.step(SupportAction(category="general", priority="low"))
    assert obs.reward <= 0.02, f"Expected ~0.01, got {obs.reward}"


def test_task2_with_kb_coverage():
    """Response mentioning multiple KB points should get partial credit."""
    env = SupportTriageEnvironment()
    env.reset(task_id="task_2_medium")
    obs = env.step(SupportAction(
        response_text="Please check status.platform.com for any active incidents. "
                       "Also verify your API key is valid. Add the Content-Type: application/json header.",
        resolution_status="needs_followup"
    ))
    # Should get 3/5 KB points (0.42) + status (0.30) = ~0.72
    assert 0.50 <= obs.reward <= 0.90, f"Expected ~0.72, got {obs.reward}"
    assert obs.done is True


def test_task2_empty_response():
    """Empty response should score near minimum."""
    env = SupportTriageEnvironment()
    env.reset(task_id="task_2_medium")
    obs = env.step(SupportAction(response_text="", resolution_status="resolved"))
    assert obs.reward <= 0.05, f"Expected near 0, got {obs.reward}"


def test_task3_full_escalation():
    """Complete escalation with all identified issues should score high."""
    env = SupportTriageEnvironment()
    env.reset(task_id="task_3_hard")
    obs = env.step(SupportAction(
        resolution_status="escalate",
        escalation_team="security,legal",
        identified_issues=["account_compromise", "data_breach", "unauthorized_transactions", "legal_threat"],
        escalation_reason="Enterprise account compromised with unauthorized financial transactions and data breach. Legal threat from customer requires immediate security and legal team involvement."
    ))
    assert 0.85 <= obs.reward <= 0.99, f"Expected ~0.99, got {obs.reward}"
    assert obs.done is True


def test_task3_partial_escalation():
    """Partial escalation (correct status but missing issues) gives partial credit."""
    env = SupportTriageEnvironment()
    env.reset(task_id="task_3_hard")
    obs = env.step(SupportAction(
        resolution_status="escalate",
        escalation_team="security",
        identified_issues=["account_compromise"],
        escalation_reason="Account has been compromised, needs security review."
    ))
    # Should get: escalate(0.25) + security(0.20) + 1 issue(0.10) + reason(0.15) = 0.70
    assert 0.40 <= obs.reward <= 0.80, f"Expected ~0.70, got {obs.reward}"


def test_task3_no_escalation():
    """Not escalating a security incident should score poorly."""
    env = SupportTriageEnvironment()
    env.reset(task_id="task_3_hard")
    obs = env.step(SupportAction(
        resolution_status="resolved",
        escalation_team="",
        identified_issues=[],
        escalation_reason=""
    ))
    assert obs.reward <= 0.05, f"Expected near minimum, got {obs.reward}"


def test_score_clamping():
    """Scores must be strictly between 0 and 1 (exclusive)."""
    env = SupportTriageEnvironment()

    # Perfect score should be clamped to 0.99, not 1.0
    env.reset(task_id="task_1_easy")
    obs = env.step(SupportAction(category="billing", priority="high"))
    assert obs.reward < 1.0, f"Score must be < 1.0, got {obs.reward}"
    assert obs.reward > 0.0, f"Score must be > 0.0, got {obs.reward}"

    # Zero score should be clamped to 0.01, not 0.0
    env.reset(task_id="task_1_easy")
    obs = env.step(SupportAction(category="wrong", priority="wrong"))
    assert obs.reward > 0.0, f"Score must be > 0.0, got {obs.reward}"
    assert obs.reward < 1.0, f"Score must be < 1.0, got {obs.reward}"


def test_reset_produces_clean_state():
    """After reset(), the environment should be in a fresh state."""
    env = SupportTriageEnvironment()

    # Run task 1
    env.reset(task_id="task_1_easy")
    obs1 = env.step(SupportAction(category="billing", priority="high"))
    assert obs1.done is True

    # Reset to task 2
    obs2_start = env.reset(task_id="task_2_medium")
    assert obs2_start.done is False
    assert obs2_start.reward is None
    assert obs2_start.task_id == "task_2_medium"
    assert obs2_start.attempt == 0
    assert obs2_start.score_so_far == 0.0


def test_double_step_after_done():
    """Stepping after episode is done should return the same terminal observation."""
    env = SupportTriageEnvironment()
    env.reset(task_id="task_1_easy")
    obs1 = env.step(SupportAction(category="billing", priority="high"))
    assert obs1.done is True

    # Second step should not crash, should return done=True with feedback
    obs2 = env.step(SupportAction(category="technical", priority="low"))
    assert obs2.done is True
    assert "finished" in obs2.feedback.lower() or "reset" in obs2.feedback.lower()


def test_deterministic_scoring():
    """Same action on same task should always produce exactly the same score."""
    scores = []
    for _ in range(5):
        env = SupportTriageEnvironment()
        env.reset(task_id="task_2_medium")
        obs = env.step(SupportAction(
            response_text="Check status.platform.com and use X-Debug-Mode header.",
            resolution_status="needs_followup"
        ))
        scores.append(obs.reward)

    assert len(set(scores)) == 1, f"Scores should be identical across runs: {scores}"


def test_task_cycling():
    """Without explicit task_id, reset() should cycle through tasks in order."""
    env = SupportTriageEnvironment()
    obs1 = env.reset()
    assert obs1.task_id == "task_1_easy"
    env.step(SupportAction(category="billing", priority="high"))

    obs2 = env.reset()
    assert obs2.task_id == "task_2_medium"
    env.step(SupportAction(response_text="test", resolution_status="resolved"))

    obs3 = env.reset()
    assert obs3.task_id == "task_3_hard"


def test_empty_action_no_crash():
    """Completely empty action should still produce a valid graded observation."""
    env = SupportTriageEnvironment()
    for task_id in ["task_1_easy", "task_2_medium", "task_3_hard"]:
        env.reset(task_id=task_id)
        obs = env.step(SupportAction())  # All fields None
        assert obs.done is True
        assert 0.0 < obs.reward < 1.0
        assert len(obs.feedback) > 0


if __name__ == "__main__":
    tests = [
        test_task1_perfect,
        test_task1_partial,
        test_task1_wrong,
        test_task2_with_kb_coverage,
        test_task2_empty_response,
        test_task3_full_escalation,
        test_task3_partial_escalation,
        test_task3_no_escalation,
        test_score_clamping,
        test_reset_produces_clean_state,
        test_double_step_after_done,
        test_deterministic_scoring,
        test_task_cycling,
        test_empty_action_no_crash,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  FAIL {test.__name__}: EXCEPTION: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed")
