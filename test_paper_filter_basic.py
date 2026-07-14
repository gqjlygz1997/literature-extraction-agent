"""Unit tests for configurable paper-filter strictness (no LLM calls)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _config(uncertain_policy: str):
    from src.agent.config_schema import PaperFilterConfig, PaperFilterCriterion

    return PaperFilterConfig(criteria=[
        PaperFilterCriterion(
            name="has_eligible_intervention_evidence",
            question="Is eligible intervention evidence explicitly reported?",
            uncertain_policy=uncertain_policy,
        )
    ])


def _results(answer: str):
    from src.agent.tools.llm_labeler import CriterionResult

    return {
        "has_eligible_intervention_evidence": CriterionResult(
            answer=answer,
            reason="test",
        )
    }


def test_uncertain_policy_pass_preserves_legacy_behavior():
    from src.agent.tools.llm_labeler import LLMLabeler

    decision, _ = LLMLabeler._apply_pass_condition(_results("uncertain"), _config("pass"))
    assert decision == "pass"


def test_uncertain_policy_rejects_strict_criterion():
    from src.agent.tools.llm_labeler import LLMLabeler

    decision, reason = LLMLabeler._apply_pass_condition(_results("uncertain"), _config("reject"))
    assert decision == "reject"
    assert "strict criterion" in reason


def test_false_always_rejects():
    from src.agent.tools.llm_labeler import LLMLabeler

    decision, _ = LLMLabeler._apply_pass_condition(_results("false"), _config("pass"))
    assert decision == "reject"


def test_dspy_filter_uses_same_strict_rule():
    from src.agent.tools.dspy_paper_filter import DSPyPaperFilter

    decision, _ = DSPyPaperFilter._apply_pass_condition(_results("uncertain"), _config("reject"))
    assert decision == "reject"


if __name__ == "__main__":
    tests = [
        test_uncertain_policy_pass_preserves_legacy_behavior,
        test_uncertain_policy_rejects_strict_criterion,
        test_false_always_rejects,
        test_dspy_filter_uses_same_strict_rule,
    ]
    for test in tests:
        test()
        print(f"  ✓ {test.__name__}")
    print(f"✅ All {len(tests)} paper-filter tests passed!")
