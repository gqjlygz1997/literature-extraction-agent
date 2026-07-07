"""DSPy-based paper filter using ChainOfThought for structured classification."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config_schema import PaperFilterConfig

from .llm_labeler import CriterionResult, PaperFilterDecision

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DSPy Signature
# ---------------------------------------------------------------------------

try:
    import dspy
except ImportError:
    dspy = None
    logger.warning("dspy is not installed. DSPyPaperFilter will not be available.")


if dspy is not None:
    class PaperFilterSignature(dspy.Signature):
        """
        You are a scientific literature classifier. Evaluate whether a paper should
        pass an early-stage relevance filter for a structured data extraction project.

        RULES:
        - Prioritise recall. Answer "uncertain" (not "false") when you are not sure.
        - Answer "false" ONLY when highly confident the criterion is NOT met.
        - "uncertain" is treated as pass (inclusive by default).
        - Evaluate each criterion independently.
        """

        paper_text: str = dspy.InputField(
            desc="Title and abstract (or front matter) of the paper."
        )
        criteria_description: str = dspy.InputField(
            desc='JSON array of criteria: [{"name": "...", "question": "...", "required": true/false}, ...]'
        )
        evaluation: str = dspy.OutputField(
            desc='Respond in JSON: {"criteria": {"<criterion_name>": {"answer": "true|false|uncertain", "reason": "<one sentence>"}}}'
        )


# ---------------------------------------------------------------------------
# DSPy Paper Filter
# ---------------------------------------------------------------------------

class DSPyPaperFilter:
    """DSPy-based paper classifier using ChainOfThought.

    Provides the same interface as LLMLabeler.classify_paper() for drop-in replacement.
    DSPy LM must be configured globally via dspy.configure() before instantiation.
    """

    def __init__(self, model_name: str = ""):
        if dspy is None:
            raise ImportError(
                "dspy is not installed. Run: pip install 'dspy>=2.5.22'"
            )
        self.predictor = dspy.ChainOfThought(PaperFilterSignature)
        self.model_name = model_name

    def classify_paper(
        self, article_meta, paper_filter_config: PaperFilterConfig
    ) -> PaperFilterDecision:
        """Use title and text_for_filter to decide pass / reject.

        Each required criterion is evaluated independently. The answer
        is one of "true" / "false" / "uncertain":
          - "false"     → paper is rejected (high confidence it does not match)
          - "uncertain" → treated as pass (inclusive_when_uncertain=True)
          - "true"      → passes this criterion

        If title and text_for_filter are both empty, the paper passes by
        default (empty_metadata_pass_by_default).

        If the LLM response cannot be parsed, the paper passes by default
        (parse_failed_pass_by_default).
        """
        paper_id = article_meta.paper_id

        # --- guard: empty metadata ---
        if not article_meta.title and not article_meta.text_for_filter:
            return PaperFilterDecision(
                paper_id=paper_id,
                decision="pass",
                reason="empty_metadata_pass_by_default",
                model=self.model_name,
                reasoning="",
            )

        # --- build inputs ---
        paper_text = self._build_paper_text(article_meta)
        criteria_description = self._build_criteria_description(
            paper_filter_config.criteria
        )

        # --- call DSPy ChainOfThought ---
        try:
            prediction = self.predictor(
                paper_text=paper_text,
                criteria_description=criteria_description,
            )
            # Extract rationale (ChainOfThought automatically adds this)
            reasoning = getattr(prediction, "rationale", "")
            evaluation_raw = prediction.evaluation
            criteria_results = self._parse_evaluation(
                evaluation_raw, paper_filter_config.criteria
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("DSPy call or parse failed for %s: %s", paper_id, exc)
            return PaperFilterDecision(
                paper_id=paper_id,
                decision="pass",
                reason=f"parse_failed_pass_by_default: {exc}",
                model=self.model_name,
                reasoning="",
            )

        # --- apply pass condition ---
        decision, reason = self._apply_pass_condition(
            criteria_results, paper_filter_config
        )

        return PaperFilterDecision(
            paper_id=paper_id,
            decision=decision,
            criteria=criteria_results,
            reason=reason,
            model=self.model_name,
            reasoning=reasoning,
        )

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _build_paper_text(article_meta) -> str:
        """Build paper_text from title and text_for_filter."""
        parts = []
        if article_meta.title:
            parts.append(f"Title: {article_meta.title}")
        if article_meta.text_for_filter:
            # Truncate to 3000 chars (same as LLMLabeler)
            parts.append(f"\nText:\n{article_meta.text_for_filter[:3000]}")

        quality_note = ""
        if article_meta.metadata_quality in ("html_front_matter", "title_only"):
            quality_note = (
                "\n\nNOTE: The text above may be incomplete (front-matter proxy or title only). "
                "When information is insufficient to answer confidently, "
                'answer "uncertain" instead of "false".'
            )

        return "\n".join(parts) + quality_note

    @staticmethod
    def _build_criteria_description(criteria) -> str:
        """Serialize criteria list to JSON string."""
        required_criteria = [c for c in criteria if c.required]
        criteria_list = [
            {
                "name": c.name,
                "question": c.question,
                "required": c.required,
            }
            for c in required_criteria
        ]
        return json.dumps(criteria_list, ensure_ascii=False, indent=2)

    @staticmethod
    def _parse_evaluation(
        raw: str, criteria
    ) -> dict[str, CriterionResult]:
        """Parse the DSPy evaluation JSON response into CriterionResult objects.

        Any criterion not present in the response is treated as "uncertain".
        Follows the same logic as LLMLabeler._parse_llm_response.
        """
        valid_answers = {"true", "false", "uncertain"}

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # try to extract JSON substring (same fallback as LLMLabeler)
            import re
            match = re.search(r"\{.*\}", raw, re.S)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    raise ValueError(f"Cannot parse JSON from DSPy response: {raw[:200]!r}")
            else:
                raise ValueError(f"Cannot parse JSON from DSPy response: {raw[:200]!r}")

        raw_criteria: dict = data.get("criteria", {})

        results: dict[str, CriterionResult] = {}
        for criterion in criteria:
            if not criterion.required:
                continue
            entry = raw_criteria.get(criterion.name, {})
            if isinstance(entry, dict):
                answer = str(entry.get("answer", "uncertain")).lower()
                if answer not in valid_answers:
                    answer = "uncertain"
                reason = str(entry.get("reason", ""))
            else:
                # unexpected format → treat as uncertain
                answer = "uncertain"
                reason = f"unexpected DSPy output format: {entry!r}"
            results[criterion.name] = CriterionResult(answer=answer, reason=reason)

        return results

    @staticmethod
    def _apply_pass_condition(
        criteria_results: dict[str, CriterionResult],
        paper_filter_config,
    ) -> tuple[str, str]:
        """Apply pass_condition=all_required_not_false.

        Returns (decision, reason_string).
        Reuses logic from LLMLabeler.
        """
        rejected_by: list[str] = []
        for name, result in criteria_results.items():
            if result.answer == "false":
                rejected_by.append(f"{name}: {result.reason}")

        if rejected_by:
            return "reject", "Rejected because: " + "; ".join(rejected_by)

        uncertain = [n for n, r in criteria_results.items() if r.answer == "uncertain"]
        if uncertain:
            return (
                "pass",
                f"No required criterion is explicitly false. "
                f"Uncertain criteria treated as pass: {', '.join(uncertain)}.",
            )

        return "pass", "All required criteria satisfied."
