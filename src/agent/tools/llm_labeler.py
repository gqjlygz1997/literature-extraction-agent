"""LLM-based paper and evidence labelers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

CriterionAnswer = Literal["true", "false", "uncertain"]


@dataclass(frozen=True)
class CriterionResult:
    """Three-state result for one filter criterion."""

    answer: CriterionAnswer = "uncertain"
    reason: str = ""


@dataclass(frozen=True)
class PaperFilterDecision:
    """Structured result from the paper-level classifier.

    decision rules (pass_condition = all_required_not_false):
      - "reject" if any required criterion has answer == "false"
      - "pass"   if all required criteria have answer == "true" or "uncertain"
      - "pass"   on LLM parse failure  (reason: parse_failed_pass_by_default)
      - "pass"   on empty metadata     (reason: empty_metadata_pass_by_default)
    """

    paper_id: str
    decision: Literal["pass", "reject"]
    criteria: dict[str, CriterionResult] = field(default_factory=dict)
    reason: str = ""
    model: str = ""
    reasoning: str = ""  # Chain-of-thought rationale (DSPy only)


# ---------------------------------------------------------------------------
# Labeler
# ---------------------------------------------------------------------------

class LLMLabeler:
    """Wrapper around an LLM client for classification and labeling.

    llm_client should expose a .chat(...) method compatible with the
    OpenAI chat-completions API (e.g. langchain ChatOpenAI or the raw
    openai.OpenAI client).  The classify_paper implementation uses
    raw JSON mode for maximum compatibility across providers.
    """

    def __init__(self, llm_client=None, model_name: str = ""):
        self.llm_client = llm_client
        self.model_name = model_name

    # ---- paper-filter ------------------------------------------------------

    def classify_paper(self, article_meta, paper_filter_config) -> PaperFilterDecision:
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
            )

        prompt = self._build_classification_prompt(article_meta, paper_filter_config)

        try:
            raw = self._call_llm(prompt)
            criteria_results = self._parse_llm_response(raw, paper_filter_config)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM call or parse failed for %s: %s", paper_id, exc)
            return PaperFilterDecision(
                paper_id=paper_id,
                decision="pass",
                reason=f"parse_failed_pass_by_default: {exc}",
                model=self.model_name,
            )

        decision, reason = self._apply_pass_condition(criteria_results, paper_filter_config)
        return PaperFilterDecision(
            paper_id=paper_id,
            decision=decision,
            criteria=criteria_results,
            reason=reason,
            model=self.model_name,
        )

    def label_evidence(self, candidate_chunk, label_prompt: str):
        """Decide whether a chunk contains evidence for one target field (phase 2+)."""
        raise NotImplementedError("Evidence labeling will be implemented after paper filtering.")

    # ---- prompt building ---------------------------------------------------

    @staticmethod
    def _build_classification_prompt(article_meta, paper_filter_config) -> str:
        """Build the classification prompt sent to the LLM.

        The LLM is asked to evaluate each criterion independently and return
        structured JSON.  The prompt emphasises high recall: when uncertain,
        answer "uncertain" rather than "false".
        """
        criteria_lines = "\n".join(
            f'  - name: "{c.name}"\n    question: "{c.question}"'
            for c in paper_filter_config.criteria
            if c.required
        )

        quality_note = ""
        if article_meta.metadata_quality in ("html_front_matter", "title_only"):
            quality_note = (
                "\nNOTE: The text below may be incomplete (front-matter proxy or title only). "
                "When information is insufficient to answer confidently, "
                'answer "uncertain" instead of "false".'
            )

        text_block = ""
        if article_meta.title:
            text_block += f"Title: {article_meta.title}\n"
        if article_meta.text_for_filter:
            text_block += f"\nText:\n{article_meta.text_for_filter[:3000]}"

        schema_example = json.dumps(
            {
                "criteria": {
                    "<criterion_name>": {
                        "answer": "true | false | uncertain",
                        "reason": "<brief one-sentence reason>",
                    }
                }
            },
            indent=2,
        )

        return f"""You are a scientific literature classifier. Your task is to evaluate whether
a paper is relevant for a structured data extraction project.

IMPORTANT RULES:
- This is an early-stage coarse filter. Prioritise recall: do NOT reject papers
  when you are unsure.
- Answer "uncertain" (not "false") whenever the title/text does not provide
  enough information to make a confident negative judgement.
- Answer "false" only when you are highly confident the paper does NOT satisfy
  a criterion.{quality_note}

Evaluate the following criteria for the paper below:

{criteria_lines}

Paper text:
---
{text_block}
---

Return JSON matching this schema exactly (no extra keys):
{schema_example}
"""

    # ---- LLM call ----------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM client and return the raw text response."""
        if self.llm_client is None:
            raise RuntimeError("llm_client is not set on LLMLabeler")

        # Support both langchain ChatOpenAI (invoke) and raw openai client
        if hasattr(self.llm_client, "invoke"):
            # langchain interface
            from langchain_core.messages import HumanMessage
            response = self.llm_client.invoke([HumanMessage(content=prompt)])
            return response.content
        elif hasattr(self.llm_client, "chat"):
            # raw openai-compatible client
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        else:
            raise TypeError(f"Unsupported llm_client type: {type(self.llm_client)}")

    # ---- response parsing --------------------------------------------------

    @staticmethod
    def _parse_llm_response(
        raw: str, paper_filter_config
    ) -> dict[str, CriterionResult]:
        """Parse the LLM JSON response into CriterionResult objects.

        Any criterion not present in the response is treated as "uncertain".
        """
        valid_answers: set[CriterionAnswer] = {"true", "false", "uncertain"}

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # try to extract JSON substring
            match = __import__("re").search(r"\{.*\}", raw, __import__("re").S)
            if match:
                data = json.loads(match.group())
            else:
                raise ValueError(f"Cannot parse JSON from LLM response: {raw[:200]!r}")

        raw_criteria: dict = data.get("criteria", {})

        results: dict[str, CriterionResult] = {}
        for criterion in paper_filter_config.criteria:
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
                reason = f"unexpected LLM output format: {entry!r}"
            results[criterion.name] = CriterionResult(answer=answer, reason=reason)

        return results

    # ---- decision logic ----------------------------------------------------

    @staticmethod
    def _apply_pass_condition(
        criteria_results: dict[str, CriterionResult],
        paper_filter_config,
    ) -> tuple[Literal["pass", "reject"], str]:
        """Apply pass_condition=all_required_not_false.

        Returns (decision, reason_string).
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
