"""Data contracts for first-stage paper-filter configuration files."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class TargetField:
    """A final field the user wants in the clean output table."""

    name: str
    description: str = ""
    final_type: str = "string"


@dataclass(frozen=True)
class PaperFilterCriterion:
    """One boolean criterion used by the paper-level filter.

    required=True (default): if this criterion is explicitly "false", the paper
    is rejected regardless of other criteria.
    required=False: informational only; does not affect pass/reject decision.
    """

    name: str
    question: str
    expected_answer: bool = True
    rationale: str = ""
    required: bool = True
    uncertain_policy: Literal["pass", "reject"] = "pass"


@dataclass(frozen=True)
class PaperFilterConfig:
    """Configuration for early title-and-abstract filtering.

    pass_condition is fixed at "all_required_not_false":
      - pass  if every required criterion is "true" or "uncertain"
      - reject if any required criterion is explicitly "false"
    inclusive_when_uncertain=True means "uncertain" counts as pass.
    """

    input_scope: Literal["title_and_abstract"] = "title_and_abstract"
    criteria: list[PaperFilterCriterion] = field(default_factory=list)
    pass_condition: Literal["all_required_not_false"] = "all_required_not_false"
    inclusive_when_uncertain: bool = True


@dataclass(frozen=True)
class PaperFilterConfigFile:
    """Top-level object stored in a generated paper_filter.yaml file.

    This is intentionally narrower than a full domain extraction config.
    Stage 1 only decides whether a local paper should enter later parsing,
    retrieval, and extraction stages.
    """

    domain_name: str
    domain_description: str
    target_fields: list[TargetField]
    paper_filter: PaperFilterConfig


def validate_paper_filter_config(config: PaperFilterConfigFile) -> list[str]:
    """Return human-readable validation problems for a generated paper filter config."""

    problems: list[str] = []
    if not config.domain_name.strip():
        problems.append("domain_name is required")
    if not config.target_fields:
        problems.append("at least one target field is required")
    if not config.paper_filter.criteria:
        problems.append("paper_filter.criteria must not be empty")
    return problems
