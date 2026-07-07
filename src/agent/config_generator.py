"""Generate paper-filter configurations from user requirements."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import yaml

from .config_schema import (
    PaperFilterConfig,
    PaperFilterConfigFile,
    PaperFilterCriterion,
    TargetField,
    validate_paper_filter_config,
)
from .user_requirements import UserRequirements

logger = logging.getLogger(__name__)


class ConfigGenerator:
    """LLM-backed generator for first-stage paper-filter configs.

    Phase 1 target: generate paper_filter section only.
    Later phases: semantic queries, regex patterns, extraction schema.
    """

    def __init__(self, llm_client=None, model_name: str = ""):
        self.llm_client = llm_client
        self.model_name = model_name

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        requirements: UserRequirements,
        sample_title: str = "",
        sample_abstract: str = "",
    ) -> PaperFilterConfigFile:
        """Generate a paper_filter.yaml config from UserRequirements.

        Stage 1 only generates paper-level filtering criteria. Later stages
        will introduce separate retrieval/extraction configuration.
        """
        prompt = self.build_generation_prompt(
            domain_description=requirements.domain_description,
            target_fields=[f.name for f in requirements.target_fields],
            field_definitions={f.name: f.definition for f in requirements.target_fields},
            sample_title=sample_title,
            sample_abstract=sample_abstract,
        )

        raw = self._call_llm(prompt)
        criteria = self._parse_criteria_response(raw)

        target_field_objects = [
            TargetField(
                name=f.name,
                description=f.definition,
                final_type=f.type,
            )
            for f in requirements.target_fields
        ]

        paper_filter = PaperFilterConfig(
            input_scope="title_and_abstract",
            criteria=criteria,
            pass_condition="all_required_not_false",
            inclusive_when_uncertain=True,
        )

        config = PaperFilterConfigFile(
            domain_name=requirements.project_name,
            domain_description=requirements.domain_description,
            target_fields=target_field_objects,
            paper_filter=paper_filter,
        )

        problems = validate_paper_filter_config(config)
        if problems:
            raise ValueError(f"Generated config failed validation: {problems}")

        return config

    def build_generation_prompt(
        self,
        domain_description: str,
        target_fields: list[str],
        field_definitions: dict[str, str] | None = None,
        sample_title: str = "",
        sample_abstract: str = "",
    ) -> str:
        """Build the prompt used to ask an LLM to generate paper_filter criteria."""

        field_lines = "\n".join(
            f"  - {name}: {(field_definitions or {}).get(name, '')}"
            for name in target_fields
        )

        sample_block = ""
        if sample_title or sample_abstract:
            sample_block = "\nSample paper (title + abstract):\n"
            if sample_title:
                sample_block += f"Title: {sample_title}\n"
            if sample_abstract:
                sample_block += f"Abstract: {sample_abstract[:1000]}\n"

        schema_example = json.dumps(
            {
                "criteria": [
                    {
                        "name": "is_domain_topic",
                        "question": "Is this paper about <topic>?",
                        "rationale": "Exclude papers on unrelated topics.",
                        "required": True,
                    }
                ]
            },
            indent=2,
        )

        return f"""You are designing a paper-level filter for a scientific literature extraction system.

Domain:
{domain_description}

Target output fields (what we want to extract from passing papers):
{field_lines}

IMPORTANT RULES for generating criteria:
1. Generate 3-4 criteria maximum. Do NOT generate one criterion per field.
2. Group related fields: e.g., all survival endpoints into one "may report outcomes" criterion.
3. Use "might / likely / possibly" language — this is a coarse early filter, NOT precise extraction.
4. Criteria should NOT check for specific numeric values. Later stages handle that.
5. The filter must PRIORITISE RECALL. A missed relevant paper is worse than a false positive.
6. Use "uncertain" semantics in question wording: ask what the paper "might" contain, not what it "does" contain.
7. One criterion should check domain membership (is it the right research area?).
8. One criterion should check study type (exclude reviews, pure theory, animal-only if applicable).
9. One criterion should broadly check for target information ("might this paper report any of the target fields?").
10. Set required=true for every criterion.
{sample_block}
Return JSON matching this schema exactly:
{schema_example}
"""

    # ------------------------------------------------------------------ #
    #  Save / load                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def save(config: PaperFilterConfigFile, path: str | Path) -> None:
        """Write the generated paper-filter YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        field_defs = {f.name: f.description for f in config.target_fields if f.description}

        data = {
            "domain_name": config.domain_name,
            "domain_description": config.domain_description,
            "target_fields": [
                {
                    "name": f.name,
                    "description": f.description,
                    "final_type": f.final_type,
                }
                for f in config.target_fields
            ],
            "field_definitions": field_defs,
            "paper_filter": {
                "input_scope": config.paper_filter.input_scope,
                "inclusive_when_uncertain": config.paper_filter.inclusive_when_uncertain,
                "pass_condition": config.paper_filter.pass_condition,
                "criteria": [
                    {
                        "name": c.name,
                        "question": c.question,
                        "expected_answer": c.expected_answer,
                        "rationale": c.rationale,
                        "required": c.required,
                    }
                    for c in config.paper_filter.criteria
                ],
            },
        }

        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(data, fh, allow_unicode=True, sort_keys=False, default_flow_style=False)

        logger.info("Saved paper filter config to %s", path)

    @staticmethod
    def load(path: str | Path) -> PaperFilterConfigFile:
        """Load a previously generated paper_filter.yaml."""
        path = Path(path)
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        target_fields = [
            TargetField(
                name=f["name"],
                description=f.get("description", ""),
                final_type=f.get("final_type", "string"),
            )
            for f in data.get("target_fields", [])
        ]

        pf_raw = data.get("paper_filter", {})
        criteria = [
            PaperFilterCriterion(
                name=c["name"],
                question=c["question"],
                expected_answer=c.get("expected_answer", True),
                rationale=c.get("rationale", ""),
                required=c.get("required", True),
            )
            for c in pf_raw.get("criteria", [])
        ]

        paper_filter = PaperFilterConfig(
            input_scope=pf_raw.get("input_scope", "title_and_abstract"),
            criteria=criteria,
            pass_condition=pf_raw.get("pass_condition", "all_required_not_false"),
            inclusive_when_uncertain=pf_raw.get("inclusive_when_uncertain", True),
        )

        return PaperFilterConfigFile(
            domain_name=data["domain_name"],
            domain_description=data.get("domain_description", ""),
            target_fields=target_fields,
            paper_filter=paper_filter,
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _call_llm(self, prompt: str) -> str:
        if self.llm_client is None:
            raise RuntimeError("llm_client is not set on ConfigGenerator")

        if hasattr(self.llm_client, "invoke"):
            from langchain_core.messages import HumanMessage
            response = self.llm_client.invoke([HumanMessage(content=prompt)])
            return response.content
        elif hasattr(self.llm_client, "chat"):
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        else:
            raise TypeError(f"Unsupported llm_client type: {type(self.llm_client)}")

    @staticmethod
    def _parse_criteria_response(raw: str) -> list[PaperFilterCriterion]:
        """Parse the LLM JSON response into PaperFilterCriterion objects."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.S)
            if match:
                data = json.loads(match.group())
            else:
                raise ValueError(f"Cannot parse JSON from config LLM response: {raw[:300]!r}")

        raw_criteria = data.get("criteria", [])
        if not isinstance(raw_criteria, list):
            raise ValueError(f"Expected 'criteria' to be a list, got: {type(raw_criteria)}")

        result: list[PaperFilterCriterion] = []
        for item in raw_criteria:
            if not isinstance(item, dict):
                continue
            result.append(PaperFilterCriterion(
                name=str(item.get("name", "unnamed")),
                question=str(item.get("question", "")),
                expected_answer=bool(item.get("expected_answer", True)),
                rationale=str(item.get("rationale", "")),
                # Generated paper-filter criteria are gates, not optional notes.
                # Reuse configs can still encode optional criteria via load().
                required=True,
            ))

        if not result:
            raise ValueError("LLM returned no criteria in config generation response")

        return result
