# Presets

Preset files are hand-written, stable configs/prompts used before any DSPy/LLM
generation. Put them under:

```text
presets/<project_name>/
```

`<project_name>` must match `project_name` in `user_requirements.yaml`.

Supported files:

```text
paper_filter.yaml        # skips paper-filter config generation
labeling_config.yaml     # skips DSPy labeling config generation
extraction_prompt.yaml   # skips dynamic extraction prompt generation
postprocess_config.yaml  # controls numeric parsing, standardization, filtering
```

In `labeling_config.yaml`, `labeling_strategy.llm_binary_confirm` defaults to
`false` for faster retrieval-only labeling. Set it to `true` only when you want
an extra LLM binary relevance check over retrieved top-k chunks.

For pancreatic cancer, labeling uses only core evidence-locator fields; final
extraction still follows all fields in `user_requirements.yaml`.

`paper_filter.yaml` supports `uncertain_policy` on each required criterion:
`pass` keeps uncertain papers for recall, while `reject` requires explicit
evidence for a critical domain condition.

`extraction_prompt.yaml` can define `endpoint_constraints` for a domain. In
strict mode, it permits only configured endpoint/record-type pairs and
canonicalizes configured aliases. Without this block, extraction is
unrestricted.

The pancan preset uses these rules to keep intervention-linked drug-development
evidence and reject clearly biomarker-only or prognostic papers. Its
post-processing config requires `record_type`, `compound_or_treatment`,
`model_or_population`, and `endpoint` for a final record.

Priority:

```text
explicit --config / --prompt-preset
↓
presets/<project_name>/...
↓
existing output config
↓
DSPy/LLM generation
```

This lets production runs behave like ALLMAT: stable hand-written rules by
default, with DSPy/LLM generation as fallback for new domains.

For Stage 3, presets are especially important: generic code handles empty
values, numeric parsing, strict deduplication, and CSV export, while each domain
defines its own synonym tables, units, and validity rules.
