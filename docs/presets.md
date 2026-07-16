# Preset Configs

The project now supports ALLMAT-style preset configs/prompts.

For a project:

```yaml
project_name: pancan_treatment_outcomes
```

place presets here:

```text
presets/pancan_treatment_outcomes/
```

Supported presets:

```text
paper_filter.yaml
labeling_config.yaml
extraction_prompt.yaml
postprocess_config.yaml
```

For labeling presets, `labeling_strategy.llm_binary_confirm: false` keeps
labeling fast by using retrieval results directly. Change it to `true` if a
domain needs stricter LLM confirmation.

`paper_filter.yaml` can set `uncertain_policy: pass` or `uncertain_policy:
reject` per required criterion. The generic default is `pass`; use `reject`
only when a domain has a critical condition that must be explicit in the title
or abstract.

`extraction_prompt.yaml` can optionally define `endpoint_constraints`. In
`strict` mode it maps each `record_type` to allowed canonical endpoints and
aliases. The pipeline filters incompatible records locally after extraction;
without this config, extraction remains unrestricted.

The pipeline uses presets before generating anything with DSPy/LLM:

```text
explicit --config / --prompt-preset
↓
project preset
↓
existing output config
↓
DSPy/LLM generation
```

This means a mature domain can run mostly from hand-written stable prompts, while
new domains still fall back to DSPy-generated configs.

Because stage outputs are resumable, changing a preset does not automatically
rewrite papers that were already completed. After editing `paper_filter.yaml`,
`labeling_config.yaml`, `extraction_prompt.yaml`, or `postprocess_config.yaml`,
rerun the affected stage with `--force` if existing papers should be recomputed.

`postprocess_config.yaml` is used after extraction. It can define numeric field
units, standard terms/synonyms, and validity filters. This mirrors ALLMAT's
rule-based normalization/entity-resolution layer, but keeps the rules in a
domain preset instead of hard-coding HEA-only logic.

The pancan preset is a complete reference configuration: it uses five
evidence-locator fields during labeling, all record fields during extraction,
strict record-type/endpoint compatibility, and a post-processing rule requiring
record type, intervention, model/population, and endpoint.
