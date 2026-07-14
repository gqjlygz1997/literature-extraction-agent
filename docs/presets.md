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

`postprocess_config.yaml` is used after extraction. It can define numeric field
units, standard terms/synonyms, and validity filters. This mirrors ALLMAT's
rule-based normalization/entity-resolution layer, but keeps the rules in a
domain preset instead of hard-coding HEA-only logic.
