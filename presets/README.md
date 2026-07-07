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
