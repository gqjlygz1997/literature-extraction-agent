# Post-processing Stage MVP

Stage 3 turns `extracted_records.jsonl` into records that are easier to compare,
analyze, and export.

This stage follows ALLMAT's post-extraction engineering style, but keeps the
first version conservative:

- keep generic code for empty values, numeric parsing, strict deduplication, and
  CSV export
- keep domain knowledge in `postprocess_config.yaml`
- do not run LLMs in the MVP
- do not do fuzzy merge or cross-paper entity resolution yet

## Quick Start

```bash
python run_postprocess.py \
  --requirements experiments/pancan/user_requirements.yaml \
  --records experiments/pancan/outputs/extracted_records.jsonl \
  --output experiments/pancan/outputs
```

Output:

```text
experiments/pancan/outputs/
├── postprocessed_records.jsonl
├── records.csv
├── postprocess_config.yaml
└── postprocessing_summary.json
```

Post-processing is deterministic and cheap. Rerun it after extraction changes;
it rewrites the final JSONL/CSV from the current `extracted_records.jsonl`.

## Data Flow

```text
extracted_records.jsonl + user_requirements.yaml
        ↓
load postprocess_config.yaml from preset if present
        ↓
empty/null value cleanup
        ↓
numeric parsing for number fields
        ↓
domain standardization by preset synonyms
        ↓
validity filtering
        ↓
strict deduplication
        ↓
postprocessed_records.jsonl + records.csv + summary
```

## Generic Line

The generic line works without any domain preset:

- fields with `type: number` are parsed with the numeric parser
- empty values such as `NA`, `N/A`, `not reported`, and `-` become `null`
- strict duplicate records are merged only when all post-processed field values
  match
- `records.csv` is exported with numeric helper columns

Example:

```json
{
  "os": "10-20 months",
  "os_norm": {
    "raw": "10-20 months",
    "operator": "range",
    "value": 15.0,
    "value_min": 10.0,
    "value_max": 20.0,
    "unit": "month"
  }
}
```

## Preset Line

The preset line carries domain-specific rules:

```yaml
numeric_fields:
  os:
    unit: month
  hr:
    unit: ratio
    detect_unit: false
    parse_ranges: false
  ci:
    unit: ratio
    detect_unit: false
    parse_ranges: true

standardize:
  phase:
    multiple: true
    terms:
      FCC: [FCC, f.c.c., face-centered cubic]

validity:
  required_any:
    - os
    - pfs
```

Supported preset file:

```text
presets/<project_name>/postprocess_config.yaml
```

The CLI priority is:

```text
explicit --config
↓
presets/<project_name>/postprocess_config.yaml
↓
generic defaults
```

## ALLMAT Alignment

ALLMAT uses domain-specific normalizers such as composition normalization,
processing keyword normalization, strict partitioning, fuzzy partitioning, and
merge. This MVP mirrors the stable parts:

| Our Stage 3 | ALLMAT idea |
|---|---|
| numeric parser | prompt-enforced numeric strings + later normalization |
| standardize preset | `normalize_composition`, `normalize_label`, `normalize_processing_kw` |
| strict deduplication | `partition_strict` / rule entity resolution |
| CSV export | final analysis-ready table |

Deferred:

- fuzzy merge
- LLM entity resolution
- field-specific advanced parsers
- task-specific ML table builders

## Field-Level Numeric Controls

Some clinical strings contain multiple numbers, so the preset can control how a
field is parsed:

- `detect_unit: false` prevents unrelated text such as `95% CI` from making the
  CI unit become `percent`
- `parse_ranges: false` makes HR parse the first value in
  `HR 0.82 (95% CI 0.71-0.94)` instead of the CI range

This is the same general idea as ALLMAT's domain-specific normalizers: keep a
generic parser, but let the domain preset decide field-specific behavior.
