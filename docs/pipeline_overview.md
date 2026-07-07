# Pipeline Overview

This document summarizes the MVP workflow and the main artifact produced by
each stage.

## Inputs

The user provides:

```text
user_requirements.yaml
local JATS/XML paper folder
optional presets/<project_name>/
```

`user_requirements.yaml` defines the project name, target record fields, and
domain requirements. Presets are optional hand-written configs/prompts used
before any DSPy/LLM-generated configuration.

## 1. Paper Filter

The paper filter reads title, abstract, and front matter, then decides whether
each paper should enter the full pipeline.

Execution policy:

```text
use paper_filter.yaml preset if available
otherwise generate paper_filter.yaml from user_requirements.yaml
use DSPy/LLM to classify pass/reject
write pass/reject decisions and reasons
```

Main outputs:

```text
paper_filter_results.jsonl
passed_papers.jsonl
rejected_papers.jsonl
run_summary.json
```

This follows the ALLMAT idea of title/abstract-level paper classification, but
makes the filter configurable for non-HEA domains.

## 2. Article Processing

The parser fully parses JATS/XML papers, preserves section hierarchy, and
converts article content into unified chunks.

Processing steps:

```text
parse full JATS/XML
preserve section_path
clean paragraphs
split long paragraphs into paragraph chunks
parse tables into table chunks
store paragraph, abstract, and table chunks in one schema
```

Main outputs:

```text
parsed_chunks.jsonl
preprocessing_summary.json
```

Table chunks preserve caption, headers, raw rows, and a text representation for
retrieval and LLM labeling.

## 3. Labeling

Labeling finds chunks relevant to each target field.

Execution policy:

```text
use labeling_config.yaml preset if available
otherwise generate semantic query, regex, and section rules with DSPy/LLM
embed parsed chunks into a Chroma vector store
apply section exclude/include rules
retrieve Text/Table candidates with semantic + regex signals
rank candidates with RRF
use DSPy/LLM to confirm relevant top-k chunks
merge labels by chunk_id
```

Main output:

```text
labeled_chunks.jsonl
```

Each output row is chunk-centric:

```json
{
  "paper_id": "PMC10389558",
  "chunk_id": "PMC10389558::p0015",
  "chunk_index": 15,
  "chunk_type": "paragraph",
  "section_path": ["Results"],
  "labels": ["treatment_regimen", "os", "p_value"]
}
```

## 4. Extraction

Extraction uses contextualized extraction only in the MVP. It collects relevant
labeled chunks for each paper, preserves original article order, and extracts
full records with a JSON schema.

Execution policy:

```text
build local evidence context in article order
generate JSON schema from user_requirements.yaml
use extraction_prompt.yaml preset if available
otherwise generate prompt dynamically
call LLM for contextualized structured extraction
deduplicate records, fill missing fields, and preserve source chunk ids
```

Main output:

```text
extracted_records.jsonl
```

## 5. Post-Processing

Post-processing makes extracted JSONL easier to analyze.

Execution policy:

```text
use postprocess_config.yaml preset if available
otherwise use generic defaults
normalize numeric fields
standardize domain terms with preset dictionaries
filter invalid records
deduplicate and mark conflicts where applicable
export JSONL and CSV
```

Main outputs:

```text
postprocessed_records.jsonl
records.csv
postprocessing_summary.json
```

## Current Scope

Implemented:

```text
paper filtering
JATS/XML preprocessing
labeling
contextualized extraction
lightweight post-processing
domain presets
```

Deferred:

```text
ALLMAT-style DetectProcesses/template injection
advanced entity resolution
downstream ML dataset construction
large-scale evaluation
```
