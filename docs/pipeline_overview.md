# Pipeline Overview

This document summarizes the MVP workflow and the main artifact produced by
each stage.

## Inputs

The user provides:

```text
user_requirements.yaml
WOS savedrecs.txt and/or local JATS/XML paper folder
optional presets/<project_name>/
```

`user_requirements.yaml` defines the project name, target record fields, and
domain requirements. Presets are optional hand-written configs/prompts used
before any DSPy/LLM-generated configuration.

## 0. WOS Metadata Ingestion

When the input starts from a Web of Science tagged-text export, the system
parses bibliographic metadata before attempting any full-text download.

Processing steps:

```text
parse savedrecs.txt
extract title, abstract, DOI, PMID, and WOS UID
write candidate metadata rows for paper filtering
```

Main outputs:

```text
candidate_papers.jsonl
wos_ingestion_summary.json
```

## 1. Paper Filter

The paper filter reads title and abstract from WOS metadata, or title,
abstract, and front matter from local XML/HTML files. It decides whether each
paper should enter full-text acquisition and parsing.

Execution policy:

```text
use paper_filter.yaml preset if available
otherwise generate paper_filter.yaml from user_requirements.yaml
use DSPy/LLM to classify pass/reject from title and abstract
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

## 2. Full-Text Acquisition

When paper filtering starts from WOS metadata, only passed papers are resolved
to full text. This avoids wasting downloads on clearly irrelevant papers.

Processing steps:

```text
use pass-paper DOI/PMID to query PMCID
download available PMC JATS/XML
write XML-backed rows for preprocessing
```

Main outputs:

```text
fulltext_acquisition_results.jsonl
downloaded_papers.jsonl
fulltext_acquisition_summary.json
pmc_xml/
```

If the user already has local JATS/XML files, this stage can be skipped.

## 3. Article Processing

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

## 4. Labeling

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

## 5. Extraction

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

## 6. Post-Processing

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
WOS metadata ingestion
PMC XML acquisition
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
