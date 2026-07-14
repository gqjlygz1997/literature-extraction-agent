# Literature Extraction Agent

A configurable agent pipeline for extracting structured records from scientific
literature. The project is inspired by ALLMAT/Sisyphus, but the domain logic is
moved into `user_requirements.yaml` and optional preset files so that the same
engineering pipeline can be reused across biomedical and materials-science
tasks.

The current MVP includes a pancreatic-cancer preset for drug-development
evidence: in-vitro efficacy, in-vivo efficacy, pharmacokinetics, animal
toxicity, and clinical outcomes.

## What It Does

The system can start from either Web of Science metadata or local JATS/XML
papers:

```text
user_requirements.yaml + WOS savedrecs.txt / local JATS XML
↓
WOS metadata ingestion: title / abstract / DOI / PMID
↓
Paper filter: pass / reject from title and abstract
↓
Full-text acquisition: DOI/PMID → PMCID → PMC XML
↓
Article processing: section-aware paragraph and table chunks
↓
Labeling: semantic + regex retrieval to find evidence chunks
↓
Extraction: contextualized JSON-schema record extraction
↓
Post-processing: normalize, clean, deduplicate, export CSV
```

For local XML papers, the WOS ingestion and full-text acquisition stages can be
skipped.

## Inputs

```text
1. user_requirements.yaml
   Defines the project name, record meaning, and output fields.

2. Paper source
   - Web of Science savedrecs.txt, or
   - a local folder of JATS/XML papers

3. Optional presets/<project_name>/
   - paper_filter.yaml
   - labeling_config.yaml
   - extraction_prompt.yaml
   - postprocess_config.yaml
```

If a preset exists, it is used directly. If not, the system falls back to
DSPy/LLM generation or generic defaults. This keeps stable domains close to
ALLMAT-style engineered prompts while still allowing new domains to be
bootstrapped.

The pancan preset is intentionally stricter than the generic fallback: it
rejects papers that are clearly biomarker-only, diagnostic, prognostic, or
risk-factor studies without intervention-linked evidence, while retaining
genuinely ambiguous treatment papers for recall.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local `.env` file:

```bash
cp .env.example .env
```

Fill in at least:

```text
LLM_API_KEY       OpenAI-compatible chat API key
LLM_BASE_URL      OpenAI-compatible API base URL
LLM_MODEL         chat model name
GEMINI_API_KEY    Google AI Studio key for Gemini embeddings
EMBEDDING_MODEL   gemini-embedding-001
```

`NCBI_EMAIL` is optional but recommended for DOI/PMID to PMCID lookup. Never
commit the real `.env` file.

## Quick Test: WOS Metadata Route

Use this route when the input is a Web of Science `savedrecs.txt` file.

```bash
python run_wos_ingest.py \
  --input my_project/savedrecs.txt \
  --output my_project/output/wos_ingest

python run_paper_filter.py \
  --requirements examples/pancan_treatment_outcomes/user_requirements.yaml \
  --metadata my_project/output/wos_ingest/candidate_papers.jsonl \
  --output my_project/output/paper_filter

python run_fulltext_acquisition.py \
  --passed my_project/output/paper_filter/passed_papers.jsonl \
  --output my_project/output/fulltext
```

At this point the system has filtered papers and downloaded available PMC XML.
Not every DOI/PMID has open JATS XML; such papers are recorded as
`no_pmcid` or `xml_unavailable`.

Continue the full pipeline:

```bash
python run_preprocess.py \
  --passed my_project/output/fulltext/downloaded_papers.jsonl \
  --output my_project/output/preprocess

python run_labeling.py \
  --requirements examples/pancan_treatment_outcomes/user_requirements.yaml \
  --chunks my_project/output/preprocess/parsed_chunks.jsonl \
  --output my_project/output/labeling \
  --domain pancan_treatment_outcomes

python run_extraction.py \
  --requirements examples/pancan_treatment_outcomes/user_requirements.yaml \
  --chunks my_project/output/preprocess/parsed_chunks.jsonl \
  --labels my_project/output/labeling/labeled_chunks.jsonl \
  --output my_project/output/extraction

python run_postprocess.py \
  --requirements examples/pancan_treatment_outcomes/user_requirements.yaml \
  --records my_project/output/extraction/extracted_records.jsonl \
  --output my_project/output/postprocess
```

## Quick Test: Local XML Route

Use this route when the input is already a folder of JATS/XML files.

```bash
python run_paper_filter.py \
  --requirements examples/pancan_treatment_outcomes/user_requirements.yaml \
  --input my_project/input_papers \
  --output my_project/output/paper_filter

python run_preprocess.py \
  --passed my_project/output/paper_filter/passed_papers.jsonl \
  --output my_project/output/preprocess
```

Then run labeling, extraction, and post-processing using the same commands shown
above.

## Main Outputs

```text
candidate_papers.jsonl                 WOS metadata converted to paper rows
passed_papers.jsonl / rejected_papers.jsonl
downloaded_papers.jsonl                rows with local PMC XML paths
parsed_chunks.jsonl                    paragraph / table / abstract chunks
labeled_chunks.jsonl                   chunk-level evidence labels
extracted_records.jsonl                raw structured records
postprocessed_records.jsonl
records.csv                            analysis-ready table
*_summary.json                         stage summaries and errors
```

For the pancan preset, `extraction_summary.json` also reports records rejected
by record-type/endpoint constraints. This makes it possible to audit whether a
domain endpoint list is too narrow without storing verbose retrieval traces.

## Verification

The repository includes focused no-LLM tests for WOS ingestion, paper-filter
policy, extraction helpers, and post-processing:

```bash
python test_wos_metadata_basic.py
python test_paper_filter_basic.py
python test_extraction_basic.py
python test_postprocess_basic.py
```

## Repository Map

```text
examples/       Example user_requirements.yaml files
presets/        Hand-written domain configs and prompts
src/agent/      Pipeline implementation
run_*.py        Stage-level CLI entry points
docs/           More detailed design and usage notes
```

Useful docs:

```text
docs/pipeline_overview.md
docs/quickstart.md
docs/presets.md
```

Generated outputs, local papers, vector stores, caches, and credentials are
ignored by git.
