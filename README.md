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
EXTRACTOR_MAX_TOKENS=4000
```

`NCBI_EMAIL` is optional but recommended for DOI/PMID to PMCID lookup. Never
commit the real `.env` file.

For extraction, `EXTRACTOR_TIMEOUT` defaults to 90 seconds and
`EXTRACTOR_MAX_RETRIES` defaults to 0. A timed-out paper is recorded as failed
and the batch continues; completed papers are checkpointed to
`extracted_records.jsonl` after each paper.

Batch runs are resumable by default:

- Reuse one output directory, for example `my_project/outputs`.
- `--limit 10` means "process at most 10 unfinished papers in this stage".
- Successfully completed papers are skipped on later runs.
- Failed extraction papers are retried on later runs.
- `--force` recomputes a stage instead of skipping existing results.

You can change `--limit` freely. If the first 10 papers are already complete,
the next `--limit 10` run continues with the next unfinished papers.

## Fast Start: Local XML Batch

Use this when you already have JATS/PMC XML files. Keep every stage in the same
output directory so later batches extend the previous results instead of
creating isolated folders.

```bash
.venv/bin/python run_pipeline.py \
  --requirements my_project/user_requirements.yaml \
  --xml my_project/input_papers \
  --output my_project/outputs \
  --limit 10
```

By default, `run_pipeline.py` reads the labeling domain from `project_name` in
`user_requirements.yaml`. Use `--domain ...` only when you need to override it.

## Fast Start: WOS Index Batch

Use this when you start from a Web of Science `savedrecs.txt` export or a local
folder of WOS `.txt` exports. WOS ingestion reads the index, then `--limit 10`
controls how many unfinished papers move through the expensive stages.

```bash
.venv/bin/python run_pipeline.py \
  --requirements my_project/user_requirements.yaml \
  --wos my_project/savedrecs.txt \
  --output my_project/outputs \
  --limit 10
```

The repository also includes the pancan WOS index files used for the example
run:

```bash
.venv/bin/python run_pipeline.py \
  --requirements examples/pancan_treatment_outcomes/user_requirements.yaml \
  --wos examples/pancan_treatment_outcomes/wos_savedrecs \
  --output outputs/pancan \
  --limit 10
```

`records.csv` is rewritten by the final post-processing stage from all current
postprocessed records. It is not a per-batch CSV and it is not blindly appended
line by line.

## Quick Test: WOS Metadata Route

Use this route when the input is a Web of Science `savedrecs.txt` file.

```bash
python run_wos_ingest.py \
  --input my_project/savedrecs.txt \
  --output my_project/outputs

python run_paper_filter.py \
  --requirements examples/pancan_treatment_outcomes/user_requirements.yaml \
  --metadata my_project/outputs/candidate_papers.jsonl \
  --output my_project/outputs \
  --limit 10

python run_fulltext_acquisition.py \
  --passed my_project/outputs/passed_papers.jsonl \
  --output my_project/outputs \
  --limit 10
```

At this point the system has filtered papers and downloaded available PMC XML.
Not every DOI/PMID has open JATS XML; such papers are recorded as
`no_pmcid` or `xml_unavailable`.

Continue the full pipeline:

```bash
python run_preprocess.py \
  --passed my_project/outputs/downloaded_papers.jsonl \
  --output my_project/outputs \
  --limit 10

python run_labeling.py \
  --requirements examples/pancan_treatment_outcomes/user_requirements.yaml \
  --chunks my_project/outputs/parsed_chunks.jsonl \
  --output my_project/outputs \
  --domain pancan_treatment_outcomes \
  --limit 10

python run_extraction.py \
  --requirements examples/pancan_treatment_outcomes/user_requirements.yaml \
  --chunks my_project/outputs/parsed_chunks.jsonl \
  --labels my_project/outputs/labeled_chunks.jsonl \
  --output my_project/outputs \
  --limit 10

python run_postprocess.py \
  --requirements examples/pancan_treatment_outcomes/user_requirements.yaml \
  --records my_project/outputs/extracted_records.jsonl \
  --output my_project/outputs
```

## Quick Test: Local XML Route

Use this route when the input is already a folder of JATS/XML files.

```bash
python run_paper_filter.py \
  --requirements examples/pancan_treatment_outcomes/user_requirements.yaml \
  --input my_project/input_papers \
  --output my_project/outputs \
  --limit 10

python run_preprocess.py \
  --passed my_project/outputs/passed_papers.jsonl \
  --output my_project/outputs \
  --limit 10
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
python test_resume_basic.py
python test_labeling_basic.py
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
