# Literature Extraction Agent

A configurable pipeline for extracting structured records from scientific
literature. Domain-specific logic is kept in `user_requirements.yaml` and
optional presets, so the same pipeline can be reused for different research
areas.

The current example preset targets pancreatic-cancer drug-development evidence:
in-vitro efficacy, in-vivo efficacy, pharmacokinetics, animal toxicity, and
clinical outcomes.

## Quick Start

Install dependencies and create a private `.env` file:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in at least:

```text
LLM_API_KEY
LLM_BASE_URL
LLM_MODEL
GEMINI_API_KEY
EMBEDDING_MODEL=gemini-embedding-001
EXTRACTOR_MAX_TOKENS=4000
```

Run the bundled pancreatic-cancer example from Web of Science index files:

```bash
.venv/bin/python run_pipeline.py \
  --requirements examples/pancan_treatment_outcomes/user_requirements.yaml \
  --wos examples/pancan_treatment_outcomes/wos_savedrecs \
  --output experiments/pancan/outputs \
  --limit 10
```

Run the same command again to continue with the next unfinished papers.
`--limit 10` means "process up to 10 unfinished papers"; completed papers are
skipped automatically.

## Run Your Own Input

From a Web of Science `savedrecs.txt` export:

```bash
.venv/bin/python run_pipeline.py \
  --requirements my_project/user_requirements.yaml \
  --wos my_project/savedrecs.txt \
  --output my_project/outputs \
  --limit 10
```

From local PMC/JATS XML files:

```bash
.venv/bin/python run_pipeline.py \
  --requirements my_project/user_requirements.yaml \
  --xml my_project/input_papers \
  --output my_project/outputs \
  --limit 10
```

From a preselected metadata JSONL file:

```bash
.venv/bin/python run_pipeline.py \
  --requirements my_project/user_requirements.yaml \
  --metadata my_project/candidate_papers.jsonl \
  --output my_project/outputs \
  --limit 10
```

## Re-run One Paper

Use `--paper-id` to re-run extraction for selected papers without touching the
rest of the output:

```bash
EXTRACTOR_MAX_TOKENS=4000 EXTRACTOR_TIMEOUT=180 .venv/bin/python run_extraction.py \
  --requirements examples/pancan_treatment_outcomes/user_requirements.yaml \
  --chunks experiments/pancan/outputs/parsed_chunks.jsonl \
  --labels experiments/pancan/outputs/labeled_chunks.jsonl \
  --output experiments/pancan/outputs \
  --paper-id PMC9403942
```

Multiple `--paper-id` flags can be used in one command. Do not combine
`--paper-id` with `--force`; selected paper records are already replaced.

After any extraction rerun, regenerate the final CSV:

```bash
.venv/bin/python run_postprocess.py \
  --requirements examples/pancan_treatment_outcomes/user_requirements.yaml \
  --records experiments/pancan/outputs/extracted_records.jsonl \
  --output experiments/pancan/outputs
```

## Inputs

```text
user_requirements.yaml
WOS savedrecs.txt or local PMC/JATS XML files
optional presets/<project_name>/
```

Preset files can include:

```text
paper_filter.yaml
labeling_config.yaml
extraction_prompt.yaml
postprocess_config.yaml
```

If a preset exists, it is used directly. Otherwise, the pipeline falls back to
LLM-generated configuration or generic defaults where available.

## Outputs

The final analysis table is:

```text
experiments/pancan/outputs/records.csv
```

Main intermediate files:

```text
candidate_papers.jsonl
passed_papers.jsonl / rejected_papers.jsonl
downloaded_papers.jsonl
parsed_chunks.jsonl
labeled_chunks.jsonl
extracted_records.jsonl
postprocessed_records.jsonl
*_summary.json
```

`records.csv` is regenerated from the current postprocessed records. It is not
a per-batch CSV and is not appended line by line.

## Resume Rules

- Reuse the same output directory for incremental batches.
- `--limit N` processes up to N unfinished papers for that run.
- Completed papers are skipped.
- Failed extraction papers are retried later.
- Use `--force` only when intentionally recomputing a stage.

## Tests

```bash
.venv/bin/python test_wos_metadata_basic.py
.venv/bin/python test_paper_filter_basic.py
.venv/bin/python test_resume_basic.py
.venv/bin/python test_labeling_basic.py
.venv/bin/python test_extraction_basic.py
.venv/bin/python test_postprocess_basic.py
```

## More Docs

```text
docs/quickstart.md          Detailed stage-by-stage commands
docs/pipeline_overview.md   Pipeline flow and output semantics
docs/presets.md             Preset behavior and examples
docs/file_structure.md      Repository structure
```

Generated outputs, local papers, vector stores, caches, and credentials are
ignored by git.
