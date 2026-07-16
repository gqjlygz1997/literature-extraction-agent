# Quickstart

This quickstart runs the pipeline stage by stage. Paths are examples; replace
them with your own project folder and corpus.

Use one output directory for the whole pipeline, for example
`my_project/outputs`. Each stage writes different file names, so incremental
batches can reuse the same directory safely.

## 1. Install

```bash
pip install -r requirements.txt
```

Create a local `.env` file with model and embedding settings. Keep it private
and do not commit it.

Start from the template:

```bash
cp .env.example .env
```

Minimum required variables:

```text
LLM_API_KEY       OpenAI-compatible chat model API key
LLM_BASE_URL      OpenAI-compatible API base URL, for example https://api.moonshot.cn/v1
LLM_MODEL         chat model name, for example kimi-k2.6
LLM_TEMPERATURE   default: 0.6

EXTRACTOR_MAX_TOKENS=4000
EXTRACTOR_TIMEOUT=90
EXTRACTOR_MAX_RETRIES=0

GEMINI_API_KEY    Google AI Studio key for Gemini embeddings
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=gemini-embedding-001

DSPY_CACHEDIR=/tmp/dspy_cache
NCBI_EMAIL=your_email@example.com
```

`NCBI_EMAIL` is optional but recommended for polite NCBI/PMC metadata requests.
WOS parsing and article preprocessing do not call an LLM, but paper filtering,
and extraction do. Labeling uses retrieval only by default in the pancan preset;
enable its optional LLM confirmation only when stricter chunk filtering is worth
the additional cost.

## 2. Prepare Inputs

Example `user_requirements.yaml` files are included under:

```text
examples/pancan_treatment_outcomes/user_requirements.yaml
examples/hea_mechanical_properties/user_requirements.yaml
```

Input can start from either a WOS export or local JATS/XML files.

WOS input layout:

```text
my_project/
├── user_requirements.yaml
└── savedrecs.txt
```

Local XML input layout:

```text
my_project/
├── user_requirements.yaml
└── input_papers/
    ├── PMC0000001.xml
    └── PMC0000002.xml
```

If you have domain presets, place them under:

```text
presets/<project_name>/
```

where `<project_name>` matches the value in `user_requirements.yaml`.

## 2.1 Batch and Resume Rules

Every expensive stage is resumable by default. The practical rule is:

```text
same output directory + --limit 10 = run the next 10 unfinished papers
```

- Completed papers are skipped on later runs.
- Failed extraction papers are retried on later runs.
- `--limit` can be any number or omitted to process all unfinished papers.
- `--force` recomputes that stage instead of skipping existing results.
- If you change prompts, presets, model settings, or parsed input, use
  `--force` for the affected downstream stage.

## 3. WOS Metadata Ingestion

Skip this stage if you already have local XML files.

```bash
python run_wos_ingest.py \
  --input my_project/savedrecs.txt \
  --output my_project/outputs
```

Main output:

```text
my_project/outputs/candidate_papers.jsonl
```

## 4. Paper Filter

From WOS metadata:

```bash
python run_paper_filter.py \
  --requirements my_project/user_requirements.yaml \
  --metadata my_project/outputs/candidate_papers.jsonl \
  --output my_project/outputs \
  --limit 10
```

From local XML files:

```bash
python run_paper_filter.py \
  --requirements my_project/user_requirements.yaml \
  --input my_project/input_papers \
  --output my_project/outputs \
  --limit 10
```

Main output:

```text
my_project/outputs/passed_papers.jsonl
```

For the pancreatic-cancer preset, papers that are clearly biomarker-only,
diagnostic, prognostic, or risk-factor studies are rejected before full-text
acquisition. Ambiguous papers are retained for recall; clearly irrelevant
papers lack intervention-linked evidence and are rejected.

## 5. Full-Text Acquisition

Use this stage after WOS-based paper filtering. It resolves pass-paper DOI/PMID
to PMCID and downloads available PMC XML.

```bash
python run_fulltext_acquisition.py \
  --passed my_project/outputs/passed_papers.jsonl \
  --output my_project/outputs \
  --limit 10
```

Main output:

```text
my_project/outputs/downloaded_papers.jsonl
my_project/outputs/pmc_xml/
```

Skip this stage if `passed_papers.jsonl` already points to local XML files.

## 6. Article Processing

After WOS acquisition:

```bash
python run_preprocess.py \
  --passed my_project/outputs/downloaded_papers.jsonl \
  --output my_project/outputs \
  --limit 10
```

After local XML paper filtering:

```bash
python run_preprocess.py \
  --passed my_project/outputs/passed_papers.jsonl \
  --output my_project/outputs \
  --limit 10
```

Main output:

```text
my_project/outputs/parsed_chunks.jsonl
```

## 7. Labeling

```bash
python run_labeling.py \
  --requirements my_project/user_requirements.yaml \
  --chunks my_project/outputs/parsed_chunks.jsonl \
  --output my_project/outputs \
  --domain YOUR_PROJECT_NAME \
  --limit 10
```

Main output:

```text
my_project/outputs/labeled_chunks.jsonl
```

## 8. Extraction

```bash
python run_extraction.py \
  --requirements my_project/user_requirements.yaml \
  --chunks my_project/outputs/parsed_chunks.jsonl \
  --labels my_project/outputs/labeled_chunks.jsonl \
  --output my_project/outputs \
  --limit 10
```

Main output:

```text
my_project/outputs/extracted_records.jsonl
```

Some domain prompt presets also apply local constraints after extraction. The
pancan preset only keeps endpoints compatible with the selected evidence type,
for example `clearance` for `pk` rather than `clinical_outcome`.

## 9. Post-Processing

```bash
python run_postprocess.py \
  --requirements my_project/user_requirements.yaml \
  --records my_project/outputs/extracted_records.jsonl \
  --output my_project/outputs
```

Main outputs:

```text
my_project/outputs/postprocessed_records.jsonl
my_project/outputs/records.csv
```

## Notes

- Generated outputs should stay outside git.
- Local XML/PDF papers should stay outside git.
- Presets are source files and should be committed.
- `.env` files should never be committed.
- Reuse the same output directories for incremental batches. By default the
  pipeline skips papers already present in the stage output; use `--limit 10`
  to process the next 10 unfinished papers, and `--force` only when you
  intentionally want to recompute a stage.
