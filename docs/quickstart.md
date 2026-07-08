# Quickstart

This quickstart runs the pipeline stage by stage. Paths are examples; replace
them with your own project folder and corpus.

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

GEMINI_API_KEY    Google AI Studio key for Gemini embeddings
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=gemini-embedding-001

DSPY_CACHEDIR=/tmp/dspy_cache
NCBI_EMAIL=your_email@example.com
```

`NCBI_EMAIL` is optional but recommended for polite NCBI/PMC metadata requests.
WOS parsing and article preprocessing do not call an LLM, but paper filtering,
labeling, and extraction do.

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

## 3. WOS Metadata Ingestion

Skip this stage if you already have local XML files.

```bash
python run_wos_ingest.py \
  --input my_project/savedrecs.txt \
  --output my_project/output/wos_ingest
```

Main output:

```text
my_project/output/wos_ingest/candidate_papers.jsonl
```

## 4. Paper Filter

From WOS metadata:

```bash
python run_paper_filter.py \
  --requirements my_project/user_requirements.yaml \
  --metadata my_project/output/wos_ingest/candidate_papers.jsonl \
  --output my_project/output/paper_filter
```

From local XML files:

```bash
python run_paper_filter.py \
  --requirements my_project/user_requirements.yaml \
  --input my_project/input_papers \
  --output my_project/output/paper_filter
```

Main output:

```text
my_project/output/paper_filter/passed_papers.jsonl
```

## 5. Full-Text Acquisition

Use this stage after WOS-based paper filtering. It resolves pass-paper DOI/PMID
to PMCID and downloads available PMC XML.

```bash
python run_fulltext_acquisition.py \
  --passed my_project/output/paper_filter/passed_papers.jsonl \
  --output my_project/output/fulltext
```

Main output:

```text
my_project/output/fulltext/downloaded_papers.jsonl
my_project/output/fulltext/pmc_xml/
```

Skip this stage if `passed_papers.jsonl` already points to local XML files.

## 6. Article Processing

After WOS acquisition:

```bash
python run_preprocess.py \
  --passed my_project/output/fulltext/downloaded_papers.jsonl \
  --output my_project/output/preprocess
```

After local XML paper filtering:

```bash
python run_preprocess.py \
  --passed my_project/output/paper_filter/passed_papers.jsonl \
  --output my_project/output/preprocess
```

Main output:

```text
my_project/output/preprocess/parsed_chunks.jsonl
```

## 7. Labeling

```bash
python run_labeling.py \
  --requirements my_project/user_requirements.yaml \
  --chunks my_project/output/preprocess/parsed_chunks.jsonl \
  --output my_project/output/labeling
```

Main output:

```text
my_project/output/labeling/labeled_chunks.jsonl
```

## 8. Extraction

```bash
python run_extraction.py \
  --requirements my_project/user_requirements.yaml \
  --chunks my_project/output/preprocess/parsed_chunks.jsonl \
  --labels my_project/output/labeling/labeled_chunks.jsonl \
  --output my_project/output/extraction
```

Main output:

```text
my_project/output/extraction/extracted_records.jsonl
```

## 9. Post-Processing

```bash
python run_postprocess.py \
  --requirements my_project/user_requirements.yaml \
  --records my_project/output/extraction/extracted_records.jsonl \
  --output my_project/output/postprocess
```

Main outputs:

```text
my_project/output/postprocess/postprocessed_records.jsonl
my_project/output/postprocess/records.csv
```

## Notes

- Generated outputs should stay outside git.
- Local XML/PDF papers should stay outside git.
- Presets are source files and should be committed.
- `.env` files should never be committed.
