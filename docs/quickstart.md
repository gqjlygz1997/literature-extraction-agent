# Quickstart

This quickstart runs the pipeline stage by stage. Paths are examples; replace
them with your own project folder and XML corpus.

## 1. Install

```bash
pip install -r requirements.txt
```

Create a local `.env` file with model and embedding settings. Keep it private
and do not commit it.

Example variable names:

```text
LLM_MODEL
LLM_API_KEY
LLM_BASE_URL
EMBEDDING_MODEL
GEMINI_API_KEY
DSPY_CACHEDIR
```

## 2. Prepare Inputs

Expected input layout:

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

## 3. Paper Filter

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

## 4. Article Processing

```bash
python run_preprocess.py \
  --input my_project/input_papers \
  --output my_project/output/preprocess
```

Main output:

```text
my_project/output/preprocess/parsed_chunks.jsonl
```

## 5. Labeling

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

## 6. Extraction

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

## 7. Post-Processing

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
