# Literature Extraction Agent

A configurable agent pipeline for extracting structured records from scientific
literature. The implementation is inspired by ALLMAT/Sisyphus, but is designed
to be domain-transferable through `user_requirements.yaml` and optional preset
configs.

The current MVP supports:

- paper-level filtering from title and abstract
- JATS/XML full-text parsing
- section-aware paragraph, abstract, and table chunking
- embedding + regex retrieval for evidence labeling
- contextualized structured extraction
- lightweight post-processing and CSV export
- preset-first execution for stable domain rules

## Pipeline

```text
user_requirements.yaml + local XML papers
↓
Paper filter
↓
Article processing
↓
Labeling
↓
Extraction
↓
Post-processing
↓
JSONL + CSV records
```

See [Pipeline Overview](docs/pipeline_overview.md) for stage-level details.

## Preset-First Design

The pipeline first looks for hand-written presets under:

```text
presets/<project_name>/
```

Supported preset files:

```text
paper_filter.yaml
labeling_config.yaml
extraction_prompt.yaml
postprocess_config.yaml
```

If a preset exists, it is used directly. If not, the system falls back to
DSPy/LLM generation or generic defaults. This mirrors the ALLMAT style of using
stable engineered prompts and rules, while still allowing new domains to be
bootstrapped automatically.

See [Preset Guide](docs/presets.md) for details.

## Quickstart

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file locally with your model and embedding provider settings.
Do not commit `.env`.

Run the stages separately:

```bash
python run_paper_filter.py --requirements path/to/user_requirements.yaml --input path/to/xml_dir --output path/to/output/paper_filter
python run_preprocess.py --input path/to/xml_dir --output path/to/output/preprocess
python run_labeling.py --requirements path/to/user_requirements.yaml --chunks path/to/parsed_chunks.jsonl --output path/to/output/labeling
python run_extraction.py --requirements path/to/user_requirements.yaml --chunks path/to/parsed_chunks.jsonl --labels path/to/labeled_chunks.jsonl --output path/to/output/extraction
python run_postprocess.py --requirements path/to/user_requirements.yaml --records path/to/extracted_records.jsonl --output path/to/output/postprocess
```

See [Quickstart](docs/quickstart.md) for a fuller command sequence.

## Main Outputs

```text
parsed_chunks.jsonl
labeled_chunks.jsonl
extracted_records.jsonl
postprocessed_records.jsonl
records.csv
*_summary.json
```

Generated outputs, local papers, vector stores, caches, and credentials are
ignored by git.
