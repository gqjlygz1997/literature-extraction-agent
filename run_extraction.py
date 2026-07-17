#!/usr/bin/env python3
"""
Run Extraction Stage — extract structured records from labeled chunks.

用法：
  python run_extraction.py \
    --requirements experiments/pancan/user_requirements.yaml \
    --chunks       experiments/pancan/outputs/parsed_chunks.jsonl \
    --labels       experiments/pancan/outputs/labeled_chunks.jsonl \
    --output       experiments/pancan/extraction_output

  # 指定模型（默认从环境变量 EXTRACTOR_MODEL 读取）
  python run_extraction.py ... --model kimi-k2.6
"""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).with_name(".env"))
load_dotenv()

from src.agent.workflow import DomainExtractionWorkflow


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Stage 2 Extraction — extract structured records from labeled chunks"
    )
    parser.add_argument("--requirements", required=True,
                        help="Path to user_requirements.yaml")
    parser.add_argument("--chunks", required=True,
                        help="Path to parsed_chunks.jsonl (from preprocessing stage)")
    parser.add_argument("--labels", required=True,
                        help="Path to labeled_chunks.jsonl (from labeling stage)")
    parser.add_argument("--output", required=True,
                        help="Output directory for extracted_records.jsonl and summary")
    parser.add_argument("--model", default=None,
                        help="LLM model name (overrides EXTRACTOR_MODEL env var)")
    parser.add_argument("--prompt-preset", default=None,
                        help="Path to extraction_prompt.yaml. Skips dynamic prompt generation.")
    parser.add_argument("--preset-dir", default=None,
                        help="Directory containing project presets (default: ./presets)")
    parser.add_argument("--no-presets", action="store_true",
                        help="Do not auto-load presets/<project_name>/extraction_prompt.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N new papers. In resume mode, already extracted papers are skipped first.")
    parser.add_argument("--paper-id", action="append", default=[],
                        help="Re-run and replace one paper_id in the existing extraction output. Can be repeated.")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess papers even if extracted_records.jsonl already contains results.")
    args = parser.parse_args()

    if args.paper_id and args.force:
        parser.error("--paper-id already replaces selected papers; do not combine it with --force.")

    # 验证输入文件
    for label, path_str in [
        ("requirements", args.requirements),
        ("chunks",       args.chunks),
        ("labels",       args.labels),
    ]:
        if not Path(path_str).exists():
            logger.error(f"❌ {label} file not found: {path_str}")
            return 1

    workflow = DomainExtractionWorkflow(
        config_generator=None,
        document_parser=None,
        paper_filter_labeler=None,
    )

    try:
        result = workflow.run_extraction(
            requirements_path=args.requirements,
            parsed_chunks_path=args.chunks,
            labeled_chunks_path=args.labels,
            output_dir=args.output,
            model_name=args.model,
            prompt_preset_path=args.prompt_preset,
            preset_dir=args.preset_dir,
            use_presets=not args.no_presets,
            limit=args.limit,
            resume=not args.force,
            target_paper_ids=args.paper_id,
        )

        logger.info("\n✅ Extraction completed!")
        logger.info(f"   Records:          {Path(args.output) / 'extracted_records.jsonl'}")
        logger.info(f"   This run papers:  {result.get('processed_this_run', 0)}")
        logger.info(f"   Skipped existing: {result.get('skipped_existing', 0)}")
        paper_ids = result.get("processed_paper_ids_this_run") or []
        if paper_ids:
            logger.info(f"   This run IDs:     {', '.join(paper_ids)}")
        logger.info(f"   This run raw:     {result.get('records_raw_this_run', 0)}")
        logger.info(f"   This run cleaned: {result.get('records_after_cleanup_this_run', 0)}")
        logger.info(f"   This run rejected by endpoint constraints: {result.get('endpoint_constraint_rejected_this_run', 0)}")
        logger.info(f"   This run duplicates removed: {result.get('duplicates_removed_this_run', 0)}")
        logger.info(f"   Total papers:     {result['total_papers_processed']}")
        logger.info(f"   Failed papers:    {result['total_papers_failed']}")
        logger.info(f"   Total records:    {result['total_records_extracted']}")
        logger.info(f"   Total duplicates removed: {result['duplicates_removed']}")
        return 0

    except Exception as e:
        logger.exception(f"❌ Extraction failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
