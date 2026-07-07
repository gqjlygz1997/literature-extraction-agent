#!/usr/bin/env python3
"""
Run Stage 3 Post-processing — normalize extracted records and export CSV.

用法：
  python run_postprocess.py \
    --requirements experiments/pancan/user_requirements.yaml \
    --records      experiments/pancan/extraction_output/extracted_records.jsonl \
    --output       experiments/pancan/postprocess_output
"""

import argparse
import logging
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
        description="Stage 3 Post-processing — normalize extracted records"
    )
    parser.add_argument("--requirements", required=True,
                        help="Path to user_requirements.yaml")
    parser.add_argument("--records", required=True,
                        help="Path to extracted_records.jsonl")
    parser.add_argument("--output", required=True,
                        help="Output directory for postprocessed records and CSV")
    parser.add_argument("--config", default=None,
                        help="Path to postprocess_config.yaml")
    parser.add_argument("--preset-dir", default=None,
                        help="Directory containing project presets (default: ./presets)")
    parser.add_argument("--no-presets", action="store_true",
                        help="Do not auto-load presets/<project_name>/postprocess_config.yaml")
    args = parser.parse_args()

    for label, path_str in [
        ("requirements", args.requirements),
        ("records", args.records),
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
        result = workflow.run_postprocess(
            requirements_path=args.requirements,
            extracted_records_path=args.records,
            output_dir=args.output,
            config_path=args.config,
            preset_dir=args.preset_dir,
            use_presets=not args.no_presets,
        )

        logger.info("\n✅ Post-processing completed!")
        logger.info(f"   JSONL: {Path(args.output) / 'postprocessed_records.jsonl'}")
        logger.info(f"   CSV:   {Path(args.output) / 'records.csv'}")
        logger.info(f"   Input records:  {result['records_input']}")
        logger.info(f"   Output records: {result['records_output']}")
        logger.info(f"   Invalid removed: {result['invalid_removed']}")
        logger.info(f"   Duplicates removed: {result['duplicates_removed']}")
        return 0

    except Exception as e:
        logger.exception(f"❌ Post-processing failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
