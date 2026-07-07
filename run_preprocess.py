#!/usr/bin/env python3
"""CLI entry point for the preprocessing MVP (phase 2).

Parse the full text of papers that passed the filter into section-aware
content chunks (paragraph / table / abstract), preserving section hierarchy.

Usage:
    python run_preprocess.py \\
      --passed experiments/hea/outputs/passed_papers.jsonl \\
      --output experiments/hea/outputs

Smoke test (first 3 papers):
    python run_preprocess.py \\
      --passed experiments/hea/outputs/passed_papers.jsonl \\
      --output experiments/hea/outputs \\
      --limit 3

Notes:
  - Only JATS/PMC XML is supported. Non-XML rows are recorded as errors.
  - No LLM is called in this stage.
  - source_path is read from each passed_papers.jsonl row; no --input needed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src/ to path so `agent` package is importable without installation
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_preprocess")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run preprocessing MVP: parse passed papers into section-aware chunks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--passed", required=True, metavar="PATH",
        help="Path to passed_papers.jsonl produced by the paper-filter stage.",
    )
    parser.add_argument(
        "--output", required=True, metavar="DIR",
        help="Directory where parsed_chunks.jsonl and preprocessing_summary.json are written.",
    )
    parser.add_argument(
        "--limit", type=int, metavar="N",
        help="Process only the first N papers (smoke-test mode).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    passed_path = Path(args.passed)
    if not passed_path.exists():
        logger.error("--passed path not found: %s", passed_path)
        return 1

    try:
        from agent.tools.document_parser import DocumentParser
        from agent.workflow import DomainExtractionWorkflow
    except ImportError as exc:
        logger.error("Missing Python dependency while loading agent modules: %s", exc)
        return 1

    document_parser = DocumentParser()
    workflow = DomainExtractionWorkflow(
        config_generator=None,
        document_parser=document_parser,
        paper_filter_labeler=None,
    )

    logger.info("Starting preprocessing MVP.")
    summary = workflow.run_preprocess_mvp(
        passed_papers_path=passed_path,
        output_dir=args.output,
        limit=args.limit,
    )

    print(
        f"\nDone.  papers={summary['total_papers']}  "
        f"parsed_ok={summary['parsed_ok']}  "
        f"parse_error={summary['parse_error']}  "
        f"chunks={summary['total_chunks']} "
        f"(para={summary['paragraph_chunks']} "
        f"table={summary['table_chunks']} "
        f"abstract={summary['abstract_chunks']})"
    )
    print(f"Results written to: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
