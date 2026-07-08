#!/usr/bin/env python3
"""Resolve passed DOI/PMID metadata to PMCID and download PMC XML."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:
    pass
else:
    load_dotenv(find_dotenv())

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_fulltext_acquisition")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download PMC XML for papers that passed paper filtering.",
    )
    parser.add_argument(
        "--passed",
        required=True,
        metavar="PATH",
        help="passed_papers.jsonl from paper filter over WOS metadata.",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="Directory where XML and acquisition outputs are written.",
    )
    parser.add_argument(
        "--email",
        default=os.environ.get("NCBI_EMAIL", ""),
        help="Optional email for NCBI requests; can also use NCBI_EMAIL.",
    )
    parser.add_argument("--limit", type=int, metavar="N")
    parser.add_argument("--sleep", type=float, default=0.34, help="Delay between NCBI requests.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        from agent.tools.pmc_downloader import acquire_pmc_xml
    except ImportError as exc:
        logger.error("Missing dependency while loading PMC downloader: %s", exc)
        return 1

    passed_path = Path(args.passed)
    if not passed_path.exists():
        logger.error("--passed path not found: %s", passed_path)
        return 1

    rows: list[dict] = []
    with open(passed_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    summary = acquire_pmc_xml(
        rows,
        args.output,
        email=args.email,
        sleep_seconds=args.sleep,
        limit=args.limit,
    )

    print(
        f"\nDone. total={summary['total']} downloaded={summary['downloaded']} "
        f"already_exists={summary['already_exists']} no_pmcid={summary['no_pmcid']} "
        f"xml_unavailable={summary['xml_unavailable']} "
        f"error={summary['error']}"
    )
    print(f"Results written to: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
