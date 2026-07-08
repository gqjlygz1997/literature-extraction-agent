#!/usr/bin/env python3
"""Parse Web of Science savedrecs.txt files into candidate metadata JSONL."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_wos_ingest")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse WOS savedrecs text exports into candidate_papers.jsonl.",
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        metavar="PATH",
        help="One or more savedrecs.txt files or directories containing .txt files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="Directory where candidate_papers.jsonl is written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Write only the first N records for smoke testing.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        from agent.tools.wos_parser import discover_wos_files, parse_wos_files
    except ImportError as exc:
        logger.error("Missing dependency while loading WOS parser: %s", exc)
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_wos_files(args.input)
    rows = parse_wos_files(paths)
    if args.limit:
        rows = rows[:args.limit]

    candidate_path = output_dir / "candidate_papers.jsonl"
    with open(candidate_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_files": [str(p) for p in paths],
        "total_records": len(rows),
        "with_doi": sum(1 for r in rows if r.get("doi")),
        "with_pmid": sum(1 for r in rows if r.get("pmid")),
        "with_abstract": sum(1 for r in rows if r.get("abstract")),
        "candidate_papers_path": str(candidate_path.resolve()),
    }
    with open(output_dir / "wos_ingestion_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(
        f"\nDone. records={summary['total_records']} "
        f"doi={summary['with_doi']} pmid={summary['with_pmid']} "
        f"abstract={summary['with_abstract']}"
    )
    print(f"Results written to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
