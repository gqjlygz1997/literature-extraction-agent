#!/usr/bin/env python3
"""Run the literature extraction pipeline with one command.

Examples:
  # From a Web of Science savedrecs.txt export or a directory of WOS txt files.
  python run_pipeline.py --requirements experiments/pancan/user_requirements.yaml --wos experiments/pancan/savedrecs.txt --output experiments/pancan/outputs --limit 10

  # From local JATS/PMC XML files.
  python run_pipeline.py --requirements experiments/pancan/user_requirements.yaml --xml experiments/pancan/input_papers --output experiments/pancan/outputs --limit 10
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.user_requirements import load_user_requirements


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run WOS/local-XML literature extraction pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--requirements", required=True, help="Path to user_requirements.yaml.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--wos",
        nargs="+",
        metavar="PATH",
        help="WOS savedrecs.txt file(s), or directories containing WOS .txt files.",
    )
    source.add_argument(
        "--xml",
        metavar="DIR",
        help="Directory of local JATS/PMC XML files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Shared output directory. Existing completed papers are skipped.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Process at most N unfinished papers per resumable stage. Default: 10.",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="Labeling domain name. Defaults to project_name in user_requirements.yaml.",
    )
    parser.add_argument(
        "--extractor-max-tokens",
        default=os.environ.get("EXTRACTOR_MAX_TOKENS", "4000"),
        help="EXTRACTOR_MAX_TOKENS for extraction. Default: 4000.",
    )
    parser.add_argument(
        "--extractor-timeout",
        default=os.environ.get("EXTRACTOR_TIMEOUT", "180"),
        help="EXTRACTOR_TIMEOUT for extraction. Default: 180.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute every stage instead of skipping existing results. Use with --limit 0.",
    )
    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Stop after extraction and do not rewrite records.csv.",
    )
    return parser


def _run(args: list[str], env: dict[str, str] | None = None) -> None:
    print("\n$ " + " ".join(args), flush=True)
    subprocess.run(args, check=True, env=env)


def _with_common_flags(cmd: list[str], limit: int | None, force: bool) -> list[str]:
    if limit is not None and limit > 0:
        cmd.extend(["--limit", str(limit)])
    if force:
        cmd.append("--force")
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.force and args.limit and args.limit > 0:
        parser.error("--force with a positive --limit can rewrite outputs partially; use --limit 0 or run one stage manually.")

    root = Path(__file__).resolve().parent
    requirements = Path(args.requirements)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    req = load_user_requirements(requirements)
    domain = args.domain or req.project_name
    py = sys.executable

    try:
        if args.wos:
            _run([
                py,
                str(root / "run_wos_ingest.py"),
                "--input",
                *args.wos,
                "--output",
                str(output),
            ])
            _run(_with_common_flags([
                py,
                str(root / "run_paper_filter.py"),
                "--requirements",
                str(requirements),
                "--metadata",
                str(output / "candidate_papers.jsonl"),
                "--output",
                str(output),
            ], args.limit, args.force))
            _run(_with_common_flags([
                py,
                str(root / "run_fulltext_acquisition.py"),
                "--passed",
                str(output / "passed_papers.jsonl"),
                "--output",
                str(output),
            ], args.limit, args.force))
            preprocess_input = output / "downloaded_papers.jsonl"
        else:
            _run(_with_common_flags([
                py,
                str(root / "run_paper_filter.py"),
                "--requirements",
                str(requirements),
                "--input",
                str(Path(args.xml)),
                "--output",
                str(output),
            ], args.limit, args.force))
            preprocess_input = output / "passed_papers.jsonl"

        _run(_with_common_flags([
            py,
            str(root / "run_preprocess.py"),
            "--passed",
            str(preprocess_input),
            "--output",
            str(output),
        ], args.limit, args.force))

        _run(_with_common_flags([
            py,
            str(root / "run_labeling.py"),
            "--requirements",
            str(requirements),
            "--chunks",
            str(output / "parsed_chunks.jsonl"),
            "--output",
            str(output),
            "--domain",
            domain,
        ], args.limit, args.force))

        extraction_env = os.environ.copy()
        extraction_env["EXTRACTOR_MAX_TOKENS"] = str(args.extractor_max_tokens)
        extraction_env["EXTRACTOR_TIMEOUT"] = str(args.extractor_timeout)
        _run(_with_common_flags([
            py,
            str(root / "run_extraction.py"),
            "--requirements",
            str(requirements),
            "--chunks",
            str(output / "parsed_chunks.jsonl"),
            "--labels",
            str(output / "labeled_chunks.jsonl"),
            "--output",
            str(output),
        ], args.limit, args.force), env=extraction_env)

        if not args.skip_postprocess:
            _run([
                py,
                str(root / "run_postprocess.py"),
                "--requirements",
                str(requirements),
                "--records",
                str(output / "extracted_records.jsonl"),
                "--output",
                str(output),
            ])

    except subprocess.CalledProcessError as exc:
        print(f"\nPipeline stopped because a stage failed with exit code {exc.returncode}.")
        return exc.returncode

    print(f"\nPipeline completed. Outputs: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
