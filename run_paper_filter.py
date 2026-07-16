#!/usr/bin/env python3
"""CLI entry point for the paper-filter MVP.

Main usage (requirements file):
    python run_paper_filter.py \\
      --requirements experiments/pancan/user_requirements.yaml \\
      --input       experiments/pancan/input_papers \\
      --output      experiments/pancan/outputs

WOS metadata usage:
    python run_paper_filter.py \\
      --requirements experiments/pancan/user_requirements.yaml \\
      --metadata    experiments/pancan/wos_ingest/candidate_papers.jsonl \\
      --output      experiments/pancan/paper_filter

Fallback (inline args, no requirements file):
    python run_paper_filter.py \\
      --domain "Extract treatment outcomes from pancreatic cancer clinical papers" \\
      --fields "treatment_regimen,os,pfs,orr" \\
      --field-definitions "treatment_regimen=treatment name;os=overall survival;pfs=progression-free survival;orr=objective response rate" \\
      --input  experiments/pancan/input_papers \\
      --output experiments/pancan/outputs

Reuse an existing generated config (skip LLM generation):
    python run_paper_filter.py \\
      --config  experiments/pancan/outputs/paper_filter.yaml \\
      --input   experiments/pancan/input_papers \\
      --output  experiments/pancan/outputs
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Load .env before any imports that need API keys, when python-dotenv is installed.
try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:
    pass
else:
    load_dotenv(find_dotenv())

# Add src/ to path so `agent` package is importable without installation
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_paper_filter")


# ---------------------------------------------------------------------------
# LLM client factory
# ---------------------------------------------------------------------------

def _make_llm_client(model_override: str | None = None):
    """Build a LangChain ChatOpenAI client from .env settings.

    Required env vars: LLM_API_KEY
    Optional env vars: LLM_BASE_URL (default: Kimi/Moonshot), LLM_MODEL, LLM_TEMPERATURE
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        logger.error("langchain-openai is not installed. Run: pip install langchain-openai")
        sys.exit(1)

    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        logger.error("LLM_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    base_url = os.environ.get("LLM_BASE_URL", "https://api.moonshot.cn/v1")
    model = model_override or os.environ.get("LLM_MODEL", "kimi-k2.6")
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.6"))

    extra_body = {"thinking": {"type": "disabled"}}
    client = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        extra_body=extra_body,
    )
    return client, model


def _configure_dspy(model_override: str | None = None) -> str:
    """Configure DSPy global LM from .env settings.

    Uses litellm's openai-compatible backend.
    Required env vars: LLM_API_KEY
    Optional env vars: LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE
    Returns the resolved model name.
    """
    try:
        import dspy
    except ImportError:
        logger.error("dspy is not installed. Run: pip install 'dspy>=2.5.22'")
        sys.exit(1)

    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        logger.error("LLM_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    base_url = os.environ.get("LLM_BASE_URL", "https://api.moonshot.cn/v1")
    model = model_override or os.environ.get("LLM_MODEL", "kimi-k2.6")
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.6"))

    # litellm format for custom OpenAI-compatible endpoints: "openai/<model>"
    lm = dspy.LM(
        model=f"openai/{model}",
        api_key=api_key,
        api_base=base_url,
        temperature=temperature,
    )
    dspy.configure(lm=lm)
    logger.info("DSPy configured: model=openai/%s, base_url=%s, temperature=%s",
                model, base_url, temperature)
    return model


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run paper-filter MVP: classify local XML/HTML papers or WOS metadata as pass/reject.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- input specification (three mutually exclusive modes) ---
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--requirements", metavar="PATH",
        help="Path to user_requirements.yaml (recommended).",
    )
    input_group.add_argument(
        "--config", metavar="PATH",
        help="Path to an existing paper_filter.yaml. "
             "Skips LLM config generation.",
    )

    # fallback inline args (used when --requirements is absent and --config is absent)
    parser.add_argument(
        "--domain", metavar="TEXT",
        help="Domain description (fallback when --requirements is not provided).",
    )
    parser.add_argument(
        "--fields", metavar="field1,field2,...",
        help="Comma-separated target field names (fallback).",
    )
    parser.add_argument(
        "--field-definitions", metavar="k=v;k=v", dest="field_definitions",
        help="Semicolon-separated field definitions; required for every --fields item in fallback mode.",
        default="",
    )

    # --- paper source specification ---
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--input", metavar="DIR",
        help="Directory containing local XML / HTML paper files.",
    )
    source_group.add_argument(
        "--metadata", metavar="PATH",
        help="candidate_papers.jsonl produced by run_wos_ingest.py.",
    )

    # --- required output ---
    parser.add_argument(
        "--output", required=True, metavar="DIR",
        help="Directory where output files will be written.",
    )
    parser.add_argument(
        "--preset-dir", metavar="DIR",
        help="Directory containing project presets (default: ./presets).",
    )
    parser.add_argument(
        "--no-presets", action="store_true",
        help="Do not auto-load presets/<project_name>/paper_filter.yaml.",
    )

    # --- optional flags ---
    parser.add_argument(
        "--model", metavar="MODEL_NAME",
        help="Override the LLM model (default: reads LLM_MODEL from .env).",
    )
    parser.add_argument(
        "--limit", type=int, metavar="N",
        help="Process at most N new papers. In resume mode, already processed papers are skipped first.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Reprocess papers even if output files already contain results.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse metadata only; skip LLM classification.",
    )
    parser.add_argument(
        "--use-dspy", action="store_true", dest="use_dspy",
        help="Use DSPyPaperFilter (ChainOfThought) instead of the default LLMLabeler.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        from agent.config_generator import ConfigGenerator
        from agent.preset_manager import find_preset_file
        from agent.tools.document_parser import DocumentParser
        from agent.tools.llm_labeler import LLMLabeler
        from agent.user_requirements import load_user_requirements, build_user_requirements_from_args
        from agent.workflow import DomainExtractionWorkflow
    except ImportError as exc:
        logger.error("Missing Python dependency while loading agent modules: %s", exc)
        logger.error("Install the project dependencies in the active Python environment.")
        return 1

    # -- resolve user requirements / existing config ----------------------
    paper_filter_config = None
    user_requirements = None

    if args.config:
        # reuse existing config, skip generation
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error("--config path not found: %s", config_path)
            return 1
        logger.info("Loading existing paper filter config from %s", config_path)
        paper_filter_config = ConfigGenerator.load(config_path)

    elif args.requirements:
        req_path = Path(args.requirements)
        if not req_path.exists():
            logger.error("--requirements path not found: %s", req_path)
            return 1
        user_requirements = load_user_requirements(req_path)
        logger.info(
            "Loaded user requirements: project=%s, fields=%d",
            user_requirements.project_name,
            len(user_requirements.target_fields),
        )
        if not args.no_presets:
            preset_path = find_preset_file(
                req_path, "paper_filter.yaml", preset_dir=args.preset_dir
            )
            if preset_path:
                logger.info("Using preset paper filter config: %s", preset_path)
                paper_filter_config = ConfigGenerator.load(preset_path)

    elif args.domain and args.fields:
        try:
            user_requirements = build_user_requirements_from_args(
                domain=args.domain,
                fields=args.fields,
                field_definitions=args.field_definitions or "",
            )
        except ValueError as exc:
            logger.error("Invalid inline requirements: %s", exc)
            return 1
        logger.info("Built requirements from CLI args.")

    else:
        parser.error(
            "Provide one of: --requirements PATH, --config PATH, "
            "or both --domain TEXT and --fields field1,field2,..."
        )

    # -- build components -------------------------------------------------
    if args.dry_run:
        llm_client = None
        model_name = args.model or os.environ.get("LLM_MODEL", "")
        labeler = LLMLabeler(llm_client=None, model_name=model_name)
    elif args.use_dspy:
        from agent.tools.dspy_paper_filter import DSPyPaperFilter
        model_name = _configure_dspy(args.model)
        labeler = DSPyPaperFilter(model_name=model_name)
        llm_client = None  # ConfigGenerator still needs its own client
        llm_client, _ = _make_llm_client(args.model)
    else:
        llm_client, model_name = _make_llm_client(args.model)
        labeler = LLMLabeler(llm_client=llm_client, model_name=model_name)

    config_generator = ConfigGenerator(llm_client=llm_client, model_name=model_name)
    document_parser = DocumentParser()

    workflow = DomainExtractionWorkflow(
        config_generator=config_generator,
        document_parser=document_parser,
        paper_filter_labeler=labeler,
    )

    # -- run --------------------------------------------------------------
    logger.info("Starting paper-filter MVP.")
    if args.dry_run:
        logger.info("DRY RUN: LLM classification is disabled.")

    counts = workflow.run_paper_filter_mvp(
        papers_dir=args.input,
        metadata_path=args.metadata,
        user_requirements=user_requirements,
        output_dir=args.output,
        paper_filter_config=paper_filter_config,
        dry_run=args.dry_run,
        limit=args.limit,
        resume=not args.force,
    )

    print(
        f"\nDone.  total={counts['total']}  "
        f"passed={counts['passed']}  "
        f"rejected={counts['rejected']}  "
        f"error={counts['error']}  "
        f"dry_run={counts.get('dry_run', 0)}"
    )
    print(f"Results written to: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
