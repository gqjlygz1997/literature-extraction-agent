#!/usr/bin/env python3
"""
Run Labeling Stage - identify relevant evidence chunks

用法：
  # 首次运行：调用 LLM 生成 labeling_config.yaml
  python run_labeling.py \
    --requirements experiments/melanoma_trials/user_requirements.yaml \
    --chunks experiments/melanoma_trials/parsed_chunks.jsonl \
    --output experiments/melanoma_trials/labeling_output \
    --domain melanoma_trials

  # 复用已有配置（跳过 LLM 生成步骤，可手动编辑配置后再跑）
  python run_labeling.py \
    --requirements experiments/melanoma_trials/user_requirements.yaml \
    --chunks experiments/melanoma_trials/parsed_chunks.jsonl \
    --output experiments/melanoma_trials/labeling_output \
    --domain melanoma_trials \
    --config experiments/melanoma_trials/labeling_output/labeling_config.yaml

注意：
  - 如果 --output 目录下已有 labeling_config.yaml，也会自动复用（无需 --config）
  - 如需强制重新生成，先删除 labeling_config.yaml 再运行
"""

import argparse
import logging
import os
from pathlib import Path

# 必须在导入 dspy 之前设置 DSPY_CACHEDIR，否则 DSPy 会尝试写默认只读路径
# .env 里设置 DSPY_CACHEDIR=/tmp/dspy_cache
from dotenv import load_dotenv
load_dotenv()
if not os.environ.get("DSPY_CACHEDIR"):
    os.environ["DSPY_CACHEDIR"] = "/tmp/dspy_cache"

from src.agent.workflow import DomainExtractionWorkflow


def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Run Labeling Stage - identify relevant evidence chunks"
    )

    parser.add_argument(
        "--requirements",
        type=str,
        required=True,
        help="Path to user_requirements.yaml"
    )

    parser.add_argument(
        "--chunks",
        type=str,
        required=True,
        help="Path to parsed_chunks.jsonl (from preprocessing stage)"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for labeled_chunks.jsonl and config"
    )

    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain name (for Chroma vector store organization)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional: path to existing labeling_config.yaml to reuse (skip LLM generation)"
    )
    parser.add_argument(
        "--preset-dir",
        type=str,
        default=None,
        help="Directory containing project presets (default: ./presets)"
    )
    parser.add_argument(
        "--no-presets",
        action="store_true",
        help="Do not auto-load presets/<project_name>/labeling_config.yaml"
    )

    args = parser.parse_args()

    # 验证输入文件存在
    req_path = Path(args.requirements)
    chunks_path = Path(args.chunks)

    if not req_path.exists():
        logger.error(f"❌ Requirements file not found: {req_path}")
        return 1

    if not chunks_path.exists():
        logger.error(f"❌ Chunks file not found: {chunks_path}")
        return 1

    # 创建 workflow（labeling 阶段不需要 document_parser）
    workflow = DomainExtractionWorkflow(
        config_generator=None,
        document_parser=None,
        paper_filter_labeler=None,
    )

    # 运行 labeling
    try:
        result = workflow.run_labeling_mvp(
            requirements_path=req_path,
            parsed_chunks_path=chunks_path,
            output_dir=args.output,
            domain_name=args.domain,
            config_path=args.config,  # 传入 --config 参数
            preset_dir=args.preset_dir,
            use_presets=not args.no_presets,
        )

        logger.info("\n✅ Labeling completed successfully!")
        logger.info(f"   Labeled chunks: {result['labeled_chunks_path']}")
        logger.info(f"   Summary: {result['summary_path']}")
        logger.info(f"   Total labeled chunks: {result['total_labeled_chunks']}")
        logger.info(f"   Total label assignments: {result['total_label_assignments']}")

        return 0

    except Exception as e:
        logger.exception(f"❌ Labeling failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
