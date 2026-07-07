"""
Labeling Configuration Generator

使用 DSPy 调用 LLM 自动生成 labeling_config.yaml
从 user_requirements.yaml 读取领域描述和目标字段，
为每个字段生成：
- semantic_query
- regex_patterns
- table_header_keywords
- section_include / section_exclude
"""

import dspy
from typing import List, Dict
import yaml
import os
import json
from dotenv import load_dotenv

from .user_requirements import load_user_requirements


class GenerateLabelingConfigSignature(dspy.Signature):
    """Generate labeling configuration for a field in literature extraction."""

    domain_description: str = dspy.InputField(
        desc="Overall domain and paper type description"
    )
    field_name: str = dspy.InputField(
        desc="Name of the target field to extract"
    )
    field_definition: str = dspy.InputField(
        desc="Definition and expected content of the field"
    )

    # 输出字段
    semantic_query: str = dspy.OutputField(
        desc=(
            "One or two natural-language semantic retrieval sentences. "
            "Include expanded terms, synonyms, abbreviations, and common expressions "
            "for this field, but do not format it as a bare keyword list."
        )
    )
    regex_patterns_json: str = dspy.OutputField(
        desc='JSON array of regex patterns, each with "pattern", "description", and "strength" (strong/medium/weak)'
    )
    table_header_keywords: str = dspy.OutputField(
        desc="Comma-separated keywords that should appear in table headers or captions"
    )
    section_include: str = dspy.OutputField(
        desc="Comma-separated section names that likely contain this information (e.g., results, abstract, methods)"
    )
    section_exclude: str = dspy.OutputField(
        desc="Comma-separated section names to exclude: references, acknowledgements, conflict, funding"
    )


class LabelingConfigGenerator:
    def __init__(self):
        load_dotenv()

        # 从 .env 读取 LLM 配置
        model_name = os.getenv("LLM_MODEL", "gpt-4o")
        api_key = os.getenv("LLM_API_KEY")
        api_base = os.getenv("LLM_BASE_URL")
        # Kimi k2.6 在 JSON mode 下要求 temperature=1.0
        temperature = 1.0 if "kimi" in model_name.lower() else float(os.getenv("LLM_TEMPERATURE", "0.0"))

        # 配置 DSPy LM（max_tokens=8000 避免生成 config 时截断）
        self.lm = dspy.LM(
            model=f"openai/{model_name}",
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=8000
        )
        dspy.settings.configure(lm=self.lm)
        self.predictor = dspy.ChainOfThought(GenerateLabelingConfigSignature)

    def generate_field_config(
        self,
        domain_description: str,
        field_name: str,
        field_definition: str
    ) -> Dict:
        """为单个字段生成配置"""

        print(f"  Generating config for field: {field_name}")

        result = self.predictor(
            domain_description=domain_description,
            field_name=field_name,
            field_definition=field_definition
        )

        # 解析输出（result 字段可能因响应截断而为 None，需做防御）
        raw_table = result.table_header_keywords or ""
        raw_include = result.section_include or ""
        raw_exclude = result.section_exclude or ""
        table_header_keywords = [k.strip() for k in raw_table.split(",") if k.strip()]
        section_include = [s.strip() for s in raw_include.split(",") if s.strip()]
        section_exclude = [s.strip() for s in raw_exclude.split(",") if s.strip()]

        # 解析 regex_patterns_json
        try:
            regex_patterns = json.loads(result.regex_patterns_json)
            if not isinstance(regex_patterns, list):
                regex_patterns = []
        except:
            print(f"    Warning: Failed to parse regex_patterns_json, using empty list")
            regex_patterns = []

        return {
            "field_name": field_name,
            "field_definition": field_definition,
            "semantic_query": result.semantic_query,
            "regex_patterns": regex_patterns,
            "table_header_keywords": table_header_keywords,
            "section_include": section_include,
            "section_exclude": section_exclude,
            "retrieval_settings": {
                "semantic_fetch_k": 20,
                "regex_fetch_k": 20,
                "table_top_k": 5,
                "text_top_k": 5,
                "rrf_k": 60
            }
        }

    def generate_from_requirements(
        self,
        requirements_path: str,
        output_path: str
    ) -> Dict:
        """从 user_requirements.yaml 生成完整配置"""

        print(f"Loading requirements from: {requirements_path}")
        reqs = load_user_requirements(requirements_path)

        domain_desc = reqs.domain_description
        fields_config = []

        print(f"\nGenerating labeling config for {len(reqs.target_fields)} fields...")

        for field_spec in reqs.target_fields:
            field_config = self.generate_field_config(
                domain_description=domain_desc,
                field_name=field_spec.name,
                field_definition=field_spec.definition
            )
            fields_config.append(field_config)

        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        embedding_provider = os.getenv("EMBEDDING_PROVIDER")
        if not embedding_provider:
            embedding_provider = "gemini" if embedding_model.startswith("gemini-") else "openai"

        # 组装完整配置
        full_config = {
            "fields": fields_config,
            "embedding": {
                "model": embedding_model,
                "provider": embedding_provider
            },
            "labeler": {
                "model": os.getenv("LLM_MODEL", "gpt-4o"),
                # Kimi k2.6 在 JSON mode 下要求 temperature=1.0
                "temperature": 1.0 if "kimi" in os.getenv("LLM_MODEL", "gpt-4o").lower() else float(os.getenv("LLM_TEMPERATURE", "0.0")),
                "max_tokens": 1000
            }
        }

        print(f"\nSaving config to: {output_path}")
        with open(output_path, 'w') as f:
            yaml.dump(full_config, f, allow_unicode=True, sort_keys=False)

        print("✓ Config generation completed")
        return full_config
