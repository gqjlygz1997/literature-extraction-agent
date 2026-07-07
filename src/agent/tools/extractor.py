"""
extractor.py — LLM-backed structured extractor for Stage 2 Extraction.

对应 ALLMAT:
  template | model.with_structured_output(result_model, method='json_schema')
  + extract() 函数调用链（extract_lc.py）

设计原则：
- 使用 ChatPromptTemplate（system + user）三段式，对应 ALLMAT prompt.py
- 通过 with_structured_output(method="json_schema") 强制结构化输出
- 异常捕获：token 超限、格式错误等，失败时返回空列表并记录错误
- 不含任何领域硬编码，所有 prompt 内容由 extraction_schema.py 生成后传入

与 ALLMAT 的当前有意偏离之一：
- 暂不实现 DetectProcesses / 模板注入（对应 synthesis.py），保留扩展位置
"""

from __future__ import annotations

import logging
import os
import json
import re
from typing import Any

logger = logging.getLogger(__name__)


def extract_records(
    context_str: str,
    system_message: str,
    instruction: str,
    records_model,
    *,
    model_name: str | None = None,
) -> tuple[list[dict], str]:
    """
    调用 LLM 对 context 做 contextualized extraction，返回 record 列表。

    对应 ALLMAT:
        chain = template | extraction_model.with_structured_output(
            result_model, method='json_schema'
        )
        records = chain.invoke({...}).records

    参数：
        context_str:    context_builder.build_context() 输出的上下文字符串
        system_message: extraction_schema.build_system_message() 的输出
        instruction:    extraction_schema.create_instruction() 的输出
        records_model:  extraction_schema.build_records_model() 的输出（Pydantic Records）
        model_name:     LLM 模型名；None 时从环境变量 EXTRACTOR_MODEL 读取

    返回：
        (records_list, status)
        - records_list: List[dict]，每条为 record.fields 字段的 dict
                        字段全量补齐，缺失字段值为 None
        - status: "ok" | "failed:<ErrorType>"
    """
    from langchain_core.prompts import ChatPromptTemplate

    # 构建三段式 ChatPromptTemplate（对应 ALLMAT prompt.py 的 ChatPromptTemplate）
    template = ChatPromptTemplate([
        ("system", "{system_message}"),
        ("user",
         "[START OF EVIDENCE CONTEXT]\n{context}\n[END OF EVIDENCE CONTEXT]\n\n"
         "Instruction:\n{instruction}"),
    ])

    if os.environ.get("EXTRACTOR_MOCK") == "1":
        return _mock_extract_records(context_str, records_model), "ok:mock"

    resolved_model_name = _resolve_model_name(model_name)
    llm = _build_llm(resolved_model_name)

    # 绑定结构化输出（对应 ALLMAT with_structured_output(method="json_schema")）
    # 若目标 LLM 不支持 json_schema，自动降级到 json_mode。
    methods = ["plain_json", "json_mode"] if "kimi" in resolved_model_name.lower() else ["json_schema", "json_mode", "plain_json"]
    last_error: Exception | None = None

    for method in methods:
        try:
            if method == "plain_json":
                records = _invoke_plain_json(
                    template, llm, records_model, system_message, context_str, instruction,
                )
            else:
                result = _invoke_structured(
                    template, llm, records_model, system_message, context_str, instruction,
                    method=method,
                )
                records = _records_to_dicts(result)
            return records, "ok" if method == "json_schema" else f"ok:{method}"
        except Exception as e:
            last_error = e
            logger.warning(f"{method} extraction failed: {e}")

    err_type = type(last_error).__name__ if last_error else "UnknownError"
    logger.error(f"Extraction failed ({err_type}): {last_error}")
    return [], f"failed:{err_type}"


def _invoke_structured(
    template,
    llm,
    records_model,
    system_message: str,
    context_str: str,
    instruction: str,
    *,
    method: str,
):
    structured_llm = llm.with_structured_output(records_model, method=method)
    chain = template | structured_llm
    return chain.invoke({
        "system_message": system_message,
        "context": context_str,
        "instruction": instruction,
    })


def _invoke_plain_json(
    template,
    llm,
    records_model,
    system_message: str,
    context_str: str,
    instruction: str,
) -> list[dict]:
    plain_instruction = (
        instruction
        + "\n\nReturn ONLY a valid JSON object. Do not include markdown fences, explanation, or reasoning.\n"
        + "The JSON object must have this exact top-level shape: {\"records\": [...]}.\n"
        + "Each item in records must only use the fields listed above. Use null for unknown values."
    )
    chain = template | llm
    result = chain.invoke({
        "system_message": system_message,
        "context": context_str,
        "instruction": plain_instruction,
    })
    text = getattr(result, "content", result)
    if isinstance(text, list):
        text = "\n".join(str(part) for part in text)
    data = _parse_json_object(str(text))
    data = _coerce_record_values_to_strings(data)
    validated = records_model.model_validate(data)
    return _records_to_dicts(validated)


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _coerce_record_values_to_strings(data: dict) -> dict:
    """Coerce scalar/list record values to strings before Pydantic validation.

    Stage 2 intentionally stores extracted values as strings so Stage 3 can
    normalize numbers and units. Some LLMs output JSON numbers for fields such
    as sample_size; accepting that here keeps extraction robust without changing
    the public output contract.
    """

    records = data.get("records")
    if not isinstance(records, list):
        return data

    coerced_records = []
    for rec in records:
        if not isinstance(rec, dict):
            coerced_records.append(rec)
            continue
        row = {}
        for key, value in rec.items():
            if value is None or isinstance(value, str):
                row[key] = value
            elif isinstance(value, (list, dict)):
                row[key] = json.dumps(value, ensure_ascii=False)
            else:
                row[key] = str(value)
        coerced_records.append(row)

    data = dict(data)
    data["records"] = coerced_records
    return data


def _mock_extract_records(context_str: str, records_model) -> list[dict]:
    """Deterministic smoke-test extractor used only when EXTRACTOR_MOCK=1."""
    schema = records_model.model_json_schema()
    defs = schema.get("$defs", {})
    record_schema = defs.get("Record", {})
    field_names = list(record_schema.get("properties", {}).keys())

    record: dict[str, str | None] = {name: None for name in field_names}
    lower = context_str.lower()

    if "patient_group" in record:
        match = re.search(r"\b(group\s+[a-z])\b", context_str, re.I)
        record["patient_group"] = match.group(1) + " patients" if match else "patients"
    if "treatment_regimen" in record:
        if "surgery" in lower:
            record["treatment_regimen"] = "surgery"
        elif "chemotherapy" in lower:
            record["treatment_regimen"] = "chemotherapy"
    if "os" in record:
        match = re.search(r"(?:median\s+)?(?:overall survival|OS)\s+(?:was|=|:)?\s*([0-9]+(?:\.[0-9]+)?\s*months?)", context_str, re.I)
        record["os"] = match.group(1) if match else None

    return [record]


def _records_to_dicts(result) -> list[dict]:
    """Convert LangChain structured output into plain dict records."""
    if hasattr(result, "records"):
        records = result.records
    elif isinstance(result, dict):
        records = result.get("records", [])
    else:
        records = []

    rows = []
    for rec in records:
        if hasattr(rec, "model_dump"):
            rows.append(rec.model_dump())
        elif isinstance(rec, dict):
            rows.append(dict(rec))
    return rows


def _build_llm(model_name: str | None):
    """构建 LangChain ChatOpenAI 实例，支持 OpenAI 兼容协议（Kimi 等）。"""
    from langchain_openai import ChatOpenAI

    name = _resolve_model_name(model_name)
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("LLM_API_KEY", "")
    )
    temperature = float(os.environ.get(
        "EXTRACTOR_TEMPERATURE",
        os.environ.get("LLM_TEMPERATURE", "0.6" if "kimi" in name.lower() else "0"),
    ))

    kwargs: dict[str, Any] = {
        "model": name,
        "api_key": api_key,
        "temperature": temperature,
        "timeout": float(os.environ.get("EXTRACTOR_TIMEOUT", "90")),
        "max_tokens": int(os.environ.get("EXTRACTOR_MAX_TOKENS", "4000")),
    }
    if base_url:
        kwargs["base_url"] = base_url
    if "kimi" in name.lower():
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    return ChatOpenAI(**kwargs)


def _resolve_model_name(model_name: str | None) -> str:
    return (
        model_name
        or os.environ.get("EXTRACTOR_MODEL")
        or os.environ.get("LLM_MODEL", "gpt-4o")
    )
