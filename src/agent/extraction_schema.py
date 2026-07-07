"""
extraction_schema.py — Dynamic Pydantic schema + prompt generation for Stage 2 Extraction.

对应 ALLMAT: create_result_model_dynamic() in extract_lc.py
             create_instruction_dynamic() in prompt.py

设计原则：
- schema 和 instruction 全部由 user_requirements.yaml 的 record 块动态生成
- 不硬编码任何领域词汇（clinical / patient / treatment 等），确保跨领域可用
- 字段类型：第一版全部 Optional[str]（包括 type: number 的字段）
  原因：Stage 2 保留 LLM 的原始表达；数值规范化在 Stage 3 Post-processing 处理
"""

from __future__ import annotations

from typing import Optional, List

from pydantic import BaseModel, ConfigDict, Field, create_model


# ---------------------------------------------------------------------------
# 类型映射（第一版全部 Optional[str]，Stage 3 再做数值规范化）
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[str, type] = {
    "string": Optional[str],
    "number": Optional[str],   # Stage 2 保留原始字符串，不做数值解析
    "list":   Optional[str],
}


def build_records_model(fields: list[dict]) -> type[BaseModel]:
    """
    根据 record.fields 动态生成 Records 顶层 Pydantic 模型。

    对应 ALLMAT: create_result_model_dynamic()

    用法：
        Records = build_records_model(req.record.fields)
        schema = Records.model_json_schema()

    参数：
        fields: 来自 user_requirements.yaml 的 record.fields，
                每项为 {'name': str, 'definition': str, 'type': str}
                也接受 FieldSpec dataclass 对象（有 .name / .definition / .type 属性）

    返回：
        Records(BaseModel)，顶层模型，含 records: List[Record]

    注意 model_config 的写法：
        用 create_model(__base__=BaseModel) 再单独赋值 model_config，
        而不是把 model_config 传给 create_model() 的 **kwargs，
        否则 Pydantic v2 中 model_config 会被当作普通字段。
        对应 ALLMAT 的：
            Record = create_model('Record', __base__=BaseModel, **fields)
            (ALLMAT 版 ConfigDict 通过 __base__ 继承，我们通过赋值保持兼容)
    """
    field_defs: dict[str, tuple] = {}
    for f in fields:
        # 兼容 FieldSpec dataclass 和普通 dict
        if hasattr(f, "name"):
            name, definition, ftype = f.name, f.definition, getattr(f, "type", "string")
        else:
            name, definition, ftype = f["name"], f["definition"], f.get("type", "string")

        python_type = _TYPE_MAP.get(ftype, Optional[str])
        field_defs[name] = (
            python_type,
            Field(default=None, description=definition),
        )

    # 动态生成 Record 模型
    Record = create_model("Record", __base__=BaseModel, **field_defs)
    Record.model_config = ConfigDict(extra="forbid")

    # 包装为 Records 顶层模型（对应 ALLMAT 的 Records(records=List[Record])）
    Records = create_model(
        "Records",
        __base__=BaseModel,
        records=(
            List[Record],
            Field(..., description="List of extracted records"),
        ),
    )
    Records.model_config = ConfigDict(extra="forbid")

    return Records


def create_instruction(record: object) -> str:
    """
    根据 record.meaning 和 record.fields 动态生成 extraction instruction。

    对应 ALLMAT: create_instruction_dynamic() in prompt.py

    参数：
        record: UserRequirements.record（RecordSpec dataclass），
                含 name / meaning / fields 属性

    返回：
        instruction 字符串，放入 user message 的 Instruction 部分

    注意：不含任何领域硬编码词汇，全部由 user_requirements 驱动。
    """
    # 字段列表描述
    field_lines = []
    for f in record.fields:
        if hasattr(f, "name"):
            name, definition = f.name, f.definition
        else:
            name, definition = f["name"], f["definition"]
        field_lines.append(f"  - **{name}**: {definition}")

    field_block = "\n".join(field_lines)

    instruction = f"""\
Extract all {record.name} records from the evidence context above.

Each record must contain the following fields:

{field_block}

Additional rules:
  - If a field value is not explicitly stated for a specific record, output null.
  - Create separate records when any of the above fields has a different value.
  - Do not merge records with different field values.
  - Do not infer or hallucinate values not present in the text.
  - Preserve the original wording of field values; do not normalize or convert units."""

    # ── 扩展位置：未来在此注入 DetectProcesses 产出的模板字符串（对应 ALLMAT synthesis.py） ──
    # 本版（Stage 2 第一版）暂不实现模板注入，这是当前有意偏离 ALLMAT 的地方之一。
    # 当 DetectProcesses 实现后，在此追加：
    #   instruction += "\n\n" + synthesis_prompt_str

    return instruction


def build_system_message(record: object) -> str:
    """
    根据 record.name / record.meaning 动态生成 system message。

    对应 ALLMAT: SYSTEM_MESSAGE_NO_SYN / SYSTEM_MESSAGE_SYN in prompt.py

    不含任何领域硬编码词汇。
    """
    record_name = record.name if hasattr(record, "name") else record["name"]
    record_meaning = record.meaning if hasattr(record, "meaning") else record["meaning"]

    return f"""\
You are a precise scientific data extraction assistant.

Your task is to extract structured records from scientific paper excerpts.

A {record_name} is defined as:
{record_meaning.strip()}

Rules:
- Create one record per distinct {record_name}.
- If any field value differs between two records, keep them as separate records.
- Only extract information explicitly stated in the text.
- If a field is not mentioned for a specific record, output null for that field.
- Do not infer, estimate, or hallucinate any values.
- Do not merge records unless ALL field values are identical."""
