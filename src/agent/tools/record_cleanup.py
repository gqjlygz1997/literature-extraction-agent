"""
record_cleanup.py — Record deduplication and ID assignment for Stage 2 Extraction.

对应 ALLMAT: entity_resolution_rule() + entity_resolution_utils.py 的简化版

设计原则（Stage 2 最小 cleanup，不做 Stage 3 Post-processing）：
- 只在同一 paper_id 内处理，不跨 paper 合并
- normalize：strip + 折叠空格 + 统一小写
- 严格规则：所有字段 normalize 后完全相同 → duplicate，只保留一条
- 任意字段不同 → 保留为独立 record
- 不做 fuzzy merge，不调 LLM，不做数值换算，不做领域标准化
- 每条 record 分配唯一 record_id，附加 source_chunk_ids
- 输出字段全量补齐（缺失字段为 null），保持 yaml 定义顺序

空值处理约定：
- null 和 "" normalize 后均为 ""，视为相同
- null vs 非空值 normalize 后不同 → 不合并（保留两条）
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

def _normalize(v: object) -> str:
    """
    规范化字段值为可比较字符串。

    对应 ALLMAT entity_resolution_utils.py 的 normalize_* 函数，
    但去除领域特定逻辑（无 composition / processing_kw 等），保持通用。

    规则：
      - None → ""
      - 其他 → str().strip()，折叠内部多余空格，统一小写
    """
    if v is None:
        return ""
    s = str(v).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(
    records: list[dict],
    field_names: list[str],
) -> tuple[list[dict], int]:
    """
    对同一 paper_id 内的 records 做严格规则去重。

    对应 ALLMAT: entity_resolution_rule() + partition_strict()

    参数：
        records:     LLM 抽取的 raw records（list of dict）
        field_names: record.fields 的字段名列表（按 yaml 顺序）

    返回：
        (deduped_records, n_removed)
        - deduped_records: 去重后的 records，保持原始字段值（不做修改）
        - n_removed: 删除的重复条数

    去重规则：
        key = tuple(_normalize(rec.get(f)) for f in field_names)
        key 相同 → duplicate，只保留第一次出现的那条
        key 不同 → 独立 record，保留

    注意：
        - null 和 "" 的 normalize 结果相同，视为相同
        - null vs 非空字符串 normalize 后不同，不合并
        - 这意味着「os=24.3 months」和「os=null」（其他字段相同）会保留两条
    """
    seen: dict[tuple, dict] = {}
    for rec in records:
        key = tuple(_normalize(rec.get(f)) for f in field_names)
        if key not in seen:
            seen[key] = rec

    deduped = list(seen.values())
    n_removed = len(records) - len(deduped)
    return deduped, n_removed


# ---------------------------------------------------------------------------
# Field complement（字段全量补齐）
# ---------------------------------------------------------------------------

def complement_fields(record: dict, field_names: list[str]) -> dict:
    """
    确保 record 包含所有 field_names 字段，缺失字段补 null。

    对 Stage 3 重要：输出的每条 record 字段集合必须一致，
    便于 CSV 导出和 ML 表格对齐。
    """
    return {f: record.get(f) for f in field_names}


# ---------------------------------------------------------------------------
# ID assignment + source attachment
# ---------------------------------------------------------------------------

def assign_ids_and_source(
    records: list[dict],
    paper_id: str,
    field_names: list[str],
    source_chunk_ids: list[str] | list[list[str]],
) -> list[dict]:
    """
    给去重后的 records 分配 record_id、补齐字段、附加 source_chunk_ids。

    输出格式（每条 record）：
        {
            "paper_id":        str,
            "record_id":       "{paper_id}::r{n:04d}",
            <field_1>:         value | null,
            ...
            <field_n>:         value | null,
            "source_chunk_ids": list[str],
        }

    字段顺序：paper_id → record_id → 所有 record.fields（yaml 顺序）→ source_chunk_ids
    """
    result = []
    for i, rec in enumerate(records, 1):
        row: dict = {
            "paper_id":  paper_id,
            "record_id": f"{paper_id}::r{i:04d}",
        }
        # record.fields 全量补齐，保持 yaml 顺序
        row.update(complement_fields(rec, field_names))
        row["source_chunk_ids"] = list(_source_ids_for_record(source_chunk_ids, i - 1))
        result.append(row)

    return result


def _source_ids_for_record(
    source_chunk_ids: list[str] | list[list[str]],
    index: int,
) -> list[str]:
    if source_chunk_ids and isinstance(source_chunk_ids[0], list):
        if index < len(source_chunk_ids):
            return source_chunk_ids[index]
        return []
    return source_chunk_ids
