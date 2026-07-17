"""
context_builder.py — Build contextualized evidence for extraction.

对应 ALLMAT: reorder_paras() + ParagraphExtend.from_paragraphs()

设计原则：
- 收集 labeled chunks + abstract chunks
- 按 chunk_index 排序（对应 reorder_paras 的原文顺序）
- abstract chunks 必须优先纳入，不应被截断
- 渲染每个 chunk 为带 header 的文本块
- 控制最大上下文长度
"""

from __future__ import annotations

import re
from typing import Any


# 最大上下文字符数（约 3000 tokens，为 system + instruction 留余量）
MAX_CONTEXT_CHARS = 12_000
MAX_SOURCE_CHUNKS = 4
MAX_NUMERIC_EVIDENCE_CHUNKS = 8

NUMERIC_ENDPOINT_RE = re.compile(
    r"\b("
    r"IC\s*50|IC50|IC\s*90|IC90|GI\s*50|GI50|EC\s*50|EC50|"
    r"Cmax|Tmax|AUC|half[-\s]?life|clearance|"
    r"tumou?r\s+(?:volume|weight|growth)|growth inhibition|TGI|"
    r"cell viability|cell proliferation|cytotoxicity|"
    r"colony formation|migration|invasion|apoptosis|"
    r"overall survival|progression[-\s]?free survival|OS|PFS|ORR|DCR"
    r")\b",
    re.IGNORECASE,
)

NUMERIC_VALUE_WITH_UNIT_RE = re.compile(
    r"(?:[<>~=≈≤≥]?\s*)?"
    r"\d[\d,]*(?:\.\d+)?"
    r"(?:\s*(?:-|–|to)\s*\d[\d,]*(?:\.\d+)?)?"
    r"(?:\s*(?:±|\+/-)\s*\d[\d,]*(?:\.\d+)?)?"
    r"\s*"
    r"(?:%|nM|pM|uM|µM|μM|mM|"
    r"mg/kg|mg|g|mm3|mm³|cm3|cm³|"
    r"%\s*ID/g|%ID/g|% ID|"
    r"months?|days?|hours?|h|"
    r"ng/mL|µg/g|μg/g|µg/mL|μg/mL|"
    r"kBq(?:/mL)?)",
    re.IGNORECASE,
)

FIELD_SOURCE_WEIGHTS = {
    "value": 8.0,
    "statistics": 4.0,
    "endpoint": 3.0,
    "compound_or_treatment": 3.0,
    "comparator_or_control": 2.5,
    "dose": 2.5,
    "duration": 2.5,
    "model_or_population": 2.0,
    "assay_or_study_type": 1.5,
    "sample_size": 1.5,
    "unit": 1.0,
    "route": 1.0,
}

SOURCE_STOP_WORDS = {
    "and",
    "the",
    "with",
    "without",
    "group",
    "groups",
    "patient",
    "patients",
    "cell",
    "cells",
    "mouse",
    "mice",
    "model",
    "models",
    "study",
    "therapy",
    "treatment",
    "cancer",
    "pdac",
    "pancreatic",
}


def build_context(
    paper_id: str,
    chunk_store: dict[str, dict],
    labeled_chunk_ids: set[str],
) -> tuple[str, list[str]]:
    """
    为指定 paper 构建 contextualized evidence 字符串。

    对应 ALLMAT: reorder_paras() + ParagraphExtend.from_paragraphs()

    参数：
        paper_id: 论文 ID
        chunk_store: {chunk_id → chunk_dict}，全局 parsed chunks
        labeled_chunk_ids: 该 paper 的 labeled chunk ids set

    返回：
        (context_str, used_chunk_ids)
        - context_str: 拼接好的上下文字符串，可直接放入 user message
        - used_chunk_ids: 实际纳入上下文的 chunk ids（用于记录来源）

    截断策略（对应用户反馈修正）：
        1. abstract chunks 必须优先纳入（全部保留，不被截断）
        2. 剩余 labeled chunks 按 chunk_index 排序
        3. 整体按 chunk_index 稳定排序（保持原文顺序）
        4. 若总长度超限，优先保留 abstract，再按顺序保留 labeled
    """
    # 收集候选 chunks（labeled + abstract）
    abstract_ids = set()
    labeled_ids = set()

    for cid, chunk in chunk_store.items():
        if chunk.get("paper_id") != paper_id:
            continue
        if chunk.get("chunk_type") == "abstract":
            abstract_ids.add(cid)
        if cid in labeled_chunk_ids:
            labeled_ids.add(cid)

    # Add a narrow safety net for concrete numeric endpoint evidence. Retrieval
    # can miss short result paragraphs that carry the exact values (for example
    # "IC50 values ranged from 3 to 10 uM"), while post-processing correctly
    # requires a numeric value. These chunks get priority in the extraction
    # context without loosening the final numeric validation.
    numeric_evidence_ids = _select_numeric_evidence_ids(paper_id, chunk_store)

    # 合并候选
    candidate_ids = abstract_ids | labeled_ids | numeric_evidence_ids

    if not candidate_ids:
        return "", []

    # 按 chunk_index 排序（对应 ALLMAT reorder_paras）
    candidates = [chunk_store[cid] for cid in candidate_ids if cid in chunk_store]
    candidates.sort(
        key=lambda c: (
            c.get("metadata", {}).get("chunk_index", float("inf")),
            c["chunk_id"],  # chunk_index 相同时的稳定排序
        )
    )

    # 渲染 + 截断策略：abstract 优先，不被截断；随后优先纳入明确数值证据，
    # 再按原文顺序补充 labeled chunks。
    rendered_texts = []
    total_chars = 0
    used_ids = []

    # 第一轮：渲染所有 abstract chunks（必须全部纳入）
    for chunk in candidates:
        if chunk["chunk_id"] in abstract_ids:
            text = _render_chunk(chunk)
            rendered_texts.append(text)
            total_chars += len(text)
            used_ids.append(chunk["chunk_id"])

    # 第二轮：优先纳入明确数值证据 chunks
    total_chars = _append_chunks_with_budget(
        candidates,
        numeric_evidence_ids - abstract_ids,
        rendered_texts,
        used_ids,
        total_chars,
    )

    # 第三轮：按顺序纳入剩余 labeled chunks，遇到超限停止
    _append_chunks_with_budget(
        candidates,
        labeled_ids - set(used_ids),
        rendered_texts,
        used_ids,
        total_chars,
    )

    context_str = "\n\n".join(rendered_texts)
    return context_str, used_ids


def build_context_from_chunk_ids(
    chunk_store: dict[str, dict],
    chunk_ids: list[str],
) -> str:
    """Render a context from an explicit ordered list of chunk ids."""

    rendered = [
        _render_chunk(chunk_store[cid])
        for cid in chunk_ids
        if cid in chunk_store
    ]
    return "\n\n".join(rendered)


def _append_chunks_with_budget(
    candidates: list[dict],
    include_ids: set[str],
    rendered_texts: list[str],
    used_ids: list[str],
    total_chars: int,
) -> int:
    used_set = set(used_ids)
    for chunk in candidates:
        cid = chunk["chunk_id"]
        if cid not in include_ids or cid in used_set:
            continue

        text = _render_chunk(chunk)
        if total_chars + len(text) > MAX_CONTEXT_CHARS:
            break

        rendered_texts.append(text)
        total_chars += len(text)
        used_ids.append(cid)
        used_set.add(cid)
    return total_chars


def _select_numeric_evidence_ids(
    paper_id: str,
    chunk_store: dict[str, dict],
    *,
    max_chunks: int = MAX_NUMERIC_EVIDENCE_CHUNKS,
) -> set[str]:
    matches: list[tuple[float, int, str]] = []
    for cid, chunk in chunk_store.items():
        if chunk.get("paper_id") != paper_id:
            continue
        score = _numeric_endpoint_evidence_score(chunk)
        if score <= 0:
            continue
        matches.append((
            -score,
            chunk.get("metadata", {}).get("chunk_index", float("inf")),
            cid,
        ))
    matches.sort()
    return {cid for _, _, cid in matches[:max_chunks]}


def _looks_like_numeric_endpoint_evidence(chunk: dict) -> bool:
    return _numeric_endpoint_evidence_score(chunk) > 0


def _numeric_endpoint_evidence_score(chunk: dict) -> float:
    text = _chunk_raw_text(chunk)
    if not NUMERIC_ENDPOINT_RE.search(text) or not NUMERIC_VALUE_WITH_UNIT_RE.search(text):
        return 0.0

    normalized = _normalize_search_text(text)
    section = " ".join(chunk.get("section_path") or []).lower()
    score = 1.0

    if chunk.get("chunk_type") == "table":
        score += 4.0
    if "result" in section:
        score += 5.0
    if any(term in section for term in ("method", "materials", "introduction", "discussion")):
        score -= 4.0

    if re.search(r"\b(?:IC\s*50|IC50|IC\s*90|IC90|GI\s*50|GI50|EC\s*50|EC50)\b", text, re.IGNORECASE):
        score += 4.0
    if any(term in normalized for term in (" values ", " ranged ", " ranging ", " were ", " respectively", "table")):
        score += 1.5

    unit_hits = len(NUMERIC_VALUE_WITH_UNIT_RE.findall(text))
    score += min(unit_hits, 4) * 0.75
    return score


def select_record_source_chunk_ids(
    record: dict[str, Any],
    chunk_store: dict[str, dict],
    used_chunk_ids: list[str],
    *,
    max_chunks: int = MAX_SOURCE_CHUNKS,
) -> list[str]:
    """Pick the context chunks most likely to support one extracted record.

    Extraction sees a paper-level context package, so ``used_chunk_ids`` can be
    long. This best-effort attribution narrows each record to chunks containing
    its concrete values and field terms while preserving original document order.
    """

    used_ids = [cid for cid in used_chunk_ids if cid in chunk_store]
    if not used_ids:
        return []

    scored: list[tuple[float, int, str]] = []
    for order, cid in enumerate(used_ids):
        score = _score_record_chunk(record, chunk_store[cid])
        if score > 0:
            scored.append((score, order, cid))

    if not scored:
        return used_ids[:max_chunks]

    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = {cid for _, _, cid in scored[:max_chunks]}
    return [cid for cid in used_ids if cid in chosen]


def _render_chunk(chunk: dict) -> str:
    """
    渲染单个 chunk 为带 header 的文本块。

    对应 ALLMAT: ParagraphExtend.from_paragraphs() 的序列化逻辑

    格式：
        [CHUNK {chunk_id} | {chunk_type} | {section_path}]
        {body}
    """
    chunk_id = chunk["chunk_id"]
    chunk_type = chunk.get("chunk_type", "unknown")
    section_path = chunk.get("section_path", [])
    section = " > ".join(section_path) if section_path else "(unknown)"

    header = f"[CHUNK {chunk_id} | {chunk_type} | {section}]"

    # 渲染 body
    if chunk_type == "table":
        body = _render_table_chunk(chunk)
    else:
        # paragraph / abstract
        body = chunk.get("text", "")

    return f"{header}\n{body}"


def _render_table_chunk(chunk: dict) -> str:
    """
    渲染 table chunk：Caption + markdown_text。

    对应 ALLMAT 对 table 的处理（caption 作为检索信号，markdown_text 作为完整内容）。
    """
    metadata = chunk.get("metadata", {})
    caption = metadata.get("caption", "")
    markdown_text = metadata.get("markdown_text", "")

    if markdown_text:
        return f"Caption: {caption}\n{markdown_text}"
    else:
        # fallback：只有 caption
        return f"Caption: {caption}"


def _score_record_chunk(record: dict[str, Any], chunk: dict) -> float:
    text = _chunk_search_text(chunk)
    score = 0.0
    for field, value in record.items():
        if field in {"paper_id", "record_id", "source_chunk_ids"}:
            continue
        weight = FIELD_SOURCE_WEIGHTS.get(field, 1.0)
        score += _score_value_against_text(value, text, weight)
    return score


def _score_value_against_text(value: Any, text: str, weight: float) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (list, tuple, set)):
        return sum(_score_value_against_text(v, text, weight) for v in value)
    if isinstance(value, dict):
        return sum(_score_value_against_text(v, text, weight) for v in value.values())

    raw = str(value).strip()
    normalized = _normalize_search_text(raw)
    if not normalized or normalized in {"none", "null", "not stated", "not reported"}:
        return 0.0

    score = 0.0
    if _phrase_in_text(normalized, text):
        score += weight

    numbers = _extract_search_numbers(normalized)
    for number in numbers:
        if _number_in_text(number, text):
            score += weight * 0.7

    tokens = [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9+\-/]{2,}", normalized)
        if token not in SOURCE_STOP_WORDS and not token.isdigit()
    ]
    token_hits = sum(1 for token in set(tokens) if token in text)
    if token_hits:
        score += min(weight * 0.5, token_hits * 0.25)

    return score


def _chunk_search_text(chunk: dict) -> str:
    return _normalize_search_text(_chunk_raw_text(chunk))


def _chunk_raw_text(chunk: dict) -> str:
    metadata = chunk.get("metadata") or {}
    parts: list[str] = [
        chunk.get("chunk_id", ""),
        chunk.get("chunk_type", ""),
        " ".join(chunk.get("section_path") or []),
        chunk.get("text", ""),
        metadata.get("caption", ""),
        metadata.get("markdown_text", ""),
    ]
    headers = metadata.get("headers")
    if isinstance(headers, list):
        parts.extend(str(header) for header in headers)
    return " ".join(str(part or "") for part in parts)


def _normalize_search_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_search_numbers(text: str) -> list[str]:
    return [
        match.replace(",", "")
        for match in re.findall(r"(?<![a-z])\d+(?:\.\d+)?(?:e[+-]?\d+)?", text)
    ]


def _number_in_text(number: str, text: str) -> bool:
    return re.search(rf"(?<![\w.]){re.escape(number)}(?![\w.])", text) is not None


def _phrase_in_text(phrase: str, text: str) -> bool:
    if len(phrase) <= 3 and re.fullmatch(r"[a-z0-9]+", phrase):
        return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None
    return phrase in text
