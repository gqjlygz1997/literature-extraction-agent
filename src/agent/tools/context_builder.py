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


# 最大上下文字符数（约 3000 tokens，为 system + instruction 留余量）
MAX_CONTEXT_CHARS = 12_000


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

    # 合并候选
    candidate_ids = abstract_ids | labeled_ids

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

    # 渲染 + 截断策略：abstract 优先，不被截断
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

    # 第二轮：按顺序纳入 labeled chunks，遇到超限停止
    for chunk in candidates:
        cid = chunk["chunk_id"]
        if cid in abstract_ids:
            continue  # 已在第一轮处理

        text = _render_chunk(chunk)
        if total_chars + len(text) > MAX_CONTEXT_CHARS:
            break  # 超限，停止纳入后续 chunks

        rendered_texts.append(text)
        total_chars += len(text)
        used_ids.append(cid)

    context_str = "\n\n".join(rendered_texts)
    return context_str, used_ids


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
