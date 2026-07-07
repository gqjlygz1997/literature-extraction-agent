"""
简单测试脚本，验证 Labeling 模块的基本功能
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """测试所有模块是否可以正常导入"""
    print("Testing imports...")

    try:
        from src.agent.labeling_config import LabelingConfigGenerator, GenerateLabelingConfigSignature
        print("✓ labeling_config imported")
    except Exception as e:
        print(f"✗ labeling_config import failed: {e}")
        return False

    try:
        from src.agent.tools.vector_store import VectorStoreBuilder, chunk_to_langchain_doc
        print("✓ vector_store imported")
    except Exception as e:
        print(f"✗ vector_store import failed: {e}")
        return False

    try:
        from src.agent.tools.hybrid_retriever import (
            SectionFilter,
            SemanticRetriever,
            RegexRetriever,
            TableHeaderRetriever,
            RRFFusion
        )
        print("✓ hybrid_retriever imported")
    except Exception as e:
        print(f"✗ hybrid_retriever import failed: {e}")
        return False

    try:
        from src.agent.tools.dspy_evidence_labeler import EvidenceLabeler, EvidenceLabelSignature
        print("✓ dspy_evidence_labeler imported")
    except Exception as e:
        print(f"✗ dspy_evidence_labeler import failed: {e}")
        return False

    try:
        from src.agent.workflow import DomainExtractionWorkflow
        print("✓ workflow.run_labeling_mvp imported")
    except Exception as e:
        print(f"✗ workflow import failed: {e}")
        return False

    return True


def test_user_requirements_record_fields():
    """测试 user_requirements.yaml 新 record.fields 格式"""
    print("\nTesting user_requirements record.fields schema...")

    from src.agent.user_requirements import load_user_requirements

    req_path = Path(__file__).parent / "experiments" / "pancan" / "user_requirements.yaml"
    reqs = load_user_requirements(req_path)

    assert reqs.record is not None, "record block should be parsed"
    assert reqs.record.name == "treatment_outcome_record"
    assert reqs.target_fields == reqs.record.fields
    assert any(field.name == "treatment_regimen" for field in reqs.target_fields)
    assert any(field.name == "os" for field in reqs.target_fields)
    assert "different value" in reqs.record.meaning

    print("✓ user_requirements record.fields schema works correctly")
    return True


def test_section_filter():
    """测试 SectionFilter 的 exclude 硬过滤 + include 软偏好逻辑"""
    print("\nTesting SectionFilter...")

    from src.agent.tools.hybrid_retriever import SectionFilter

    # 测试数据
    chunks = [
        {"chunk_id": "c1", "section_path": ["Introduction"]},
        {"chunk_id": "c2", "section_path": ["Results", "Primary Outcomes"]},
        {"chunk_id": "c3", "section_path": ["References"]},
        {"chunk_id": "c4", "section_path": ["Acknowledgements"]},
        {"chunk_id": "c5", "section_path": ["Methods"]},
        {"chunk_id": "c6", "section_path": ["Supplementary Materials"]},
    ]

    # 测试：include=["results"], exclude 默认
    filter1 = SectionFilter(include=["results"], exclude=None)
    result1 = filter1.filter_chunks(chunks)

    print(f"  Excluded: {len(result1['excluded'])} chunks")
    print(f"  Preferred: {len(result1['preferred'])} chunks")
    print(f"  Fallback: {len(result1['fallback'])} chunks")
    print(f"  Allowed: {len(result1['allowed'])} chunks")

    # 验证
    assert len(result1['excluded']) == 2, "Should exclude References + Acknowledgements"
    assert len(result1['preferred']) == 1, "Should prefer Results"
    assert len(result1['fallback']) == 3, "Should fallback to Introduction + Methods + Supplementary"
    assert result1['excluded'][0]['chunk_id'] in ['c3', 'c4']
    assert result1['preferred'][0]['chunk_id'] == 'c2'

    print("✓ SectionFilter works correctly")
    return True


def test_rrf_fusion():
    """测试 RRF 融合逻辑（0-based rank）"""
    print("\nTesting RRFFusion...")

    from src.agent.tools.hybrid_retriever import RRFFusion

    # 模拟 semantic 和 regex 结果
    semantic_results = [
        {"chunk_id": "c1", "semantic_rank": 0, "paper_id": "p1", "chunk_type": "paragraph",
         "section_path_text": "Results", "source": "semantic"},
        {"chunk_id": "c2", "semantic_rank": 1, "paper_id": "p1", "chunk_type": "paragraph",
         "section_path_text": "Methods", "source": "semantic"},
    ]

    regex_results = [
        {"chunk_id": "c2", "regex_rank": 0, "paper_id": "p1", "chunk_type": "paragraph",
         "section_path_text": "Methods", "matched_patterns": ["pattern1"], "source": "regex"},
        {"chunk_id": "c3", "regex_rank": 1, "paper_id": "p1", "chunk_type": "paragraph",
         "section_path_text": "Results", "matched_patterns": ["pattern2"], "source": "regex"},
    ]

    fusion = RRFFusion(k=60)
    merged = fusion.merge([semantic_results, regex_results], top_k=3)

    print(f"  Merged {len(merged)} chunks")
    for item in merged:
        print(f"    {item['chunk_id']}: rrf_score={item['rrf_score']:.4f}, "
              f"sources={item['sources']}, rank={item['hybrid_rank']}")

    # 验证：c2 应该排第一（出现在两个列表中）
    assert merged[0]['chunk_id'] == 'c2', "c2 should rank first (appears in both lists)"
    assert len(merged[0]['sources']) == 2, "c2 should have 2 sources"

    # 验证 RRF 公式：1 / (k + rank + 1)
    # c1: semantic rank=0 → 1/(60+0+1) = 0.0164
    # c2: semantic rank=1 + regex rank=0 → 1/(60+1+1) + 1/(60+0+1) = 0.0161 + 0.0164 = 0.0325
    expected_c2_score = 1/(60+1+1) + 1/(60+0+1)
    assert abs(merged[0]['rrf_score'] - expected_c2_score) < 0.0001, f"RRF score mismatch: {merged[0]['rrf_score']} vs {expected_c2_score}"

    print("✓ RRFFusion works correctly")
    return True


def test_table_metadata_reading():
    """测试从 chunk["metadata"] 读取表格字段"""
    print("\nTesting table metadata reading...")

    from src.agent.tools.vector_store import build_table_labeling_text, build_table_retrieval_text

    table_chunk = {
        "chunk_id": "t1",
        "chunk_type": "table",
        "metadata": {
            "caption": "Overall Survival Results",
            "headers": ["Treatment", "Median OS (months)", "HR", "95% CI"],
            "markdown_text": "| Treatment | Median OS | HR | CI |\n|---|---|---|---|\n| A | 24.5 | 0.68 | 0.52-0.88 |",
            "footnotes": ["p < 0.001"]
        }
    }

    labeling_text = build_table_labeling_text(table_chunk)
    retrieval_text = build_table_retrieval_text(table_chunk)

    print(f"  Labeling text length: {len(labeling_text)} chars")
    print(f"  Retrieval text length: {len(retrieval_text)} chars")

    # 验证
    assert "Overall Survival Results" in labeling_text
    assert "markdown_text" in labeling_text.lower() or "table content" in labeling_text.lower()
    assert "p < 0.001" in labeling_text

    assert "Overall Survival Results" in retrieval_text
    assert "markdown" not in retrieval_text.lower() or len(retrieval_text) < len(labeling_text)

    print("✓ Table metadata reading works correctly")
    return True


def test_table_channel_few_tables():
    """
    构造 3 个 table chunks（<= table_top_k=5）：
    - table_1: header keyword 命中
    - table_2: regex 命中
    - table_3: 未命中任何检索

    期望：
    - 所有 3 个 table 都进入 candidates
    - table_1 / table_2 有检索信号（rrf_score > 0 或 matched_keywords / matched_patterns 非空）
    - table_3 以 source=table_all、rrf_score=0 追加在后
    - 最终 candidates 顺序：有信号的排在前面
    """
    print("\nTesting Table channel (few tables ≤ table_top_k)...")

    from src.agent.tools.hybrid_retriever import RRFFusion

    table_top_k = 5

    # 模拟 table header retrieval 结果（只命中 table_1）
    table_header_results = [
        {
            "chunk_id": "t1", "paper_id": "p1", "chunk_type": "table",
            "section_path_text": "Results",
            "table_rank": 0, "table_score": 2.0,
            "matched_keywords": ["caption:overall survival"],
            "source": "table_header"
        }
    ]

    # 模拟 regex retrieval 结果（只命中 table_2）
    regex_results = [
        {
            "chunk_id": "t2", "paper_id": "p1", "chunk_type": "table",
            "section_path_text": "Methods",
            "regex_rank": 0, "regex_score": 3.0,
            "matched_patterns": ["OS duration"],
            "source": "regex"
        }
    ]

    # table_3 未被任何检索命中

    # --- 执行 RRF 融合 ---
    rrf_fusion = RRFFusion(k=60)
    rrf_ranked = rrf_fusion.merge(
        ranked_lists=[table_header_results, regex_results],
        top_k=3   # len(table_chunks)=3，全部纳入
    )

    # 已经在 RRF 中的 chunk ids
    rrf_ids = {item["chunk_id"] for item in rrf_ranked}

    # 模拟 table_chunks 列表
    table_chunks = [
        {"chunk_id": "t1", "paper_id": "p1", "chunk_type": "table", "section_path": ["Results"]},
        {"chunk_id": "t2", "paper_id": "p1", "chunk_type": "table", "section_path": ["Methods"]},
        {"chunk_id": "t3", "paper_id": "p1", "chunk_type": "table", "section_path": ["Supplementary"]},
    ]

    # --- 执行补全逻辑（复制自 workflow._process_paper_field_labeling）---
    table_candidates = list(rrf_ranked)
    next_rank = len(table_candidates) + 1
    for chunk in table_chunks:
        if chunk["chunk_id"] not in rrf_ids:
            table_candidates.append({
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "chunk_type": "table",
                "section_path_text": " > ".join(chunk.get("section_path", [])),
                "rrf_score": 0.0,
                "hybrid_rank": next_rank,
                "sources": ["table_all"],
                "semantic_rank": None,
                "semantic_similarity": None,
                "regex_rank": None,
                "regex_score": None,
                "matched_patterns": [],
                "table_rank": None,
                "table_score": None,
                "matched_keywords": [],
            })
            next_rank += 1

    print(f"  Candidates: {len(table_candidates)}")
    for c in table_candidates:
        print(f"    {c['chunk_id']}: rrf_score={c['rrf_score']:.4f}, "
              f"sources={c['sources']}, keywords={c.get('matched_keywords', [])}, "
              f"patterns={c.get('matched_patterns', [])}")

    # 验证 1：所有 3 个 table 都在 candidates 里
    candidate_ids = {c["chunk_id"] for c in table_candidates}
    assert candidate_ids == {"t1", "t2", "t3"}, \
        f"Expected all 3 tables, got {candidate_ids}"

    # 验证 2：t1 有 table header 信号
    t1 = next(c for c in table_candidates if c["chunk_id"] == "t1")
    assert t1["rrf_score"] > 0, "t1 should have positive rrf_score"
    assert t1.get("matched_keywords"), "t1 should have matched_keywords"

    # 验证 3：t2 有 regex 信号
    t2 = next(c for c in table_candidates if c["chunk_id"] == "t2")
    assert t2["rrf_score"] > 0, "t2 should have positive rrf_score"
    assert t2.get("matched_patterns"), "t2 should have matched_patterns"

    # 验证 4：t3 是 fallback，rrf_score=0，source 包含 table_all
    t3 = next(c for c in table_candidates if c["chunk_id"] == "t3")
    assert t3["rrf_score"] == 0.0, "t3 fallback should have rrf_score=0"
    assert "table_all" in t3["sources"], "t3 fallback should have source=table_all"

    # 验证 5：有信号的排在前面（t1/t2 的 hybrid_rank < t3 的 hybrid_rank）
    t3_rank = t3["hybrid_rank"]
    for c in [t1, t2]:
        assert c["hybrid_rank"] < t3_rank, \
            f"{c['chunk_id']} (rank={c['hybrid_rank']}) should rank before t3 (rank={t3_rank})"

    print("✓ Table channel (few tables) works correctly")
    return True


def test_table_channel_many_tables():
    """
    构造 8 个 table chunks（> table_top_k=5）：
    - t1~t3: table header 命中
    - t2~t6: regex 命中（共 6 个不同 table 有信号）
    - t7~t8: 未命中任何检索

    期望：
    - 只取 RRF top-5（不是全部 8 个）
    - t7/t8（无信号）不在 candidates 里
    - t2/t3（两路都命中）排在最前
    """
    print("\nTesting Table channel (many tables > table_top_k)...")

    from src.agent.tools.hybrid_retriever import RRFFusion

    table_top_k = 5

    # 模拟 8 个 table chunks
    table_chunks = [
        {"chunk_id": f"t{i}", "paper_id": "p1", "chunk_type": "table",
         "section_path": ["Results"]}
        for i in range(1, 9)   # t1~t8
    ]

    # header 命中 t1, t2, t3
    table_header_results = [
        {"chunk_id": f"t{i}", "paper_id": "p1", "chunk_type": "table",
         "section_path_text": "Results",
         "table_rank": idx, "table_score": float(3 - idx),
         "matched_keywords": [f"caption:kw{i}"],
         "source": "table_header"}
        for idx, i in enumerate([1, 2, 3])
    ]

    # regex 命中 t2, t3, t4, t5, t6（5 个）
    regex_results = [
        {"chunk_id": f"t{i}", "paper_id": "p1", "chunk_type": "table",
         "section_path_text": "Results",
         "regex_rank": idx, "regex_score": float(5 - idx),
         "matched_patterns": [f"pat{i}"],
         "source": "regex"}
        for idx, i in enumerate([2, 3, 4, 5, 6])
    ]

    # 有信号的 table：t1(header), t2(both), t3(both), t4(regex), t5(regex), t6(regex) → 共 6 个
    # 无信号的：t7, t8

    # RRF 融合（top_k=len(table_chunks) 先拿全部）
    rrf_fusion = RRFFusion(k=60)
    rrf_ranked = rrf_fusion.merge(
        ranked_lists=[table_header_results, regex_results],
        top_k=len(table_chunks)
    )

    # 截断到 table_top_k
    table_candidates = rrf_ranked[:table_top_k]

    print(f"  Total table chunks: {len(table_chunks)}")
    print(f"  Tables with signals: 6 (t1~t6)")
    print(f"  RRF ranked: {len(rrf_ranked)}")
    print(f"  Candidates after top-{table_top_k} cut: {len(table_candidates)}")
    for c in table_candidates:
        print(f"    {c['chunk_id']}: rrf_score={c['rrf_score']:.4f}, sources={c['sources']}")

    # 验证 1：candidates 数量 == table_top_k（因为有信号的 >= 5）
    assert len(table_candidates) == table_top_k, \
        f"Expected {table_top_k} candidates, got {len(table_candidates)}"

    # 验证 2：t7/t8 不在 candidates（无信号）
    candidate_ids = {c["chunk_id"] for c in table_candidates}
    assert "t7" not in candidate_ids and "t8" not in candidate_ids, \
        f"t7/t8 (no signals) should not be in candidates, got {candidate_ids}"

    # 验证 3：t2/t3（两路命中）应该排在最前 2 位
    top2_ids = {c["chunk_id"] for c in table_candidates[:2]}
    assert "t2" in top2_ids and "t3" in top2_ids, \
        f"t2 and t3 (both lists) should be top-2, got {top2_ids}"

    print("✓ Table channel (many tables) works correctly")
    return True


class _FakeVectorBuilder:
    """测试用的假 vector_builder，提供 get_chunk(chunk_id)。"""

    def __init__(self, chunk_map):
        self._chunk_map = chunk_map

    def get_chunk(self, chunk_id):
        return self._chunk_map.get(chunk_id, {})


def _make_labeling_config():
    """构造一个最小 labeling_config，用于测试 labels 排序与 summary。"""
    return {
        "fields": [
            {"field_name": "treatment_regimen"},
            {"field_name": "sample_size"},
            {"field_name": "os"},
            {"field_name": "orr"},
        ],
        "embedding": {"model": "fake-embedding"},
        "labeler": {"model": "fake-labeler"},
    }


def test_aggregate_labeled_chunks():
    """
    验证 chunk 聚合逻辑：
    - 同一个 chunk 多个 field relevant=true → 只输出一行，labels 合并
    - relevant=false 的记录不会进入（调用方已过滤，这里只喂 relevant 记录）
    - labels 按 config field 顺序排序
    - 只输出指定的 6 个字段
    - 只输出 labels 非空的 chunk
    """
    print("\nTesting _aggregate_labeled_chunks...")

    from src.agent.workflow import DomainExtractionWorkflow

    config = _make_labeling_config()

    # chunk c1 命中 3 个 field（故意乱序喂入：os, treatment_regimen, sample_size）
    # chunk c2 命中 1 个 field
    raw_labeled = [
        {"paper_id": "P1", "field_name": "os", "chunk_id": "c1", "chunk_type": "abstract"},
        {"paper_id": "P1", "field_name": "treatment_regimen", "chunk_id": "c1", "chunk_type": "abstract"},
        {"paper_id": "P1", "field_name": "sample_size", "chunk_id": "c1", "chunk_type": "abstract"},
        # 重复的 field，验证去重
        {"paper_id": "P1", "field_name": "os", "chunk_id": "c1", "chunk_type": "abstract"},
        {"paper_id": "P1", "field_name": "sample_size", "chunk_id": "c2", "chunk_type": "paragraph"},
    ]

    fake_vb = _FakeVectorBuilder({
        "c1": {"section_path": ["Abstract", "Results:"], "metadata": {"chunk_index": 1}},
        "c2": {"section_path": ["Methods"], "metadata": {"chunk_index": 7}},
    })

    rows = DomainExtractionWorkflow._aggregate_labeled_chunks(raw_labeled, config, fake_vb)

    print(f"  Aggregated rows: {len(rows)}")
    for r in rows:
        print(f"    {r['chunk_id']} (idx={r['chunk_index']}): labels={r['labels']}")

    # 验证 1：一个 chunk 一行 → 2 行
    assert len(rows) == 2, f"Expected 2 chunk rows, got {len(rows)}"

    c1 = next(r for r in rows if r["chunk_id"] == "c1")
    c2 = next(r for r in rows if r["chunk_id"] == "c2")

    # 验证 2：c1 合并了 3 个 field（去重后）
    assert len(c1["labels"]) == 3, f"c1 should have 3 labels, got {c1['labels']}"

    # 验证 3：labels 按 config field 顺序排序
    assert c1["labels"] == ["treatment_regimen", "sample_size", "os"], \
        f"c1 labels not in config order: {c1['labels']}"

    # 验证 4：chunk_index 从 metadata.chunk_index 读取
    assert c1["chunk_index"] == 1, f"c1 chunk_index should be 1, got {c1['chunk_index']}"
    assert c2["chunk_index"] == 7, f"c2 chunk_index should be 7, got {c2['chunk_index']}"

    # 验证 5：section_path 从 parsed chunk 读取
    assert c1["section_path"] == ["Abstract", "Results:"]

    # 验证 6：每行只含指定的 6 个字段
    expected_keys = {"paper_id", "chunk_id", "chunk_index", "chunk_type", "section_path", "labels"}
    for r in rows:
        assert set(r.keys()) == expected_keys, \
            f"Unexpected keys in output row: {set(r.keys())}"

    print("✓ _aggregate_labeled_chunks works correctly")
    return True


def test_labeling_summary_new_shape():
    """
    验证新版 summary 统计：
    - total_labeled_chunks / total_label_assignments
    - by_field[field] = {chunks, papers}
    - by_paper[paper] = {labeled_chunks, label_assignments}
    """
    print("\nTesting _generate_labeling_summary (new shape)...")

    from src.agent.workflow import DomainExtractionWorkflow

    config = _make_labeling_config()

    main_rows = [
        {"paper_id": "P1", "chunk_id": "c1", "chunk_index": 1, "chunk_type": "abstract",
         "section_path": ["Abstract"], "labels": ["treatment_regimen", "sample_size", "os"]},
        {"paper_id": "P1", "chunk_id": "c2", "chunk_index": 7, "chunk_type": "paragraph",
         "section_path": ["Methods"], "labels": ["sample_size"]},
        {"paper_id": "P2", "chunk_id": "c9", "chunk_index": 0, "chunk_type": "abstract",
         "section_path": ["Abstract"], "labels": ["os"]},
    ]

    summary = DomainExtractionWorkflow._generate_labeling_summary(main_rows, config)

    print(f"  total_labeled_chunks: {summary['total_labeled_chunks']}")
    print(f"  total_label_assignments: {summary['total_label_assignments']}")
    print(f"  by_field: {summary['by_field']}")
    print(f"  by_paper: {summary['by_paper']}")

    # 验证顶层计数
    assert summary["total_labeled_chunks"] == 3
    assert summary["total_label_assignments"] == 3 + 1 + 1  # = 5

    # 验证 by_field
    assert summary["by_field"]["sample_size"]["chunks"] == 2
    assert summary["by_field"]["sample_size"]["papers"]["P1"] == 2
    assert summary["by_field"]["os"]["chunks"] == 2
    assert summary["by_field"]["os"]["papers"] == {"P1": 1, "P2": 1}
    assert summary["by_field"]["treatment_regimen"]["chunks"] == 1
    # orr 没有命中，不应出现在 by_field
    assert "orr" not in summary["by_field"]

    # 验证 by_paper
    assert summary["by_paper"]["P1"] == {"labeled_chunks": 2, "label_assignments": 4}
    assert summary["by_paper"]["P2"] == {"labeled_chunks": 1, "label_assignments": 1}

    # 验证元信息
    assert summary["config_fields"] == ["treatment_regimen", "sample_size", "os", "orr"]
    assert summary["embedding_model"] == "fake-embedding"
    assert summary["labeler_model"] == "fake-labeler"

    # 确认旧字段已移除
    for old_key in ("total_records", "total_relevant", "relevance_rate"):
        assert old_key not in summary, f"Old key {old_key} should be removed"

    print("✓ _generate_labeling_summary (new shape) works correctly")
    return True


def main():
    print("=" * 60)
    print("Labeling Stage - Basic Tests")
    print("=" * 60)

    all_pass = True

    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import test failed")
        all_pass = False

    # Test 2: user_requirements record.fields
    try:
        if not test_user_requirements_record_fields():
            all_pass = False
    except Exception as e:
        print(f"\n❌ user_requirements record.fields test failed: {e}")
        all_pass = False

    # Test 3: Section Filter
    try:
        if not test_section_filter():
            all_pass = False
    except Exception as e:
        print(f"\n❌ SectionFilter test failed: {e}")
        all_pass = False

    # Test 4: RRF Fusion
    try:
        if not test_rrf_fusion():
            all_pass = False
    except Exception as e:
        print(f"\n❌ RRFFusion test failed: {e}")
        all_pass = False

    # Test 5: Table Metadata
    try:
        if not test_table_metadata_reading():
            all_pass = False
    except Exception as e:
        print(f"\n❌ Table metadata test failed: {e}")
        all_pass = False

    # Test 6: Table Channel - Few Tables
    try:
        if not test_table_channel_few_tables():
            all_pass = False
    except Exception as e:
        print(f"\n❌ Table channel (few tables) test failed: {e}")
        all_pass = False

    # Test 7: Table Channel - Many Tables
    try:
        if not test_table_channel_many_tables():
            all_pass = False
    except Exception as e:
        print(f"\n❌ Table channel (many tables) test failed: {e}")
        all_pass = False

    # Test 8: Aggregate labeled chunks (new chunk-centric output)
    try:
        if not test_aggregate_labeled_chunks():
            all_pass = False
    except Exception as e:
        print(f"\n❌ Aggregate labeled chunks test failed: {e}")
        all_pass = False

    # Test 9: Labeling summary (new shape)
    try:
        if not test_labeling_summary_new_shape():
            all_pass = False
    except Exception as e:
        print(f"\n❌ Labeling summary (new shape) test failed: {e}")
        all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("✅ All basic tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
