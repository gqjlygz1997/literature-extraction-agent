"""
test_extraction_basic.py — Unit tests for Stage 2 Extraction.

测试范围（不调 LLM）：
- build_records_model: 动态生成 Pydantic 模型
- create_instruction: 动态生成 instruction
- build_system_message: 动态生成 system message
- context builder: 上下文拼接顺序、截断策略
- record cleanup: normalize、deduplicate、字段补齐、ID 分配
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_build_records_model():
    """验证动态生成的 Pydantic 模型结构正确。"""
    print("\n测试：build_records_model")
    from src.agent.extraction_schema import build_records_model

    fields = [
        {"name": "patient_group", "definition": "Patient group", "type": "string"},
        {"name": "sample_size", "definition": "Sample size", "type": "number"},
        {"name": "os", "definition": "Overall survival", "type": "string"},
    ]

    Records = build_records_model(fields)

    # 验证：model 名称
    assert Records.__name__ == "Records"

    # 验证：有 records 字段
    schema = Records.model_json_schema()
    assert "properties" in schema
    assert "records" in schema["properties"]

    # Pydantic v2 中 Record 定义在 $defs["Record"]，items 只有 $ref
    assert "$defs" in schema, "Pydantic v2 应把 Record 放在 $defs"
    record_schema = schema["$defs"]["Record"]

    # 验证：Record 含指定字段
    assert "patient_group" in record_schema["properties"]
    assert "sample_size" in record_schema["properties"]
    assert "os" in record_schema["properties"]

    # 验证：extra="forbid" 有效（additionalProperties: false）
    assert record_schema.get("additionalProperties") is False

    print("  ✓ Pydantic 模型结构正确")
    return True


def test_create_instruction():
    """验证 instruction 不含硬编码领域词汇。"""
    print("\n测试：create_instruction")
    from src.agent.extraction_schema import create_instruction

    class FakeRecord:
        name = "test_record"
        fields = [
            type("F", (), {"name": "f1", "definition": "Field one"}),
            type("F", (), {"name": "f2", "definition": "Field two"}),
        ]

    instruction = create_instruction(FakeRecord())

    # 验证：包含 record name 和 field names
    assert "test_record" in instruction
    assert "f1" in instruction
    assert "f2" in instruction
    assert "Field one" in instruction

    # 验证：不含硬编码领域词汇
    for term in ["clinical", "patient", "treatment", "trial"]:
        assert term.lower() not in instruction.lower(), f"不应出现硬编码词汇：{term}"

    print("  ✓ instruction 不含硬编码领域词汇")
    return True


def test_build_system_message():
    """验证 system message 不含硬编码领域词汇。"""
    print("\n测试：build_system_message")
    from src.agent.extraction_schema import build_system_message

    class FakeRecord:
        name = "material_record"
        meaning = "A material record represents one unique alloy composition."
        fields = []

    sys_msg = build_system_message(FakeRecord())

    # 验证：包含 record name 和 meaning
    assert "material_record" in sys_msg
    assert "unique alloy composition" in sys_msg

    # 验证：不含硬编码领域词汇
    for term in ["clinical", "patient", "treatment"]:
        assert term.lower() not in sys_msg.lower(), f"不应出现硬编码词汇：{term}"

    print("  ✓ system message 不含硬编码领域词汇")
    return True


def test_context_order():
    """验证上下文按 chunk_index 排序，abstract 优先纳入。"""
    print("\n测试：context builder 排序")
    from src.agent.tools.context_builder import build_context

    # 构造假 chunks（故意乱序）
    chunk_store = {
        "c1": {
            "paper_id": "P1", "chunk_id": "c1", "chunk_type": "paragraph",
            "text": "Para at index 5", "section_path": ["Results"],
            "metadata": {"chunk_index": 5}
        },
        "c2": {
            "paper_id": "P1", "chunk_id": "c2", "chunk_type": "abstract",
            "text": "Abstract at index 0", "section_path": ["Abstract"],
            "metadata": {"chunk_index": 0}
        },
        "c3": {
            "paper_id": "P1", "chunk_id": "c3", "chunk_type": "paragraph",
            "text": "Para at index 3", "section_path": ["Methods"],
            "metadata": {"chunk_index": 3}
        },
    }

    labeled_ids = {"c1", "c3"}  # 只有 c1/c3 是 labeled

    context_str, used_ids = build_context("P1", chunk_store, labeled_ids)

    # 验证：abstract(c2) 被自动补充
    assert "c2" in used_ids

    # 验证：顺序为 c2(index=0) → c3(index=3) → c1(index=5)
    assert used_ids == ["c2", "c3", "c1"]

    # 验证：context 中 abstract 文本在最前
    assert context_str.index("Abstract at index 0") < context_str.index("Para at index 3")

    print("  ✓ 上下文按 chunk_index 排序，abstract 优先")
    return True


def test_context_table_render():
    """验证 table chunk 渲染为 Caption + markdown_text。"""
    print("\n测试：table chunk 渲染")
    from src.agent.tools.context_builder import build_context

    chunk_store = {
        "t1": {
            "paper_id": "P1", "chunk_id": "t1", "chunk_type": "table",
            "section_path": ["Results"], "metadata": {
                "chunk_index": 1,
                "caption": "Table 1. Survival data",
                "markdown_text": "| Group | OS |\n|---|---|\n| A | 24.5 |"
            }
        }
    }

    context_str, _ = build_context("P1", chunk_store, {"t1"})

    # 验证：包含 caption 和 markdown_text
    assert "Table 1. Survival data" in context_str
    assert "| Group | OS |" in context_str

    print("  ✓ table chunk 正确渲染")
    return True


def test_normalize_value():
    """验证 normalize 规则。"""
    print("\n测试：normalize_value")
    from src.agent.tools.record_cleanup import _normalize

    # None → ""
    assert _normalize(None) == ""

    # 折叠空格 + 统一小写
    assert _normalize("Stage  I–III  ") == "stage i–iii"

    # 数值转字符串
    assert _normalize(24.5) == "24.5"

    print("  ✓ normalize 规则正确")
    return True


def test_context_max_length():
    """验证超出最大长度时，abstract 不被截断，labeled 按顺序截断。"""
    print("\n测试：context 长度截断")
    from src.agent.tools import context_builder as cb

    # 临时缩小 MAX_CONTEXT_CHARS 为 200，方便测试截断
    original = cb.MAX_CONTEXT_CHARS
    cb.MAX_CONTEXT_CHARS = 200
    try:
        chunk_store = {
            "a1": {
                "paper_id": "P1", "chunk_id": "a1", "chunk_type": "abstract",
                "text": "Short abstract.",
                "section_path": ["Abstract"], "metadata": {"chunk_index": 0}
            },
            "p1": {
                "paper_id": "P1", "chunk_id": "p1", "chunk_type": "paragraph",
                "text": "A" * 100,
                "section_path": ["Methods"], "metadata": {"chunk_index": 1}
            },
            "p2": {
                "paper_id": "P1", "chunk_id": "p2", "chunk_type": "paragraph",
                "text": "B" * 100,
                "section_path": ["Results"], "metadata": {"chunk_index": 2}
            },
        }
        _, used = build_context("P1", chunk_store, {"p1", "p2"})

        # abstract 必须在
        assert "a1" in used, "abstract 不应被截断"
        # p1 / p2 至多保留一个（总长受限）
        assert len(used) < 3, "超长时应截断"
    finally:
        cb.MAX_CONTEXT_CHARS = original

    print("  ✓ 截断时 abstract 优先，labeled 按顺序")
    return True


def _make_config():
    return {
        "fields": [
            {"name": "patient_group", "definition": "Group", "type": "string"},
            {"name": "treatment_regimen", "definition": "Treatment", "type": "string"},
            {"name": "os", "definition": "Overall survival", "type": "string"},
        ]
    }


def test_deduplicate_exact_same():
    """两条所有字段相同 → 去重后只剩一条。"""
    print("\n测试：deduplicate — 完全相同记录")
    from src.agent.tools.record_cleanup import deduplicate

    config = _make_config()
    field_names = [f["name"] for f in config["fields"]]

    records = [
        {"patient_group": "Group A", "treatment_regimen": "Drug X", "os": "12 months"},
        {"patient_group": "Group A", "treatment_regimen": "Drug X", "os": "12 months"},
    ]
    deduped, n = deduplicate(records, field_names)

    assert len(deduped) == 1, f"应去重至 1 条，得到 {len(deduped)}"
    assert n == 1

    print("  ✓ 完全相同记录正确去重")
    return True


def test_deduplicate_one_field_diff():
    """两条某字段不同 → 保留两条。"""
    print("\n测试：deduplicate — 一个字段不同")
    from src.agent.tools.record_cleanup import deduplicate

    config = _make_config()
    field_names = [f["name"] for f in config["fields"]]

    records = [
        {"patient_group": "Group A", "treatment_regimen": "Drug X", "os": "12 months"},
        {"patient_group": "Group B", "treatment_regimen": "Drug X", "os": "12 months"},
    ]
    deduped, n = deduplicate(records, field_names)

    assert len(deduped) == 2
    assert n == 0

    print("  ✓ 字段不同时保留两条")
    return True


def test_deduplicate_null_vs_value():
    """os=null vs os='24.3 months'（其他字段相同）→ 保留两条。"""
    print("\n测试：deduplicate — null vs 非空值")
    from src.agent.tools.record_cleanup import deduplicate

    config = _make_config()
    field_names = [f["name"] for f in config["fields"]]

    records = [
        {"patient_group": "Group A", "treatment_regimen": "Drug X", "os": "24.3 months"},
        {"patient_group": "Group A", "treatment_regimen": "Drug X", "os": None},
    ]
    deduped, n = deduplicate(records, field_names)

    assert len(deduped) == 2, "null vs 非空值不应合并"
    assert n == 0

    print("  ✓ null vs 非空值保留两条")
    return True


def test_assign_ids():
    """验证 record_id 格式和字段全量补齐。"""
    print("\n测试：assign_ids_and_source")
    from src.agent.tools.record_cleanup import assign_ids_and_source

    config = _make_config()
    field_names = [f["name"] for f in config["fields"]]

    records = [
        {"patient_group": "Group A", "os": "12 months"},  # 缺 treatment_regimen
    ]
    cleaned = assign_ids_and_source(records, "PMC123", field_names, ["c1", "c2"])

    assert len(cleaned) == 1
    row = cleaned[0]

    # record_id 格式
    assert row["record_id"] == "PMC123::r0001"
    assert row["paper_id"] == "PMC123"

    # 字段全量补齐（缺失字段为 null）
    assert row["treatment_regimen"] is None

    # 字段顺序：paper_id → record_id → fields → source_chunk_ids
    keys = list(row.keys())
    assert keys[0] == "paper_id"
    assert keys[1] == "record_id"
    assert keys[-1] == "source_chunk_ids"
    assert set(keys) == {"paper_id", "record_id", "patient_group",
                         "treatment_regimen", "os", "source_chunk_ids"}

    # source_chunk_ids 正确附加
    assert row["source_chunk_ids"] == ["c1", "c2"]

    print("  ✓ record_id 格式和字段全量补齐正确")
    return True


# ── 上文需要 build_context import ──────────────────────────────────────

def build_context(*args, **kwargs):
    from src.agent.tools.context_builder import build_context as _bc
    return _bc(*args, **kwargs)


def main():
    print("=" * 60)
    print("Extraction Stage — Basic Tests")
    print("=" * 60)

    all_pass = True
    tests = [
        test_build_records_model,
        test_create_instruction,
        test_build_system_message,
        test_context_order,
        test_context_table_render,
        test_context_max_length,
        test_normalize_value,
        test_deduplicate_exact_same,
        test_deduplicate_one_field_diff,
        test_deduplicate_null_vs_value,
        test_assign_ids,
    ]

    for test_fn in tests:
        try:
            if not test_fn():
                all_pass = False
        except Exception as e:
            print(f"\n❌ {test_fn.__name__} failed: {e}")
            import traceback; traceback.print_exc()
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print(f"✅ All {len(tests)} tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
