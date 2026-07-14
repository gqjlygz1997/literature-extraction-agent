"""Unit tests for Stage 2 Extraction (no LLM calls)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_build_records_model():
    from src.agent.extraction_schema import build_records_model

    records = build_records_model([
        {"name": "patient_group", "definition": "Patient group", "type": "string"},
        {"name": "sample_size", "definition": "Sample size", "type": "number"},
        {"name": "os", "definition": "Overall survival", "type": "string"},
    ])
    schema = records.model_json_schema()
    assert "records" in schema["properties"]
    record_schema = schema["$defs"]["Record"]
    assert {"patient_group", "sample_size", "os"} <= set(record_schema["properties"])
    assert record_schema.get("additionalProperties") is False


def test_create_instruction_is_domain_neutral():
    from src.agent.extraction_schema import create_instruction

    class FakeRecord:
        name = "test_record"
        fields = [
            type("F", (), {"name": "f1", "definition": "Field one"}),
            type("F", (), {"name": "f2", "definition": "Field two"}),
        ]

    instruction = create_instruction(FakeRecord())
    assert all(value in instruction for value in ("test_record", "f1", "f2", "Field one"))
    for term in ("clinical", "patient", "treatment", "trial"):
        assert term not in instruction.lower()


def test_build_system_message_is_domain_neutral():
    from src.agent.extraction_schema import build_system_message

    class FakeRecord:
        name = "material_record"
        meaning = "A material record represents one unique alloy composition."
        fields = []

    message = build_system_message(FakeRecord())
    assert "material_record" in message
    assert "unique alloy composition" in message
    for term in ("clinical", "patient", "treatment"):
        assert term not in message.lower()


def test_context_order_and_abstract_inclusion():
    from src.agent.tools.context_builder import build_context

    chunks = {
        "c1": {"paper_id": "P1", "chunk_id": "c1", "chunk_type": "paragraph", "text": "At index 5", "section_path": ["Results"], "metadata": {"chunk_index": 5}},
        "c2": {"paper_id": "P1", "chunk_id": "c2", "chunk_type": "abstract", "text": "At index 0", "section_path": ["Abstract"], "metadata": {"chunk_index": 0}},
        "c3": {"paper_id": "P1", "chunk_id": "c3", "chunk_type": "paragraph", "text": "At index 3", "section_path": ["Methods"], "metadata": {"chunk_index": 3}},
    }
    context, used = build_context("P1", chunks, {"c1", "c3"})
    assert used == ["c2", "c3", "c1"]
    assert context.index("At index 0") < context.index("At index 3")


def test_context_table_render():
    from src.agent.tools.context_builder import build_context

    chunks = {
        "t1": {
            "paper_id": "P1", "chunk_id": "t1", "chunk_type": "table", "section_path": ["Results"],
            "metadata": {"chunk_index": 1, "caption": "Table 1. Survival", "markdown_text": "| Group | OS |\n|---|---|\n| A | 24.5 |"},
        }
    }
    context, _ = build_context("P1", chunks, {"t1"})
    assert "Table 1. Survival" in context
    assert "| Group | OS |" in context


def test_context_truncation_keeps_abstract():
    from src.agent.tools import context_builder as cb

    previous = cb.MAX_CONTEXT_CHARS
    cb.MAX_CONTEXT_CHARS = 200
    try:
        chunks = {
            "a1": {"paper_id": "P1", "chunk_id": "a1", "chunk_type": "abstract", "text": "Short abstract.", "section_path": ["Abstract"], "metadata": {"chunk_index": 0}},
            "p1": {"paper_id": "P1", "chunk_id": "p1", "chunk_type": "paragraph", "text": "A" * 100, "section_path": ["Methods"], "metadata": {"chunk_index": 1}},
            "p2": {"paper_id": "P1", "chunk_id": "p2", "chunk_type": "paragraph", "text": "B" * 100, "section_path": ["Results"], "metadata": {"chunk_index": 2}},
        }
        _, used = cb.build_context("P1", chunks, {"p1", "p2"})
        assert "a1" in used
        assert len(used) < 3
    finally:
        cb.MAX_CONTEXT_CHARS = previous


def test_record_cleanup_normalize_and_deduplicate():
    from src.agent.tools.record_cleanup import _normalize, deduplicate

    assert _normalize(None) == ""
    assert _normalize("Stage  I–III  ") == "stage i–iii"
    fields = ["patient_group", "treatment_regimen", "os"]
    records = [
        {"patient_group": "Group A", "treatment_regimen": "Drug X", "os": "12 months"},
        {"patient_group": "Group A", "treatment_regimen": "Drug X", "os": "12 months"},
        {"patient_group": "Group A", "treatment_regimen": "Drug X", "os": None},
    ]
    deduped, removed = deduplicate(records, fields)
    assert len(deduped) == 2
    assert removed == 1

    different_group, removed = deduplicate([
        {"patient_group": "Group A", "treatment_regimen": "Drug X", "os": "12 months"},
        {"patient_group": "Group B", "treatment_regimen": "Drug X", "os": "12 months"},
    ], fields)
    assert len(different_group) == 2
    assert removed == 0


def test_assign_ids_and_field_complement():
    from src.agent.tools.record_cleanup import assign_ids_and_source

    cleaned = assign_ids_and_source(
        [{"patient_group": "Group A", "os": "12 months"}],
        "PMC123",
        ["patient_group", "treatment_regimen", "os"],
        ["c1", "c2"],
    )
    row = cleaned[0]
    assert row["record_id"] == "PMC123::r0001"
    assert row["treatment_regimen"] is None
    assert row["source_chunk_ids"] == ["c1", "c2"]


def _strict_constraint():
    from src.agent.tools.endpoint_constraints import build_endpoint_constraint

    return build_endpoint_constraint({
        "mode": "strict",
        "by_record_type": {
            "clinical_outcome": ["OS", "PFS"],
            "pk": ["clearance", "AUC"],
        },
        "aliases": {"OS": ["overall survival"]},
    })


def test_endpoint_constraint_accepts_compatible_pairs():
    constraint = _strict_constraint()
    kept, stats = constraint.apply([
        {"record_type": "clinical_outcome", "endpoint": "OS"},
        {"record_type": "pk", "endpoint": "clearance"},
    ])
    assert len(kept) == 2
    assert stats["endpoint_constraint_rejected"] == 0


def test_endpoint_constraint_rejects_incompatible_pair():
    constraint = _strict_constraint()
    kept, stats = constraint.apply([
        {"record_type": "clinical_outcome", "endpoint": "clearance"},
    ])
    assert kept == []
    assert stats["endpoint_constraint_rejected"] == 1
    assert stats["endpoint_constraint_rejected_by_combo"] == {"clinical_outcome | clearance": 1}


def test_endpoint_constraint_canonicalizes_alias_before_dedupe():
    from src.agent.tools.record_cleanup import deduplicate

    constraint = _strict_constraint()
    normalized, _ = constraint.apply([
        {"record_type": "clinical_outcome", "endpoint": "overall survival", "compound": "Drug A"},
        {"record_type": "clinical_outcome", "endpoint": "OS", "compound": "Drug A"},
    ])
    deduped, removed = deduplicate(normalized, ["record_type", "endpoint", "compound"])
    assert len(deduped) == 1
    assert removed == 1
    assert deduped[0]["endpoint"] == "OS"


def test_unrestricted_constraint_is_disabled():
    from src.agent.tools.endpoint_constraints import build_endpoint_constraint

    assert build_endpoint_constraint(None) is None
    assert build_endpoint_constraint({"mode": "unrestricted"}) is None


def main():
    tests = [
        test_build_records_model,
        test_create_instruction_is_domain_neutral,
        test_build_system_message_is_domain_neutral,
        test_context_order_and_abstract_inclusion,
        test_context_table_render,
        test_context_truncation_keeps_abstract,
        test_record_cleanup_normalize_and_deduplicate,
        test_assign_ids_and_field_complement,
        test_endpoint_constraint_accepts_compatible_pairs,
        test_endpoint_constraint_rejects_incompatible_pair,
        test_endpoint_constraint_canonicalizes_alias_before_dedupe,
        test_unrestricted_constraint_is_disabled,
    ]
    for test in tests:
        test()
        print(f"  ✓ {test.__name__}")
    print(f"✅ All {len(tests)} tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
