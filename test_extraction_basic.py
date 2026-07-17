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


def test_records_model_accepts_numeric_scalars_before_string_export():
    from src.agent.extraction_schema import build_records_model
    from src.agent.tools.extractor import _records_to_dicts, _coerce_records_to_strings

    records_model = build_records_model([
        {"name": "sample_size", "definition": "Sample size", "type": "string"},
        {"name": "value", "definition": "Value", "type": "string"},
    ])
    validated = records_model.model_validate({
        "records": [{"sample_size": 2959, "value": 29}]
    })
    rows = _coerce_records_to_strings(_records_to_dicts(validated))

    assert rows == [{"sample_size": "2959", "value": "29"}]


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


def test_context_adds_unlabeled_numeric_endpoint_evidence():
    from src.agent.tools.context_builder import build_context

    chunks = {
        "a1": {
            "paper_id": "P1",
            "chunk_id": "a1",
            "chunk_type": "abstract",
            "text": "A pancreatic cancer drug study.",
            "section_path": ["Abstract"],
            "metadata": {"chunk_index": 0},
        },
        "p1": {
            "paper_id": "P1",
            "chunk_id": "p1",
            "chunk_type": "paragraph",
            "text": "General mechanism text without a concrete endpoint value.",
            "section_path": ["Results"],
            "metadata": {"chunk_index": 1},
        },
        "p2": {
            "paper_id": "P1",
            "chunk_id": "p2",
            "chunk_type": "paragraph",
            "text": "The IC50 values ranged from 3 to 10 uM across PDAC cells.",
            "section_path": ["Results"],
            "metadata": {"chunk_index": 2},
        },
    }
    context, used = build_context("P1", chunks, {"p1"})

    assert "p2" in used
    assert "IC50 values ranged from 3 to 10 uM" in context


def test_record_source_chunk_selection_prefers_matching_evidence():
    from src.agent.tools.context_builder import select_record_source_chunk_ids

    chunks = {
        "a1": {
            "paper_id": "P1",
            "chunk_id": "a1",
            "chunk_type": "abstract",
            "text": "Drug A was studied in pancreatic cancer.",
            "section_path": ["Abstract"],
            "metadata": {"chunk_index": 0},
        },
        "p1": {
            "paper_id": "P1",
            "chunk_id": "p1",
            "chunk_type": "paragraph",
            "text": "The total registry cohort contained 263,886 patients.",
            "section_path": ["Results"],
            "metadata": {"chunk_index": 1},
        },
        "p2": {
            "paper_id": "P1",
            "chunk_id": "p2",
            "chunk_type": "paragraph",
            "text": "Drug A improved median OS to 10.60 months compared with control.",
            "section_path": ["Results"],
            "metadata": {"chunk_index": 2},
        },
    }
    selected = select_record_source_chunk_ids(
        {
            "compound_or_treatment": "Drug A",
            "endpoint": "OS",
            "value": "10.60",
            "unit": "months",
        },
        chunks,
        ["a1", "p1", "p2"],
        max_chunks=2,
    )
    assert selected == ["a1", "p2"]


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


def test_assign_ids_accepts_per_record_sources():
    from src.agent.tools.record_cleanup import assign_ids_and_source

    cleaned = assign_ids_and_source(
        [
            {"patient_group": "Group A", "os": "12 months"},
            {"patient_group": "Group B", "os": "18 months"},
        ],
        "PMC123",
        ["patient_group", "os"],
        [["c1"], ["c2", "c3"]],
    )
    assert cleaned[0]["source_chunk_ids"] == ["c1"]
    assert cleaned[1]["source_chunk_ids"] == ["c2", "c3"]


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


def test_transport_error_stops_output_mode_fallback(monkeypatch=None):
    from src.agent.tools import extractor

    class DummyLLM:
        pass

    class DummyRecords:
        pass

    original_build_llm = extractor._build_llm
    original_plain = extractor._invoke_plain_json
    original_structured = extractor._invoke_structured
    structured_calls = []
    try:
        extractor._build_llm = lambda _: DummyLLM()

        def fail_plain(*args, **kwargs):
            raise TimeoutError("request timed out")

        def structured(*args, **kwargs):
            structured_calls.append(kwargs["method"])
            return {"records": []}

        extractor._invoke_plain_json = fail_plain
        extractor._invoke_structured = structured
        rows, status = extractor.extract_records(
            "context", "system", "instruction", DummyRecords(), model_name="kimi-k2.6"
        )
        assert rows == []
        assert status == "failed:TimeoutError"
        assert structured_calls == []
    finally:
        extractor._build_llm = original_build_llm
        extractor._invoke_plain_json = original_plain
        extractor._invoke_structured = original_structured


def test_extractor_disables_implicit_retries(monkeypatch=None):
    import os
    from src.agent.tools.extractor import _build_llm

    previous = os.environ.get("EXTRACTOR_MAX_RETRIES")
    try:
        os.environ["EXTRACTOR_MAX_RETRIES"] = "0"
        llm = _build_llm("kimi-k2.6")
        assert llm.max_retries == 0
    finally:
        if previous is None:
            os.environ.pop("EXTRACTOR_MAX_RETRIES", None)
        else:
            os.environ["EXTRACTOR_MAX_RETRIES"] = previous


def main():
    tests = [
        test_build_records_model,
        test_records_model_accepts_numeric_scalars_before_string_export,
        test_create_instruction_is_domain_neutral,
        test_build_system_message_is_domain_neutral,
        test_context_order_and_abstract_inclusion,
        test_context_table_render,
        test_context_truncation_keeps_abstract,
        test_context_adds_unlabeled_numeric_endpoint_evidence,
        test_record_source_chunk_selection_prefers_matching_evidence,
        test_record_cleanup_normalize_and_deduplicate,
        test_assign_ids_and_field_complement,
        test_assign_ids_accepts_per_record_sources,
        test_endpoint_constraint_accepts_compatible_pairs,
        test_endpoint_constraint_rejects_incompatible_pair,
        test_endpoint_constraint_canonicalizes_alias_before_dedupe,
        test_unrestricted_constraint_is_disabled,
        test_transport_error_stops_output_mode_fallback,
        test_extractor_disables_implicit_retries,
    ]
    for test in tests:
        test()
        print(f"  ✓ {test.__name__}")
    print(f"✅ All {len(tests)} tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
