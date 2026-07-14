"""
test_postprocess_basic.py — Unit tests for Stage 3 Post-processing.

测试范围（不调 LLM）：
- 通用数值 parser
- preset 同义词标准化
- run_postprocess 最小端到端输出 JSONL / CSV / summary
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_parse_numeric_value():
    print("\n测试：parse_numeric_value")
    from src.agent.tools.postprocess import parse_numeric_value

    ys = parse_numeric_value("200-300 MPa", default_unit="MPa")
    assert ys["operator"] == "range"
    assert ys["value"] == 250.0
    assert ys["value_min"] == 200.0
    assert ys["value_max"] == 300.0
    assert ys["unit"] == "MPa"

    p = parse_numeric_value("<0.001")
    assert p["operator"] == "<"
    assert p["value"] == 0.001
    assert p["value_max"] == 0.001

    grain = parse_numeric_value("500 nm", default_unit="um")
    assert grain["value"] == 0.5
    assert grain["unit"] == "um"

    # 测试分钟时间单位识别
    duration_min = parse_numeric_value("20 min")
    assert duration_min["value"] == 20.0
    assert duration_min["unit"] == "min"
    assert duration_min["operator"] == "eq"

    duration_minutes = parse_numeric_value("15 minutes")
    assert duration_minutes["value"] == 15.0
    assert duration_minutes["unit"] == "min"
    assert duration_minutes["operator"] == "eq"

    # 确认已有测试：72 h 保持 h 单位
    duration_hours = parse_numeric_value("72 h")
    assert duration_hours["value"] == 72.0
    assert duration_hours["unit"] == "h"

    print("  ✓ 数值解析正确")
    return True


def test_standardize_value():
    print("\n测试：standardize_value")
    from src.agent.tools.postprocess import standardize_value

    phase_cfg = {
        "multiple": True,
        "terms": {
            "FCC": ["FCC", "f.c.c.", "face-centered cubic"],
            "BCC": ["BCC", "b.c.c.", "body-centered cubic"],
        },
    }
    assert standardize_value("face-centered cubic + BCC", phase_cfg) == ["FCC", "BCC"]

    line_cfg = {
        "terms": {
            "first-line": ["first-line", "first line", "1L"],
        }
    }
    assert standardize_value("1L therapy", line_cfg) == "first-line"

    treatment_cfg = {
        "match": "exact",
        "terms": {
            "surgery": ["surgery", "surgical resection"],
        },
    }
    assert standardize_value("neoadjuvant therapy followed by surgery", treatment_cfg) == "neoadjuvant therapy followed by surgery"

    print("  ✓ 同义词标准化正确")
    return True


def test_postprocess_records():
    print("\n测试：postprocess_records")
    from src.agent.user_requirements import FieldSpec
    from src.agent.tools.postprocess import postprocess_records

    fields = [
        FieldSpec("treatment_regimen", "Treatment", "string"),
        FieldSpec("os", "Overall survival", "number"),
        FieldSpec("p_value", "p-value", "string"),
    ]
    records = [
        {
            "paper_id": "P1",
            "record_id": "P1::r0001",
            "treatment_regimen": "mFOLFIRINOX",
            "os": "24.3 months",
            "p_value": "<0.001",
            "source_chunk_ids": ["c1"],
        },
        {
            "paper_id": "P1",
            "record_id": "P1::r0002",
            "treatment_regimen": "mFOLFIRINOX",
            "os": "24.3 mo",
            "p_value": "<0.001",
            "source_chunk_ids": ["c2"],
        },
        {
            "paper_id": "P1",
            "record_id": "P1::r0003",
            "treatment_regimen": None,
            "os": None,
            "p_value": None,
            "source_chunk_ids": ["c3"],
        },
    ]
    config = {
        "numeric_fields": {"os": {"unit": "month"}, "p_value": {}},
        "standardize": {
            "treatment_regimen": {
                "terms": {"FOLFIRINOX": ["mFOLFIRINOX", "modified FOLFIRINOX"]}
            }
        },
        "validity": {"required_any": ["treatment_regimen", "os", "p_value"]},
    }

    rows, summary = postprocess_records(records, fields, config)
    assert len(rows) == 1
    assert rows[0]["treatment_regimen"] == "FOLFIRINOX"
    assert rows[0]["os_norm"]["value"] == 24.3
    assert rows[0]["source_chunk_ids"] == ["c1", "c2"]
    assert summary["invalid_removed"] == 1
    assert summary["duplicates_removed"] == 1

    print("  ✓ 清洗、标准化、过滤、去重正确")
    return True


def test_run_postprocess_minimal():
    print("\n测试：run_postprocess 最小端到端")
    from src.agent.workflow import DomainExtractionWorkflow

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        req_path = root / "user_requirements.yaml"
        records_path = root / "extracted_records.jsonl"
        config_path = root / "postprocess_config.yaml"
        output_dir = root / "out"

        req_path.write_text(
            """
project_name: minimal_postprocess
domain_description: Minimal test.
record:
  name: test_record
  meaning: One minimal record.
  fields:
    - name: group
      definition: Group name.
      type: string
    - name: outcome
      definition: Outcome value.
      type: number
""".strip(),
            encoding="utf-8",
        )
        records_path.write_text(
            json.dumps({
                "paper_id": "P1",
                "record_id": "P1::r0001",
                "group": "Group A",
                "outcome": "10-20 months",
                "source_chunk_ids": ["c1"],
            }) + "\n",
            encoding="utf-8",
        )
        config_path.write_text(
            """
numeric_fields:
  outcome:
    unit: month
validity:
  required_any: [outcome]
""".strip(),
            encoding="utf-8",
        )

        workflow = DomainExtractionWorkflow(None, None, None)
        summary = workflow.run_postprocess(
            requirements_path=req_path,
            extracted_records_path=records_path,
            output_dir=output_dir,
            config_path=config_path,
            use_presets=False,
        )

        assert summary["records_output"] == 1
        assert (output_dir / "postprocessed_records.jsonl").exists()
        assert (output_dir / "records.csv").exists()
        assert (output_dir / "postprocessing_summary.json").exists()

        row = json.loads((output_dir / "postprocessed_records.jsonl").read_text(encoding="utf-8").strip())
        assert row["outcome_norm"]["value"] == 15.0

    print("  ✓ 最小端到端输出正确")
    return True


def test_pancan_preset_examples():
    print("\n测试：pancan preset drug-development 例子")
    import yaml

    from src.agent.user_requirements import load_user_requirements
    from src.agent.tools.postprocess import postprocess_records

    root = Path(__file__).parent
    req = load_user_requirements(root / "examples" / "pancan_treatment_outcomes" / "user_requirements.yaml")
    with open(root / "presets" / "pancan_treatment_outcomes" / "postprocess_config.yaml", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    records = [
        {
            "paper_id": "PANCAN1",
            "record_id": "PANCAN1::r0001",
            "record_type": "cell assay",
            "compound_or_treatment": "gemcitabine",
            "model_or_population": "PANC-1 cell line",
            "assay_or_study_type": "IC50 assay",
            "endpoint": "half maximal inhibitory concentration",
            "value": "0.8 uM",
            "unit": "uM",
            "dose": None,
            "route": None,
            "duration": "72 h",
            "comparator_or_control": "vehicle",
            "sample_size": None,
            "statistics": "p < 0.001",
            "source_chunk_ids": ["c1"],
        },
        {
            # 只有 compound_or_treatment / endpoint，缺少 record_type / model_or_population
            # required_all 下应被过滤
            "paper_id": "PANCAN1",
            "record_id": "PANCAN1::r0002",
            "record_type": None,
            "compound_or_treatment": "erlotinib",
            "model_or_population": None,
            "assay_or_study_type": None,
            "endpoint": "cell viability",
            "value": None,
            "unit": None,
            "dose": None,
            "route": None,
            "duration": None,
            "comparator_or_control": None,
            "sample_size": None,
            "statistics": None,
            "source_chunk_ids": ["c2"],
        },
        {
            "paper_id": "PANCAN1",
            "record_id": "PANCAN1::r0003",
            "record_type": None,
            "compound_or_treatment": None,
            "model_or_population": None,
            "assay_or_study_type": None,
            "endpoint": None,
            "value": None,
            "unit": None,
            "dose": None,
            "route": None,
            "duration": None,
            "comparator_or_control": None,
            "sample_size": None,
            "statistics": None,
            "source_chunk_ids": ["c2"],
        },
    ]

    rows, summary = postprocess_records(records, req.record.fields, config)
    assert len(rows) == 1
    row = rows[0]

    assert row["record_type"] == "in_vitro_efficacy"
    assert row["endpoint"] == "IC50"
    assert row["value_norm"]["value"] == 0.8
    assert row["value_norm"]["unit"] == "uM"
    assert row["duration_norm"]["value"] == 72.0
    assert row["duration_norm"]["unit"] == "h"
    assert row["statistics_norm"]["operator"] == "<"
    assert row["statistics_norm"]["value"] == 0.001
    # r0002（缺 record_type / model_or_population）和 r0003（全空）都应被 required_all 过滤
    assert summary["invalid_removed"] == 2
    kept_ids = {r["record_id"] for r in rows}
    assert "PANCAN1::r0002" not in kept_ids
    assert "PANCAN1::r0003" not in kept_ids

    print("  ✓ pancan preset 解析和标准化正确")
    return True


def main():
    print("=" * 60)
    print("Post-processing Stage — Basic Tests")
    print("=" * 60)

    tests = [
        test_parse_numeric_value,
        test_standardize_value,
        test_postprocess_records,
        test_run_postprocess_minimal,
        test_pancan_preset_examples,
    ]

    all_pass = True
    for test_fn in tests:
        try:
            if not test_fn():
                all_pass = False
        except Exception as e:
            print(f"\n❌ {test_fn.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print(f"✅ All {len(tests)} tests passed!")
        return 0
    print("❌ Some tests failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
