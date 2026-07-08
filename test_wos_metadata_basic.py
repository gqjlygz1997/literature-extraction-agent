"""Unit tests for WOS metadata ingestion and metadata-based paper filtering."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_parse_wos_savedrecs():
    print("\nTesting WOS savedrecs parser...")

    from src.agent.tools.wos_parser import parse_wos_file

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "savedrecs.txt"
        path.write_text(
            "\n".join(
                [
                    "FN Clarivate Analytics Web of Science",
                    "VR 1.0",
                    "PT J",
                    "AU Shin, S",
                    "TI Survival outcomes in pancreatic cancer",
                    "AB This study reports treatment outcomes.",
                    "   It includes survival endpoints.",
                    "DI 10.1000/test.doi",
                    "PM 12345678",
                    "UT WOS:000000000000001",
                    "ER",
                    "EF",
                ]
            ),
            encoding="utf-8",
        )

        rows = parse_wos_file(path)

    assert len(rows) == 1
    row = rows[0]
    assert row["paper_id"] == "WOS_000000000000001"
    assert row["file_type"] == "metadata"
    assert row["doi"] == "10.1000/test.doi"
    assert row["pmid"] == "12345678"
    assert "survival endpoints" in row["abstract"]
    assert row["abstract_available"] is True

    print("  ✓ WOS parser keeps title, abstract, DOI, PMID, and WOS UID")
    return True


def test_metadata_paper_filter_outputs_identifiers():
    print("\nTesting metadata paper-filter output shape...")

    from src.agent.config_schema import PaperFilterConfig, PaperFilterConfigFile
    from src.agent.tools.llm_labeler import PaperFilterDecision
    from src.agent.workflow import DomainExtractionWorkflow

    class FakeConfigGenerator:
        @staticmethod
        def save(config, path):
            path.write_text("domain_name: test\n", encoding="utf-8")

    class FakeLabeler:
        model_name = "fake"

        @staticmethod
        def classify_paper(article_meta, paper_filter_config):
            return PaperFilterDecision(
                paper_id=article_meta.paper_id,
                decision="pass",
                reason="fake_pass",
                model="fake",
            )

    config = PaperFilterConfigFile(
        domain_name="test",
        domain_description="test",
        target_fields=[],
        paper_filter=PaperFilterConfig(criteria=[]),
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        metadata_path = tmp_path / "candidate_papers.jsonl"
        metadata_row = {
            "paper_id": "WOS_1",
            "metadata_source": "wos",
            "source_path": "",
            "source_file": "savedrecs.txt",
            "file_type": "metadata",
            "title": "Pancreatic cancer treatment outcomes",
            "abstract": "Reports overall survival after treatment.",
            "text_for_filter": "Reports overall survival after treatment.",
            "doi": "10.1000/test.doi",
            "pmid": "12345678",
            "pmcid": "",
            "wos_uid": "WOS:1",
            "abstract_available": True,
            "metadata_quality": "external_metadata",
        }
        metadata_path.write_text(json.dumps(metadata_row) + "\n", encoding="utf-8")

        workflow = DomainExtractionWorkflow(
            config_generator=FakeConfigGenerator(),
            document_parser=None,
            paper_filter_labeler=FakeLabeler(),
        )
        counts = workflow.run_paper_filter_mvp(
            papers_dir=None,
            metadata_path=metadata_path,
            user_requirements=None,
            output_dir=tmp_path / "out",
            paper_filter_config=config,
        )

        passed_path = tmp_path / "out" / "passed_papers.jsonl"
        passed = [json.loads(line) for line in passed_path.read_text().splitlines()]

    assert counts["passed"] == 1
    assert passed[0]["file_type"] == "metadata"
    assert passed[0]["doi"] == "10.1000/test.doi"
    assert passed[0]["pmid"] == "12345678"
    assert passed[0]["wos_uid"] == "WOS:1"
    assert passed[0]["metadata_source"] == "wos"

    print("  ✓ metadata pass rows preserve DOI/PMID for full-text acquisition")
    return True


if __name__ == "__main__":
    tests = [
        test_parse_wos_savedrecs,
        test_metadata_paper_filter_outputs_identifiers,
    ]

    ok = True
    for test in tests:
        try:
            ok = test() and ok
        except Exception as exc:  # noqa: BLE001
            print(f"✗ {test.__name__} failed: {exc}")
            ok = False

    if ok:
        print("\n✅ All WOS metadata tests passed!")
        sys.exit(0)

    print("\n❌ Some WOS metadata tests failed!")
    sys.exit(1)
