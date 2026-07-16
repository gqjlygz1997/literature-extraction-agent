"""Unit tests for resumable batch behavior (no network or LLM calls)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_paper_filter_limit_skips_existing():
    print("\nTesting paper-filter resume limit semantics...")

    from src.agent.config_schema import PaperFilterConfig, PaperFilterConfigFile
    from src.agent.tools.llm_labeler import PaperFilterDecision
    from src.agent.workflow import DomainExtractionWorkflow

    class FakeConfigGenerator:
        @staticmethod
        def save(config, path):
            path.write_text("domain_name: test\n", encoding="utf-8")

    class FakeLabeler:
        calls: list[str] = []

        @classmethod
        def classify_paper(cls, article_meta, paper_filter_config):
            cls.calls.append(article_meta.paper_id)
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
        rows = [
            {
                "paper_id": f"WOS_{idx}",
                "metadata_source": "wos",
                "file_type": "metadata",
                "title": f"Paper {idx}",
                "abstract": "Reports relevant evidence.",
                "text_for_filter": "Reports relevant evidence.",
                "wos_uid": f"WOS:{idx}",
            }
            for idx in range(1, 4)
        ]
        metadata_path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

        workflow = DomainExtractionWorkflow(
            config_generator=FakeConfigGenerator(),
            document_parser=None,
            paper_filter_labeler=FakeLabeler(),
        )
        out_dir = tmp_path / "out"
        first = workflow.run_paper_filter_mvp(
            papers_dir=None,
            metadata_path=metadata_path,
            user_requirements=None,
            output_dir=out_dir,
            paper_filter_config=config,
            limit=1,
        )
        second = workflow.run_paper_filter_mvp(
            papers_dir=None,
            metadata_path=metadata_path,
            user_requirements=None,
            output_dir=out_dir,
            paper_filter_config=config,
            limit=1,
        )

        results = _read_jsonl(out_dir / "paper_filter_results.jsonl")

    assert first["processed_this_run"] == 1
    assert second["processed_this_run"] == 1
    assert [row["paper_id"] for row in results] == ["WOS_1", "WOS_2"]
    assert FakeLabeler.calls == ["WOS_1", "WOS_2"]
    print("  ✓ paper-filter resumes with limit as new-paper batch size")
    return True


def test_preprocess_limit_skips_existing():
    print("\nTesting preprocess resume limit semantics...")

    from src.agent.tools.document_parser import DocumentChunk
    from src.agent.workflow import DomainExtractionWorkflow

    class FakeParser:
        @staticmethod
        def parse_full_text_with_skips(source_path):
            paper_id = Path(source_path).stem
            return [
                DocumentChunk(
                    paper_id=paper_id,
                    chunk_id=f"{paper_id}::c0001",
                    chunk_type="abstract",
                    text=f"{paper_id} abstract",
                    section_path=["Abstract"],
                    metadata={"chunk_index": 0},
                )
            ], {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        passed_path = tmp_path / "passed_papers.jsonl"
        rows = [
            {"paper_id": name, "source_path": str(tmp_path / f"{name}.xml"), "file_type": "xml"}
            for name in ("A", "B", "C")
        ]
        passed_path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

        workflow = DomainExtractionWorkflow(
            config_generator=None,
            document_parser=FakeParser(),
            paper_filter_labeler=None,
        )
        out_dir = tmp_path / "out"
        first = workflow.run_preprocess_mvp(passed_path, out_dir, limit=1)
        second = workflow.run_preprocess_mvp(passed_path, out_dir, limit=1)
        chunks = _read_jsonl(out_dir / "parsed_chunks.jsonl")

    assert first["processed_this_run"] == 1
    assert second["processed_this_run"] == 1
    assert [row["paper_id"] for row in chunks] == ["A", "B"]
    print("  ✓ preprocessing resumes with limit as new-paper batch size")
    return True


def test_fulltext_acquisition_limit_skips_existing():
    print("\nTesting full-text acquisition resume limit semantics...")

    from src.agent.tools import pmc_downloader

    original_resolve = pmc_downloader.resolve_pmcid
    original_download = pmc_downloader.download_pmc_xml
    try:
        pmc_downloader.resolve_pmcid = lambda row, **_: row["pmcid"]
        pmc_downloader.download_pmc_xml = (
            lambda pmcid, output_path: Path(output_path).write_text(
                "<article></article>", encoding="utf-8"
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = [
                {"paper_id": f"P{idx}", "pmcid": f"PMC{idx}", "title": f"Paper {idx}"}
                for idx in range(1, 4)
            ]
            first = pmc_downloader.acquire_pmc_xml(
                rows, tmp_path / "out", limit=1, sleep_seconds=0
            )
            second = pmc_downloader.acquire_pmc_xml(
                rows, tmp_path / "out", limit=1, sleep_seconds=0
            )
            results = _read_jsonl(tmp_path / "out" / "fulltext_acquisition_results.jsonl")
    finally:
        pmc_downloader.resolve_pmcid = original_resolve
        pmc_downloader.download_pmc_xml = original_download

    assert first["processed_this_run"] == 1
    assert second["processed_this_run"] == 1
    assert [row["pmcid"] for row in results] == ["PMC1", "PMC2"]
    print("  ✓ full-text acquisition resumes with limit as new-paper batch size")
    return True


if __name__ == "__main__":
    tests = [
        test_paper_filter_limit_skips_existing,
        test_preprocess_limit_skips_existing,
        test_fulltext_acquisition_limit_skips_existing,
    ]

    ok = True
    for test in tests:
        try:
            ok = test() and ok
        except Exception as exc:  # noqa: BLE001
            print(f"✗ {test.__name__} failed: {exc}")
            ok = False

    if ok:
        print("\n✅ All resume tests passed!")
        sys.exit(0)

    print("\n❌ Some resume tests failed!")
    sys.exit(1)
