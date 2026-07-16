"""Workflow controller for the literature extraction agent."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from tqdm import tqdm

from .config_schema import PaperFilterConfig, PaperFilterConfigFile, TargetField

logger = logging.getLogger(__name__)


class DomainExtractionWorkflow:
    """Coordinate config generation, paper filtering, retrieval, and extraction."""

    def __init__(
        self,
        config_generator,
        document_parser,
        paper_filter_labeler,
        retriever=None,
        regex_filter=None,
        evidence_labeler=None,
        extractor=None,
        verifier=None,
    ):
        self.config_generator = config_generator
        self.document_parser = document_parser
        self.paper_filter_labeler = paper_filter_labeler
        self.retriever = retriever
        self.regex_filter = regex_filter
        self.evidence_labeler = evidence_labeler
        self.extractor = extractor
        self.verifier = verifier

    # ------------------------------------------------------------------ #
    #  Phase 1: paper-filter MVP                                           #
    # ------------------------------------------------------------------ #

    def run_paper_filter_mvp(
        self,
        papers_dir: str | Path | None,
        user_requirements,
        output_dir: str | Path,
        metadata_path: str | Path | None = None,
        paper_filter_config=None,
        dry_run: bool = False,
        limit: int | None = None,
        resume: bool = True,
    ):
        """Run the paper-filter MVP over local papers or external metadata.

        Flow:
        1. Generate paper_filter.yaml, or reuse one only when provided via --config.
        2. Read WOS/candidate metadata rows, or scan papers_dir for XML/HTML files.
        3. Build title + text_for_filter per paper.
        4. Classify each paper (pass / reject) via LLM (skipped in dry_run).
        5. Write paper_filter_results.jsonl, passed_papers.jsonl,
           rejected_papers.jsonl, run_summary.json.

        Returns a dict with summary counts.
        """
        if (papers_dir is None) == (metadata_path is None):
            raise ValueError("Provide exactly one of papers_dir or metadata_path.")

        papers_dir = Path(papers_dir) if papers_dir is not None else None
        metadata_path = Path(metadata_path) if metadata_path is not None else None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- step 1: config -------------------------------------------
        config_path = output_dir / "paper_filter.yaml"
        if paper_filter_config is not None:
            config = paper_filter_config
            logger.info("Using provided paper filter config (skipping generation).")
            self.config_generator.save(config, config_path)
        elif resume and config_path.exists():
            logger.info("Found existing paper filter config at %s; reusing it.", config_path)
            from .config_generator import ConfigGenerator
            config = ConfigGenerator.load(config_path)
        elif dry_run:
            logger.info("Dry run: skipping LLM config generation.")
            config = self._build_dry_run_config(user_requirements)
        else:
            logger.info("Generating paper filter config via LLM...")
            config = self.config_generator.generate(user_requirements)
            self.config_generator.save(config, config_path)
            logger.info("Saved generated config to %s", config_path)

        # --- step 2 + 3 + 4: collect metadata + classify ---------------
        existing_results = (
            self._read_jsonl_if_exists(output_dir / "paper_filter_results.jsonl")
            if resume
            else []
        )
        done_index = self._identity_index(existing_results)
        all_results: list[dict] = list(existing_results)
        new_results: list[dict] = []

        if metadata_path is not None:
            metadata_rows = self._read_jsonl(metadata_path)
            if resume and done_index:
                metadata_rows = [
                    row for row in metadata_rows
                    if not self._is_done(row, done_index)
                ]
            if limit:
                metadata_rows = metadata_rows[:limit]
            logger.info(
                "Found %d metadata candidate(s) to process in %s (%d already done)",
                len(metadata_rows), metadata_path, len(existing_results),
            )
            for row in tqdm(metadata_rows, desc="Paper filter", unit="paper"):
                result = self._process_one_metadata(row, config, dry_run=dry_run)
                new_results.append(result)
                all_results.append(result)
        else:
            paper_paths = self._scan_papers(papers_dir)
            if resume and done_index:
                paper_paths = [
                    path for path in paper_paths
                    if not self._is_done(path, done_index)
                ]
            if limit:
                paper_paths = paper_paths[:limit]
            logger.info(
                "Found %d paper(s) to process in %s (%d already done)",
                len(paper_paths), papers_dir, len(existing_results),
            )
            for path in tqdm(paper_paths, desc="Paper filter", unit="paper"):
                result = self._process_one_paper(
                    path, config, dry_run=dry_run
                )
                new_results.append(result)
                all_results.append(result)

        # --- step 5: write outputs ------------------------------------
        config_path_for_summary = config_path if config_path.exists() else None
        self._write_outputs(all_results, output_dir, config, config_path_for_summary)

        counts = self._summarise(all_results)
        counts["processed_this_run"] = len(new_results)
        counts["previously_processed"] = len(existing_results)
        logger.info(
            "Done. total=%d passed=%d rejected=%d error=%d dry_run=%d new=%d existing=%d",
            counts["total"],
            counts["passed"],
            counts["rejected"],
            counts["error"],
            counts.get("dry_run", 0),
            counts["processed_this_run"],
            counts["previously_processed"],
        )
        return counts

    # ------------------------------------------------------------------ #
    #  Phase 2: preprocessing MVP (full-text parsing)                     #
    # ------------------------------------------------------------------ #

    def run_preprocess_mvp(
        self,
        passed_papers_path: str | Path,
        output_dir: str | Path,
        limit: int | None = None,
        resume: bool = True,
    ) -> dict:
        """Parse full text of papers that passed the filter.

        Reads passed_papers.jsonl (each row carries source_path + file_type),
        parses each XML paper into section-aware DocumentChunks, and writes:
          - parsed_chunks.jsonl        (one chunk per line)
          - preprocessing_summary.json (counts + skipped sections + errors)

        Only file_type == "xml" is processed; other rows are recorded as
        errors and skipped.  Per-paper parse failures are caught so one bad
        file does not stop the batch.

        Returns a dict of summary counts.
        """
        passed_papers_path = Path(passed_papers_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = self._read_jsonl(passed_papers_path)
        chunks_path = output_dir / "parsed_chunks.jsonl"
        previous_summary = self._read_json_if_exists(
            output_dir / "preprocessing_summary.json"
        ) if resume else {}
        all_chunks: list[dict] = (
            self._read_jsonl_if_exists(chunks_path) if resume else []
        )
        existing_paper_ids = {
            str(chunk.get("paper_id", "")).strip()
            for chunk in all_chunks
            if chunk.get("paper_id")
        }
        skipped_existing = 0
        if resume and existing_paper_ids:
            pending_rows = []
            for row in rows:
                paper_id = str(row.get("paper_id", "")).strip()
                if paper_id and paper_id in existing_paper_ids:
                    skipped_existing += 1
                    continue
                pending_rows.append(row)
            rows_to_process = pending_rows
        else:
            rows_to_process = list(rows)
        if limit:
            rows_to_process = rows_to_process[:limit]
        logger.info(
            "Preprocessing %d passed paper(s) from %s (%d already parsed)",
            len(rows_to_process), passed_papers_path, skipped_existing,
        )

        errors: list[dict] = []
        skipped_sections: dict[str, int] = {}
        processed_paper_ids: set[str] = set()

        for row in tqdm(rows_to_process, desc="Preprocess", unit="paper"):
            paper_id = row.get("paper_id", "")
            source_path = row.get("source_path", "")
            file_type = row.get("file_type", "")
            processed_paper_ids.add(str(paper_id or source_path))

            if file_type != "xml":
                errors.append({
                    "paper_id": paper_id,
                    "source_path": source_path,
                    "error": f"unsupported file_type for full-text parsing: {file_type!r}",
                })
                continue

            try:
                chunks, paper_skips = self.document_parser.parse_full_text_with_skips(source_path)
            except Exception as exc:  # noqa: BLE001
                errors.append({
                    "paper_id": paper_id,
                    "source_path": source_path,
                    "error": str(exc),
                })
                continue

            for c in chunks:
                all_chunks.append({
                    "paper_id": c.paper_id,
                    "chunk_id": c.chunk_id,
                    "chunk_type": c.chunk_type,
                    "text": c.text,
                    "section_path": c.section_path,
                    "metadata": c.metadata,
                })

            for tag, n in paper_skips.items():
                skipped_sections[tag] = skipped_sections.get(tag, 0) + n

        # --- write outputs --------------------------------------------
        self._write_jsonl(chunks_path, all_chunks)

        from .tools.xml_full_text_parser import (
            CHUNKING_CONFIG,
            PARSER_VERSION,
            SUPPORTED_FORMAT,
        )

        type_counts: dict[str, int] = {"paragraph": 0, "table": 0, "abstract": 0}
        parsed_paper_ids: set[str] = set()
        for chunk in all_chunks:
            chunk_type = chunk.get("chunk_type", "")
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
            paper_id = str(chunk.get("paper_id", "")).strip()
            if paper_id:
                parsed_paper_ids.add(paper_id)

        previous_errors = previous_summary.get("errors", [])
        if not isinstance(previous_errors, list):
            previous_errors = []
        completed_after = set(parsed_paper_ids)
        errors = [
            err for err in previous_errors
            if str(err.get("paper_id") or err.get("source_path") or "") not in processed_paper_ids
            and str(err.get("paper_id") or "") not in completed_after
        ] + errors

        previous_skips = previous_summary.get("skipped_sections", {})
        if isinstance(previous_skips, dict):
            for tag, n in previous_skips.items():
                skipped_sections[tag] = skipped_sections.get(tag, 0) + int(n)

        summary = {
            "parser_version": PARSER_VERSION,
            "supported_format": SUPPORTED_FORMAT,
            "chunking_config": CHUNKING_CONFIG,
            "total_papers": len(rows),
            "processed_this_run": len(rows_to_process),
            "skipped_existing": skipped_existing,
            "parsed_ok": len(parsed_paper_ids),
            "parse_error": len(errors),
            "total_chunks": len(all_chunks),
            "paragraph_chunks": type_counts.get("paragraph", 0),
            "table_chunks": type_counts.get("table", 0),
            "abstract_chunks": type_counts.get("abstract", 0),
            "skipped_sections": skipped_sections,
            "errors": errors,
        }
        with open(output_dir / "preprocessing_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)

        logger.info(
            "Done. papers=%d parsed_ok=%d parse_error=%d chunks=%d (para=%d table=%d abs=%d) new=%d existing=%d",
            summary["total_papers"], summary["parsed_ok"], len(errors), summary["total_chunks"],
            summary["paragraph_chunks"], summary["table_chunks"], summary["abstract_chunks"],
            summary["processed_this_run"], summary["skipped_existing"],
        )
        return summary

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        rows: list[dict] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _read_jsonl_if_exists(path: Path) -> list[dict]:
        if not path.exists():
            return []
        return DomainExtractionWorkflow._read_jsonl(path)

    @staticmethod
    def _read_json_if_exists(path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _identity_values(row_or_path) -> set[str]:
        """Return stable identifiers used to decide whether a paper is done."""

        values: set[str] = set()
        if isinstance(row_or_path, Path):
            candidates = [row_or_path.stem, row_or_path.name, str(row_or_path)]
        elif isinstance(row_or_path, str):
            path = Path(row_or_path)
            candidates = [row_or_path, path.stem, path.name]
        else:
            candidates = [
                row_or_path.get("paper_id"),
                row_or_path.get("pmcid"),
                row_or_path.get("pmid"),
                row_or_path.get("doi"),
                row_or_path.get("wos_uid"),
                row_or_path.get("source_path"),
                row_or_path.get("source_file"),
            ]

        for value in candidates:
            text = str(value or "").strip()
            if text:
                values.add(text)
        return values

    @classmethod
    def _identity_index(cls, rows: list[dict]) -> set[str]:
        values: set[str] = set()
        for row in rows:
            values.update(cls._identity_values(row))
        return values

    @classmethod
    def _is_done(cls, row_or_path, done_index: set[str]) -> bool:
        return bool(cls._identity_values(row_or_path) & done_index)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _scan_papers(papers_dir: Path) -> list[Path]:
        """Return all XML and HTML files in papers_dir (non-recursive)."""
        paths: list[Path] = []
        for suffix in ("*.xml", "*.nxml", "*.html", "*.htm"):
            paths.extend(papers_dir.glob(suffix))
        # also surface PDF paths so they get a proper error entry
        for suffix in ("*.pdf",):
            paths.extend(papers_dir.glob(suffix))
        return sorted(paths)

    def _process_one_paper(self, path: Path, config, dry_run: bool) -> dict:
        """Parse metadata and optionally classify one paper. Returns a result dict."""

        # -- parse metadata (light) --
        meta = self.document_parser.parse_metadata_light(path)

        base: dict = {
            "source_path": str(path),
            "source_file": "",
            "file_type": meta.file_type,
            "paper_id": meta.paper_id,
            "doi": meta.doi,
            "pmid": "",
            "pmcid": "",
            "wos_uid": "",
            "metadata_source": "local",
            "title": meta.title,
            "abstract_available": meta.abstract_available,
            "front_matter_used": meta.front_matter_used,
            "metadata_quality": meta.metadata_quality,
        }

        # -- parse error (includes PDF) --
        if meta.metadata_quality == "parse_error":
            return {
                **base,
                "decision": "error",
                "criteria": {},
                "reason": "",
                "model": "",
                "error": meta.error,
            }

        # -- dry run: skip LLM call --
        if dry_run:
            return {
                **base,
                "decision": "dry_run",
                "criteria": {},
                "reason": "dry_run mode; LLM classification skipped",
                "model": "",
            }

        # -- classify --
        decision_obj = self.paper_filter_labeler.classify_paper(
            meta, config.paper_filter
        )

        # serialise CriterionResult objects to plain dicts
        criteria_serial = {
            name: {"answer": cr.answer, "reason": cr.reason}
            for name, cr in decision_obj.criteria.items()
        }

        return {
            **base,
            "decision": decision_obj.decision,
            "criteria": criteria_serial,
            "reason": decision_obj.reason,
            "model": decision_obj.model,
            "reasoning": getattr(decision_obj, "reasoning", ""),
        }

    def _process_one_metadata(self, row: dict, config, dry_run: bool) -> dict:
        """Classify one external metadata row, such as a WOS candidate."""

        from .tools.document_parser import ArticleMeta

        title = str(row.get("title", "") or "").strip()
        abstract = str(row.get("abstract", "") or "").strip()
        text_for_filter = str(row.get("text_for_filter", "") or abstract).strip()
        source_path = str(row.get("source_path", "") or "")
        source_file = str(row.get("source_file", "") or "")
        paper_id = str(row.get("paper_id", "") or row.get("wos_uid", "") or row.get("pmid", "") or row.get("doi", "") or title[:80]).strip()

        if title or text_for_filter:
            metadata_quality = str(row.get("metadata_quality", "") or "external_metadata")
            parse_error = ""
        else:
            metadata_quality = "parse_error"
            parse_error = "No title or abstract found in metadata row"

        meta = ArticleMeta(
            source_path=Path(source_path or source_file or "."),
            file_type="metadata",
            title=title,
            abstract=abstract,
            text_for_filter=text_for_filter,
            paper_id=paper_id,
            doi=str(row.get("doi", "") or "").strip(),
            abstract_available=bool(abstract or text_for_filter),
            front_matter_used=False,
            metadata_quality=metadata_quality,  # type: ignore[arg-type]
            error=parse_error,
        )

        base: dict = {
            "source_path": source_path,
            "source_file": source_file,
            "file_type": "metadata",
            "paper_id": meta.paper_id,
            "doi": meta.doi,
            "pmid": str(row.get("pmid", "") or "").strip(),
            "pmcid": str(row.get("pmcid", "") or "").strip(),
            "wos_uid": str(row.get("wos_uid", "") or "").strip(),
            "metadata_source": str(row.get("metadata_source", "") or "external"),
            "title": meta.title,
            "abstract_available": meta.abstract_available,
            "front_matter_used": meta.front_matter_used,
            "metadata_quality": meta.metadata_quality,
        }

        if meta.metadata_quality == "parse_error":
            return {
                **base,
                "decision": "error",
                "criteria": {},
                "reason": "",
                "model": "",
                "error": meta.error,
            }

        if dry_run:
            return {
                **base,
                "decision": "dry_run",
                "criteria": {},
                "reason": "dry_run mode; LLM classification skipped",
                "model": "",
            }

        decision_obj = self.paper_filter_labeler.classify_paper(
            meta, config.paper_filter
        )
        criteria_serial = {
            name: {"answer": cr.answer, "reason": cr.reason}
            for name, cr in decision_obj.criteria.items()
        }

        return {
            **base,
            "decision": decision_obj.decision,
            "criteria": criteria_serial,
            "reason": decision_obj.reason,
            "model": decision_obj.model,
            "reasoning": getattr(decision_obj, "reasoning", ""),
        }

    @staticmethod
    def _build_dry_run_config(user_requirements) -> PaperFilterConfigFile:
        """Create a minimal config object for metadata-only dry runs."""
        target_fields = []
        if user_requirements is not None:
            target_fields = [
                TargetField(
                    name=f.name,
                    description=f.definition,
                    final_type=f.type,
                )
                for f in user_requirements.target_fields
            ]

        return PaperFilterConfigFile(
            domain_name=getattr(user_requirements, "project_name", "dry_run"),
            domain_description=getattr(user_requirements, "domain_description", ""),
            target_fields=target_fields,
            paper_filter=PaperFilterConfig(criteria=[]),
        )

    @staticmethod
    def _write_outputs(
        all_results: list[dict],
        output_dir: Path,
        config,
        config_path: Path | None,
    ) -> None:
        """Write the four output files."""

        def _write_jsonl(path: Path, rows: list[dict]) -> None:
            with open(path, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        _write_jsonl(output_dir / "paper_filter_results.jsonl", all_results)

        passed = [r for r in all_results if r["decision"] == "pass"]
        rejected = [r for r in all_results if r["decision"] == "reject"]

        def _select(row: dict, keys: tuple[str, ...]) -> dict:
            return {key: row.get(key, "") for key in keys}

        passed_slim = [
            _select(
                r,
                (
                    "paper_id", "source_path", "source_file", "file_type",
                    "doi", "pmid", "pmcid", "wos_uid", "metadata_source",
                    "title", "abstract_available", "metadata_quality",
                ),
            )
            for r in passed
        ]
        _write_jsonl(output_dir / "passed_papers.jsonl", passed_slim)

        rejected_slim = [
            _select(
                r,
                (
                    "paper_id", "source_path", "source_file", "doi", "pmid",
                    "wos_uid", "title", "criteria", "reason",
                ),
            )
            for r in rejected
        ]
        _write_jsonl(output_dir / "rejected_papers.jsonl", rejected_slim)

        counts = DomainExtractionWorkflow._summarise(all_results)
        summary = {
            "domain_name": config.domain_name,
            **counts,
            "config_path": str(config_path.resolve()) if config_path else "",
        }
        with open(output_dir / "run_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)

    @staticmethod
    def _summarise(all_results: list[dict]) -> dict:
        counts = {
            "total": len(all_results),
            "passed": 0,
            "rejected": 0,
            "error": 0,
            "dry_run": 0,
        }
        for r in all_results:
            d = r.get("decision", "error")
            if d == "pass":
                counts["passed"] += 1
            elif d == "reject":
                counts["rejected"] += 1
            elif d == "dry_run":
                counts["dry_run"] += 1
            else:
                counts["error"] += 1
        return counts

    # ------------------------------------------------------------------ #
    #  Phase 3: labeling MVP (evidence chunk identification)              #
    # ------------------------------------------------------------------ #

    def run_labeling_mvp(
        self,
        requirements_path: str | Path,
        parsed_chunks_path: str | Path,
        output_dir: str | Path,
        domain_name: str,
        config_path: str | Path = None,
        preset_dir: str | Path | None = None,
        use_presets: bool = True,
        limit: int | None = None,
        resume: bool = True,
    ) -> dict:
        """
        Labeling 阶段 MVP 主流程

        Flow:
        1. 生成 labeling_config.yaml (DSPy 调用 LLM)
        2. 构建向量库 (增量写入，避免重复 embedding)
        3. 对每篇论文、每个字段：
           a. Section 过滤 (exclude 硬过滤 + include 软偏好)
           b. Text channel: semantic + regex → RRF → text top-5 → LLM 二分类
           c. Table channel: table header → table top-5 → LLM 二分类
        4. 输出 labeled_chunks.jsonl

        Args:
            requirements_path: user_requirements.yaml 路径
            parsed_chunks_path: parsed_chunks.jsonl 路径
            output_dir: 输出目录
            domain_name: 领域名称（用于 Chroma 持久化目录）
            config_path: 可选，已有的 labeling_config.yaml 路径（用于复用配置）
            preset_dir: 可选，预设目录；默认 ./presets
            use_presets: 是否自动寻找 presets/<project_name>/labeling_config.yaml

        Returns:
            summary dict with counts
        """
        from datetime import datetime
        from .labeling_config import LabelingConfigGenerator
        from .preset_manager import find_preset_file
        from .tools.vector_store import VectorStoreBuilder
        from .tools.hybrid_retriever import (
            SectionFilter,
            SemanticRetriever,
            RegexRetriever,
            TableHeaderRetriever,
            RRFFusion,
        )

        requirements_path = Path(requirements_path)
        parsed_chunks_path = Path(parsed_chunks_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Labeling Stage - MVP")
        logger.info("=" * 60)

        # ========== Step 1: 生成或加载 labeling_config.yaml ==========
        logger.info("\n[Step 1] Loading or generating labeling_config.yaml...")

        config_output_path = output_dir / "labeling_config.yaml"

        # 如果提供了 config_path，直接加载
        if config_path is not None:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Provided config file not found: {config_path}")

            logger.info(f"Loading existing config from: {config_path}")
            import yaml
            with open(config_path) as f:
                labeling_config = yaml.safe_load(f)

            # 复制到输出目录（如果不同）
            if config_path.resolve() != config_output_path.resolve():
                import shutil
                shutil.copy(config_path, config_output_path)
                logger.info(f"Copied config to: {config_output_path}")

        elif use_presets and (
            preset_path := find_preset_file(
                requirements_path, "labeling_config.yaml", preset_dir=preset_dir
            )
        ):
            logger.info(f"Using preset labeling config: {preset_path}")
            import yaml
            with open(preset_path, encoding="utf-8") as f:
                labeling_config = yaml.safe_load(f)
            import shutil
            shutil.copy(preset_path, config_output_path)
            logger.info(f"Copied preset config to: {config_output_path}")

        # 如果输出目录已有配置，复用
        elif config_output_path.exists():
            logger.info(f"Found existing config at: {config_output_path}")
            logger.info("Reusing existing config (to regenerate, delete this file or use --force-regenerate)")
            import yaml
            with open(config_output_path, encoding="utf-8") as f:
                labeling_config = yaml.safe_load(f)

        # 否则生成新配置
        else:
            logger.info("Generating new config via LLM...")
            generator = LabelingConfigGenerator()
            labeling_config = generator.generate_from_requirements(
                requirements_path=str(requirements_path),
                output_path=str(config_output_path)
            )

        labeling_config = self._apply_embedding_env_overrides(
            labeling_config, config_output_path
        )

        logger.info(f"✓ Config at: {config_output_path}")
        logger.info(f"  Fields: {[f['field_name'] for f in labeling_config['fields']]}")

        # ========== Step 2: 构建向量库 ==========
        logger.info("\n[Step 2] Building vector store...")

        chroma_dir = output_dir / "chroma" / domain_name
        chroma_dir.mkdir(parents=True, exist_ok=True)

        vector_builder = VectorStoreBuilder(
            embedding_config=labeling_config["embedding"],
            persist_dir=str(chroma_dir)
        )

        vectorstore = vector_builder.build_or_update_from_chunks(str(parsed_chunks_path))

        logger.info(f"✓ Vector store at: {chroma_dir}")
        logger.info(f"  Total chunks: {len(vector_builder.chunk_map)}")

        # ========== Step 3: 加载所有 chunks ==========
        logger.info("\n[Step 3] Loading parsed chunks...")

        all_chunks = self._read_jsonl(parsed_chunks_path)

        # 按 paper_id 分组
        papers_map = {}
        for chunk in all_chunks:
            paper_id = chunk["paper_id"]
            if paper_id not in papers_map:
                papers_map[paper_id] = []
            papers_map[paper_id].append(chunk)

        logger.info(f"✓ Loaded {len(all_chunks)} chunks from {len(papers_map)} papers")

        # ========== Step 4: 初始化检索器和标注器 ==========
        logger.info("\n[Step 4] Initializing retrievers and labeler...")

        semantic_retriever = SemanticRetriever(vectorstore)
        llm_binary_confirm_default = self._labeling_llm_binary_confirm_default(
            labeling_config
        )
        uses_llm_binary_confirm = any(
            self._field_llm_binary_confirm(field, llm_binary_confirm_default)
            for field in labeling_config["fields"]
        )
        if uses_llm_binary_confirm:
            from .tools.dspy_evidence_labeler import EvidenceLabeler
            evidence_labeler = EvidenceLabeler(vector_builder)
        else:
            evidence_labeler = None

        logger.info(
            "✓ Retrievers ready; LLM binary confirmation: %s",
            "enabled" if uses_llm_binary_confirm else "disabled",
        )

        # ========== Step 5: 对每篇论文、每个字段进行标注 ==========
        logger.info("\n[Step 5] Labeling chunks...")

        labeled_chunks_path = output_dir / "labeled_chunks.jsonl"
        summary_path = output_dir / "labeling_summary.json"
        existing_main_rows = (
            self._read_jsonl_if_exists(labeled_chunks_path) if resume else []
        )
        existing_summary = (
            self._read_json_if_exists(summary_path) if resume else {}
        )
        processed_paper_ids = set(existing_summary.get("processed_papers") or [])
        if resume and not processed_paper_ids:
            processed_paper_ids = {
                str(row.get("paper_id", "")).strip()
                for row in existing_main_rows
                if row.get("paper_id")
            }

        # 收集所有 paper-field 的标注结果（保留原始 item，供聚合使用）
        raw_labeled: list[dict] = []

        paper_items = list(papers_map.items())
        skipped_existing = 0
        if resume and processed_paper_ids:
            pending_items = []
            for paper_id, paper_chunks in paper_items:
                if paper_id in processed_paper_ids:
                    skipped_existing += 1
                    continue
                pending_items.append((paper_id, paper_chunks))
        else:
            pending_items = paper_items
        if limit:
            pending_items = pending_items[:limit]

        total_papers = len(pending_items)
        total_fields = len(labeling_config["fields"])
        processed_this_run_ids: list[str] = []

        logger.info(
            "  Papers to label: %d (%d already labeled)",
            total_papers, skipped_existing,
        )

        for paper_idx, (paper_id, paper_chunks) in enumerate(pending_items, 1):
            logger.info(f"\n  Paper [{paper_idx}/{total_papers}]: {paper_id}")
            processed_this_run_ids.append(paper_id)

            for field_idx, field_config in enumerate(labeling_config["fields"], 1):
                field_name = field_config["field_name"]
                logger.info(f"    Field [{field_idx}/{total_fields}]: {field_name}")

                # 处理单个 paper-field 组合（双通道）—— 判断逻辑不变
                labeled = self._process_paper_field_labeling(
                    paper_id=paper_id,
                    paper_chunks=paper_chunks,
                    field_config=field_config,
                    semantic_retriever=semantic_retriever,
                    vector_builder=vector_builder,
                    evidence_labeler=evidence_labeler,
                    llm_binary_confirm_default=llm_binary_confirm_default,
                )

                # 仅保留 relevant=true 的记录，附带 paper_id / field_name
                for item in labeled:
                    if not item.get("relevant"):
                        continue
                    raw_labeled.append({
                        "paper_id": paper_id,
                        "field_name": field_name,
                        "chunk_id": item["chunk_id"],
                        "chunk_type": item["chunk_type"],
                    })

        # 按 chunk 聚合成一行一 chunk 的主输出
        new_main_rows = self._aggregate_labeled_chunks(
            raw_labeled, labeling_config, vector_builder
        )
        processed_this_run_set = set(processed_this_run_ids)
        main_rows = [
            row for row in existing_main_rows
            if row.get("paper_id") not in processed_this_run_set
        ] + new_main_rows
        main_rows.sort(
            key=lambda x: (
                x["paper_id"],
                x["chunk_index"] if x.get("chunk_index") is not None else float("inf"),
                x["chunk_id"],
            )
        )

        # ========== Step 6: 保存输出 ==========
        logger.info(f"\n[Step 6] Saving labeled chunks...")

        self._write_jsonl(labeled_chunks_path, main_rows)

        logger.info(f"✓ Saved {len(main_rows)} labeled chunks to: {labeled_chunks_path}")

        # ========== Step 7: 生成摘要 ==========
        logger.info("\n[Step 7] Generating summary...")

        processed_papers_all = sorted(
            (processed_paper_ids & set(papers_map)) | processed_this_run_set
        )
        summary = self._generate_labeling_summary(
            main_rows,
            labeling_config,
            processed_papers=processed_papers_all,
        )
        summary.update({
            "processed_this_run": len(processed_this_run_ids),
            "skipped_existing": skipped_existing,
        })

        with open(summary_path, 'w', encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Summary saved to: {summary_path}")

        logger.info("\n" + "=" * 60)
        logger.info("Labeling Stage - COMPLETED")
        logger.info("=" * 60)

        return {
            "labeled_chunks_path": str(labeled_chunks_path),
            "config_path": str(config_output_path),
            "summary_path": str(summary_path),
            **summary,
        }

    def _process_paper_field_labeling(
        self,
        paper_id: str,
        paper_chunks: list[dict],
        field_config: dict,
        semantic_retriever,
        vector_builder,
        evidence_labeler,
        llm_binary_confirm_default: bool = False,
    ) -> list[dict]:
        """
        双通道处理：text channel + table channel

        Args:
            paper_id: 论文 ID
            paper_chunks: 该论文的所有 chunks
            field_config: 字段配置
            semantic_retriever: SemanticRetriever 实例
            vector_builder: VectorStoreBuilder 实例
            evidence_labeler: EvidenceLabeler 实例

        Returns:
            labeled chunks 列表
        """
        from .tools.hybrid_retriever import (
            SectionFilter,
            RegexRetriever,
            TableHeaderRetriever,
            RRFFusion,
        )

        # Section 过滤
        section_filter = SectionFilter(
            include=field_config.get("section_include"),
            exclude=field_config.get("section_exclude")
        )
        filtered = section_filter.filter_chunks(paper_chunks)

        logger.info(
            f"      Section filter: total={len(paper_chunks)}, "
            f"excluded={len(filtered['excluded'])}, "
            f"preferred={len(filtered['preferred'])}, "
            f"fallback={len(filtered['fallback'])}"
        )

        all_labeled = []
        settings = field_config["retrieval_settings"]
        llm_binary_confirm = self._field_llm_binary_confirm(
            field_config, llm_binary_confirm_default
        )

        # ========== A. Text Channel ==========
        text_chunks = [
            c for c in filtered["allowed"]
            if c["chunk_type"] in ["paragraph", "abstract"]
        ]

        if text_chunks:
            logger.info(f"      [Text Channel] Processing {len(text_chunks)} text chunks")

            # 软偏好：优先在 preferred 中检索，如果不够再用 allowed
            preferred_text = [c for c in filtered["preferred"] if c["chunk_type"] in ["paragraph", "abstract"]]

            # Semantic retrieval（优先池）
            if preferred_text:
                logger.info(f"        Using preferred pool: {len(preferred_text)} chunks")
                semantic_pool_ids = set([c["chunk_id"] for c in preferred_text])
            else:
                logger.info(f"        Preferred pool empty, using all allowed: {len(text_chunks)} chunks")
                semantic_pool_ids = set([c["chunk_id"] for c in text_chunks])

            semantic_results = semantic_retriever.retrieve(
                query=field_config["semantic_query"],
                paper_id=paper_id,
                allowed_chunk_ids=semantic_pool_ids,
                fetch_k=settings["semantic_fetch_k"]
            )

            # 如果 semantic 结果不够，用 allowed pool 补充
            if len(semantic_results) < settings["text_top_k"] and len(text_chunks) > len(preferred_text):
                logger.info(f"        Semantic results insufficient ({len(semantic_results)}), fallback to allowed pool")
                semantic_results = semantic_retriever.retrieve(
                    query=field_config["semantic_query"],
                    paper_id=paper_id,
                    allowed_chunk_ids=set([c["chunk_id"] for c in text_chunks]),
                    fetch_k=settings["semantic_fetch_k"]
                )

            # Regex retrieval（在全部 allowed 中检索）
            regex_retriever = RegexRetriever(field_config["regex_patterns"])
            regex_results = regex_retriever.retrieve(
                chunks=text_chunks,
                fetch_k=settings["regex_fetch_k"]
            )

            # RRF 融合
            rrf_fusion = RRFFusion(k=settings["rrf_k"])
            text_top_k = rrf_fusion.merge(
                ranked_lists=[semantic_results, regex_results],
                top_k=settings["text_top_k"]
            )

            logger.info(
                f"        Semantic: {len(semantic_results)}, "
                f"Regex: {len(regex_results)} → RRF top-{len(text_top_k)}"
            )

            # Optional LLM 二分类；默认关闭时直接保留 retrieval top-k。
            if text_top_k:
                if llm_binary_confirm:
                    labeled_text = evidence_labeler.label_candidates(
                        field_config=field_config,
                        candidates=text_top_k,
                        channel="text"
                    )
                else:
                    labeled_text = self._mark_candidates_relevant(
                        text_top_k, channel="text"
                    )
                all_labeled.extend(labeled_text)
                relevant_count = sum(1 for x in labeled_text if x["relevant"])
                label_mode = "LLM-confirmed" if llm_binary_confirm else "retrieval-only"
                logger.info(
                    f"        Labeled ({label_mode}): "
                    f"{relevant_count}/{len(labeled_text)} relevant"
                )

        # ========== B. Table Channel ==========
        table_chunks = [
            c for c in filtered["allowed"]
            if c["chunk_type"] == "table"
        ]

        if table_chunks:
            table_top_k = settings.get("table_top_k", 5)
            table_fetch_k = settings.get("table_fetch_k", 10)
            logger.info(f"      [Table Channel] Processing {len(table_chunks)} table chunks")

            preferred_tables = [
                c for c in filtered["preferred"]
                if c["chunk_type"] == "table"
            ]
            table_signal_chunks = (
                preferred_tables
                if preferred_tables and len(table_chunks) > table_top_k
                else table_chunks
            )
            if preferred_tables and len(table_chunks) > table_top_k:
                logger.info(
                    f"        Using preferred table pool: {len(preferred_tables)} chunks "
                    f"(fallback to allowed if insufficient)"
                )

            # --- 步骤 1：计算 table 检索信号；表格多时优先 preferred section ---
            retrieval_lists = []

            # 1a. Table header keyword 匹配
            table_header_retriever = TableHeaderRetriever(field_config["table_header_keywords"])
            table_header_results = table_header_retriever.retrieve(
                chunks=table_signal_chunks,
                fetch_k=table_fetch_k
            )
            if table_header_results:
                retrieval_lists.append(table_header_results)
            logger.info(f"        Table header matched: {len(table_header_results)}")

            # 1b. Table semantic retrieval
            table_semantic_results = semantic_retriever.retrieve(
                query=field_config["semantic_query"],
                paper_id=paper_id,
                allowed_chunk_ids=set(c["chunk_id"] for c in table_signal_chunks),
                fetch_k=table_fetch_k
            )
            if table_semantic_results:
                retrieval_lists.append(table_semantic_results)
            logger.info(f"        Table semantic: {len(table_semantic_results)}")

            # 1c. Table regex（如果有 patterns）
            table_regex_results = []
            if field_config.get("regex_patterns"):
                table_regex_retriever = RegexRetriever(field_config["regex_patterns"])
                table_regex_results = table_regex_retriever.retrieve(
                    chunks=table_signal_chunks,
                    fetch_k=table_fetch_k
                )
                if table_regex_results:
                    retrieval_lists.append(table_regex_results)
                logger.info(f"        Table regex: {len(table_regex_results)}")

            # --- 步骤 2：RRF 融合，生成带完整信号的 ranked list ---
            # fetch_k=len(table_chunks) 确保 RRF 能覆盖所有表格（补全逻辑需要）
            if retrieval_lists:
                rrf_fusion = RRFFusion(k=settings["rrf_k"])
                rrf_ranked = rrf_fusion.merge(
                    ranked_lists=retrieval_lists,
                    top_k=len(table_chunks)  # 先拿所有，后面再截断
                )
            else:
                rrf_ranked = []

            if (
                preferred_tables
                and len(table_chunks) > table_top_k
                and len(rrf_ranked) < table_top_k
                and len(table_signal_chunks) < len(table_chunks)
            ):
                logger.info(
                    f"        Preferred table results insufficient ({len(rrf_ranked)}), "
                    "fallback to all allowed tables"
                )
                fallback_lists = []

                table_header_results = table_header_retriever.retrieve(
                    chunks=table_chunks,
                    fetch_k=table_fetch_k
                )
                if table_header_results:
                    fallback_lists.append(table_header_results)

                table_semantic_results = semantic_retriever.retrieve(
                    query=field_config["semantic_query"],
                    paper_id=paper_id,
                    allowed_chunk_ids=set(c["chunk_id"] for c in table_chunks),
                    fetch_k=table_fetch_k
                )
                if table_semantic_results:
                    fallback_lists.append(table_semantic_results)

                if field_config.get("regex_patterns"):
                    table_regex_results = table_regex_retriever.retrieve(
                        chunks=table_chunks,
                        fetch_k=table_fetch_k
                    )
                    if table_regex_results:
                        fallback_lists.append(table_regex_results)

                if fallback_lists:
                    rrf_fusion = RRFFusion(k=settings["rrf_k"])
                    rrf_ranked = rrf_fusion.merge(
                        ranked_lists=fallback_lists,
                        top_k=len(table_chunks)
                    )

            # --- 步骤 3：根据表格数量决定最终 candidates ---
            rrf_ids = {item["chunk_id"] for item in rrf_ranked}

            if len(table_chunks) <= table_top_k:
                # 表格少：如果有 LLM 审核，可以全送；retrieval-only 时只保留有检索信号的表格。
                table_candidates = list(rrf_ranked)  # 已有信号的排在前面

                if llm_binary_confirm:
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

                logger.info(
                    f"        Table count ({len(table_chunks)}) ≤ table_top_k ({table_top_k}): "
                    f"{'labeling all' if llm_binary_confirm else 'retrieval-only'} "
                    f"{len(table_candidates)} tables "
                    f"({len(rrf_ranked)} with signals, {len(table_candidates) - len(rrf_ranked)} appended)"
                )
            else:
                # 表格多：只取 RRF top-k
                table_candidates = rrf_ranked[:table_top_k]
                logger.info(
                    f"        Table count ({len(table_chunks)}) > table_top_k ({table_top_k}): "
                    f"RRF top-{len(table_candidates)}"
                )

            # --- 步骤 4：Optional LLM 二分类 ---
            if table_candidates:
                if llm_binary_confirm:
                    labeled_tables = evidence_labeler.label_candidates(
                        field_config=field_config,
                        candidates=table_candidates,
                        channel="table"
                    )
                else:
                    labeled_tables = self._mark_candidates_relevant(
                        table_candidates, channel="table"
                    )
                all_labeled.extend(labeled_tables)
                relevant_count = sum(1 for x in labeled_tables if x["relevant"])
                label_mode = "LLM-confirmed" if llm_binary_confirm else "retrieval-only"
                logger.info(
                    f"        Labeled ({label_mode}): "
                    f"{relevant_count}/{len(labeled_tables)} relevant"
                )

        return all_labeled

    @staticmethod
    def _labeling_llm_binary_confirm_default(config: dict) -> bool:
        """Return the global default for optional LLM binary confirmation."""

        strategy = config.get("labeling_strategy") or {}
        if "llm_binary_confirm" in strategy:
            return bool(strategy["llm_binary_confirm"])
        labeler = config.get("labeler") or {}
        if "llm_binary_confirm" in labeler:
            return bool(labeler["llm_binary_confirm"])
        return False

    @staticmethod
    def _field_llm_binary_confirm(
        field_config: dict,
        default: bool = False,
    ) -> bool:
        """Allow field-level override while keeping the global default light."""

        if "llm_binary_confirm" in field_config:
            return bool(field_config["llm_binary_confirm"])
        settings = field_config.get("retrieval_settings") or {}
        if "llm_binary_confirm" in settings:
            return bool(settings["llm_binary_confirm"])
        return default

    @staticmethod
    def _mark_candidates_relevant(
        candidates: list[dict],
        channel: str,
    ) -> list[dict]:
        """Label retrieval candidates as relevant without calling an LLM."""

        return [
            {
                **candidate,
                "relevant": True,
                "retrieval_channel": channel,
            }
            for candidate in candidates
        ]

    @staticmethod
    def _aggregate_labeled_chunks(
        raw_labeled: list[dict],
        config: dict,
        vector_builder,
    ) -> list[dict]:
        """
        把 (paper_id, field_name, chunk_id) 级别的 relevant 记录聚合成
        「一个 chunk 一行」的主输出。

        Args:
            raw_labeled: 每条含 paper_id / field_name / chunk_id / chunk_type，
                         调用方应已过滤只保留 relevant=true。
            config: labeling_config，用于确定 labels 的排序顺序。
            vector_builder: 提供 get_chunk(chunk_id) 以读取
                            section_path 和 metadata.chunk_index。

        Returns:
            list[dict]，每行仅含：
            paper_id / chunk_id / chunk_index / chunk_type / section_path / labels
            只返回 labels 非空的 chunk。
        """
        field_order = [f["field_name"] for f in config["fields"]]

        # chunk_id -> 主输出记录
        labeled_map: dict[str, dict] = {}

        for rec in raw_labeled:
            chunk_id = rec["chunk_id"]
            field_name = rec["field_name"]

            if chunk_id not in labeled_map:
                parsed_chunk = vector_builder.get_chunk(chunk_id) or {}
                chunk_index = parsed_chunk.get("metadata", {}).get("chunk_index")
                labeled_map[chunk_id] = {
                    "paper_id": rec["paper_id"],
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "chunk_type": rec["chunk_type"],
                    "section_path": parsed_chunk.get("section_path", []),
                    "labels": [],
                }

            if field_name not in labeled_map[chunk_id]["labels"]:
                labeled_map[chunk_id]["labels"].append(field_name)

        # labels 按 config field 顺序排序（保证确定性）
        def _label_key(f: str) -> int:
            return field_order.index(f) if f in field_order else len(field_order)

        main_rows = []
        for row in labeled_map.values():
            if not row["labels"]:
                continue
            row["labels"].sort(key=_label_key)
            main_rows.append(row)

        # 按 (paper_id, chunk_index, chunk_id) 稳定排序
        main_rows.sort(
            key=lambda x: (
                x["paper_id"],
                x["chunk_index"] if x["chunk_index"] is not None else float("inf"),
                x["chunk_id"],
            )
        )
        return main_rows

    @staticmethod
    def _generate_labeling_summary(
        main_rows: list[dict],
        config: dict,
        processed_papers: list[str] | None = None,
    ) -> dict:
        """生成标注摘要统计（基于 chunk 聚合后的主输出）"""

        total_labeled_chunks = len(main_rows)
        total_label_assignments = sum(len(r["labels"]) for r in main_rows)

        # 每个 field 命中了多少 chunks（含每篇 paper 的分布）
        by_field: dict[str, dict] = {}
        # 每篇 paper 命中多少 chunks / label assignments
        by_paper: dict[str, dict] = {}

        for record in main_rows:
            paper_id = record["paper_id"]
            labels = record["labels"]

            if paper_id not in by_paper:
                by_paper[paper_id] = {"labeled_chunks": 0, "label_assignments": 0}
            by_paper[paper_id]["labeled_chunks"] += 1
            by_paper[paper_id]["label_assignments"] += len(labels)

            for field in labels:
                if field not in by_field:
                    by_field[field] = {"chunks": 0, "papers": {}}
                by_field[field]["chunks"] += 1
                by_field[field]["papers"][paper_id] = (
                    by_field[field]["papers"].get(paper_id, 0) + 1
                )

        return {
            "total_labeled_chunks": total_labeled_chunks,
            "total_label_assignments": total_label_assignments,
            "total_papers_processed": len(processed_papers or by_paper),
            "processed_papers": processed_papers or sorted(by_paper),
            "by_field": by_field,
            "by_paper": by_paper,
            "config_fields": [f["field_name"] for f in config["fields"]],
            "embedding_model": config["embedding"]["model"],
            "labeler_model": config["labeler"]["model"],
        }

    @staticmethod
    def _apply_embedding_env_overrides(config: dict, config_path: Path) -> dict:
        """Apply EMBEDDING_* env settings to a labeling config and persist them."""
        import os
        import yaml
        from dotenv import load_dotenv

        load_dotenv()

        model = os.getenv("EMBEDDING_MODEL")
        if not model:
            return config

        provider = os.getenv("EMBEDDING_PROVIDER")
        if not provider:
            provider = "gemini" if model.startswith("gemini-") else "openai"

        embedding_config = config.setdefault("embedding", {})
        old_embedding = dict(embedding_config)
        embedding_config["model"] = model
        embedding_config["provider"] = provider

        if embedding_config != old_embedding:
            with open(config_path, "w", encoding="utf-8") as fh:
                yaml.dump(config, fh, allow_unicode=True, sort_keys=False)
            logger.info("Applied EMBEDDING_* overrides to labeling_config.yaml")

        return config

    # ================================================================== #
    #  Stage 2: Extraction                                                #
    # ================================================================== #

    def run_extraction(
        self,
        requirements_path: str | Path,
        parsed_chunks_path: str | Path,
        labeled_chunks_path: str | Path,
        output_dir: str | Path,
        *,
        model_name: str | None = None,
        prompt_preset_path: str | Path | None = None,
        preset_dir: str | Path | None = None,
        use_presets: bool = True,
        limit: int | None = None,
        resume: bool = True,
    ) -> dict:
        """
        Stage 2 Extraction 主流程。

        Flow（对应 ALLMAT extract_lc.py 的 extract() 函数）：
          1. 加载 user_requirements、parsed_chunks、labeled_chunks
          2. 按 paper_id 组织数据
          3. 对每篇 paper：
             a. context_builder — 收集 labeled + abstract chunks，按原文顺序拼接
             b. extraction_schema — 动态生成 Records 模型 + instruction
             c. extractor — LLM with_structured_output 一次性抽 records
             d. record_cleanup — 同 paper 内严格去重 + 分配 record_id
          4. 输出 extracted_records.jsonl + extraction_summary.json

        与 ALLMAT 的当前有意偏离之一：
          - 暂不实现 DetectProcesses / 模板注入（保留扩展位置于
            extraction_schema.create_instruction() 末尾）
        """
        from datetime import datetime, timezone

        from .user_requirements import load_user_requirements
        from .extraction_schema import (
            build_records_model,
            build_system_message,
            create_instruction,
        )
        from .preset_manager import find_preset_file, render_prompt_template
        from .tools.context_builder import build_context
        from .tools.extractor import extract_records
        from .tools.record_cleanup import deduplicate, assign_ids_and_source
        from .tools.endpoint_constraints import build_endpoint_constraint

        requirements_path   = Path(requirements_path)
        parsed_chunks_path  = Path(parsed_chunks_path)
        labeled_chunks_path = Path(labeled_chunks_path)
        output_dir          = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Extraction Stage")
        logger.info("=" * 60)

        # ── Step 1: 加载 requirements ──
        logger.info("\n[Step 1] Loading user requirements...")
        req = load_user_requirements(requirements_path)
        assert req.record is not None, "user_requirements.yaml must have 'record' block"
        field_names = [f.name for f in req.record.fields]
        logger.info(f"  Record: {req.record.name}")
        logger.info(f"  Fields: {field_names}")

        records_model   = build_records_model(req.record.fields)
        system_message  = build_system_message(req.record)
        instruction     = create_instruction(req.record)

        prompt_preset = None
        if prompt_preset_path is not None:
            prompt_preset = Path(prompt_preset_path)
            if not prompt_preset.exists():
                raise FileNotFoundError(f"Extraction prompt preset not found: {prompt_preset}")
        elif use_presets:
            prompt_preset = find_preset_file(
                requirements_path, "extraction_prompt.yaml", preset_dir=preset_dir
            )

        if prompt_preset:
            import yaml
            logger.info(f"  Using extraction prompt preset: {prompt_preset}")
            with open(prompt_preset, encoding="utf-8") as fh:
                prompt_data = yaml.safe_load(fh) or {}
            if prompt_data.get("system_message"):
                system_message = render_prompt_template(
                    str(prompt_data["system_message"]), req.record
                )
            if prompt_data.get("instruction"):
                instruction = render_prompt_template(
                    str(prompt_data["instruction"]), req.record
                )

        # Load endpoint constraints (optional, strict mode for pancan preset)
        endpoint_constraint = None
        if prompt_preset:
            import yaml
            with open(prompt_preset, encoding="utf-8") as fh:
                prompt_data = yaml.safe_load(fh) or {}
            endpoint_constraint = build_endpoint_constraint(
                prompt_data.get("endpoint_constraints")
            )
            if endpoint_constraint:
                logger.info("  Endpoint constraints: strict mode enabled")

        # ── Step 2: 加载 chunks ──
        logger.info("\n[Step 2] Loading chunks...")
        all_parsed  = self._read_jsonl(parsed_chunks_path)
        all_labeled = self._read_jsonl(labeled_chunks_path)

        chunk_store: dict[str, dict] = {c["chunk_id"]: c for c in all_parsed}
        labeled_map: dict[str, set[str]] = {}
        for row in all_labeled:
            labeled_map.setdefault(row["paper_id"], set()).add(row["chunk_id"])

        paper_ids = list(labeled_map.keys())
        logger.info(f"  {len(all_parsed)} chunks, {len(paper_ids)} papers")

        # ── Step 3: 逐篇抽取 ──
        output_records_path = output_dir / "extracted_records.jsonl"
        summary_path = output_dir / "extraction_summary.json"
        existing_records = (
            self._read_jsonl_if_exists(output_records_path) if resume else []
        )
        existing_summary = (
            self._read_json_if_exists(summary_path) if resume else {}
        )
        existing_by_paper = existing_summary.get("by_paper", {})
        if not isinstance(existing_by_paper, dict):
            existing_by_paper = {}

        completed_statuses = ("ok", "skipped:no_context")
        completed_paper_ids = {
            paper_id for paper_id, info in existing_by_paper.items()
            if str(info.get("extraction_status", "")).startswith(completed_statuses)
        }
        pending_paper_ids = [
            paper_id for paper_id in paper_ids
            if not (resume and paper_id in completed_paper_ids)
        ]
        skipped_existing = len(paper_ids) - len(pending_paper_ids)
        if limit:
            pending_paper_ids = pending_paper_ids[:limit]

        logger.info(
            f"\n[Step 3] Extracting from {len(pending_paper_ids)} papers "
            f"({skipped_existing} already complete)..."
        )

        pending_set = set(pending_paper_ids)
        all_records = [
            row for row in existing_records
            if row.get("paper_id") not in pending_set
        ]
        by_paper = {
            paper_id: info
            for paper_id, info in existing_by_paper.items()
            if paper_id in paper_ids and paper_id not in pending_set
        }

        for idx, paper_id in enumerate(pending_paper_ids, 1):
            logger.info(f"\n  [{idx}/{len(pending_paper_ids)}] {paper_id}")

            context_str, used_ids = build_context(
                paper_id, chunk_store, labeled_map[paper_id]
            )
            logger.info(f"    Context: {len(used_ids)} chunks, {len(context_str)} chars")

            if not context_str:
                logger.warning(f"    No context, skipping")
                by_paper[paper_id] = {
                    "extraction_status": "skipped:no_context",
                    "records_raw": 0,
                    "records_after_cleanup": 0,
                    "duplicates_removed": 0,
                    "endpoint_constraint_rejected": 0,
                    "context_chunks_used": 0,
                }
                continue

            raw_records, status = extract_records(
                context_str, system_message, instruction, records_model, model_name=model_name
            )
            logger.info(f"    Status: {status}, raw: {len(raw_records)}")

            # Canonicalize/filter endpoint pairs before deduplication, so aliases
            # such as "overall survival" and "OS" collapse to one record.
            constraint_rejected = 0
            constraint_stats = {}
            if endpoint_constraint:
                raw_records, constraint_stats = endpoint_constraint.apply(raw_records)
                constraint_rejected = constraint_stats.get("endpoint_constraint_rejected", 0)
                if constraint_rejected > 0:
                    logger.info(f"    Endpoint constraints rejected: {constraint_rejected}")

            deduped, n_removed = deduplicate(raw_records, field_names)

            cleaned = assign_ids_and_source(deduped, paper_id, field_names, used_ids)

            logger.info(f"    Cleaned: {len(cleaned)} ({n_removed} dedup removed)")

            all_records.extend(cleaned)
            by_paper[paper_id] = {
                "extraction_status": status,
                "records_raw": len(raw_records),
                "records_after_cleanup": len(cleaned),
                "duplicates_removed": n_removed,
                "endpoint_constraint_rejected": constraint_rejected,
                "endpoint_constraint_rejected_by_combo": constraint_stats.get(
                    "endpoint_constraint_rejected_by_combo", {}
                ),
                "context_chunks_used": len(used_ids),
            }

            # Keep completed papers available if a later LLM request is slow or
            # the user interrupts a long batch. The final write below has the
            # same format, so downstream readers need no special handling.
            self._write_jsonl(output_records_path, all_records)
            logger.info(f"    Checkpoint saved: {len(all_records)} records")

        # ── Step 4: 输出 ──
        logger.info(f"\n[Step 4] Writing outputs...")

        self._write_jsonl(output_records_path, all_records)
        logger.info(f"  ✓ {len(all_records)} records")

        import os as _os
        total_removed = sum(
            v.get("duplicates_removed", 0) for v in by_paper.values()
        )
        total_constraint_rejected = sum(
            v.get("endpoint_constraint_rejected", 0) for v in by_paper.values()
        )
        total_constraint_rejected_by_combo: dict[str, int] = {}
        for paper_summary in by_paper.values():
            for combo, count in paper_summary.get(
                "endpoint_constraint_rejected_by_combo", {}
            ).items():
                total_constraint_rejected_by_combo[combo] = (
                    total_constraint_rejected_by_combo.get(combo, 0) + count
                )
        summary = {
            "total_papers_processed": len(paper_ids),
            "total_papers_failed": sum(
                1 for v in by_paper.values()
                if not str(v.get("extraction_status", "")).startswith(completed_statuses)
            ),
            "total_records_extracted": len(all_records),
            "duplicates_removed": total_removed,
            "endpoint_constraint_rejected": total_constraint_rejected,
            "endpoint_constraint_rejected_by_combo": total_constraint_rejected_by_combo,
            "by_paper": by_paper,
            "processed_this_run": len(pending_paper_ids),
            "skipped_existing": skipped_existing,
            "extractor_model": (
                model_name
                or _os.environ.get("EXTRACTOR_MODEL")
                or _os.environ.get("LLM_MODEL", "unknown")
            ),
            "prompt_preset": str(prompt_preset) if prompt_preset else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        logger.info(f"  ✓ extraction_summary.json")

        logger.info("\n" + "=" * 60)
        logger.info("Extraction Stage - COMPLETED")
        logger.info("=" * 60)

        return summary

    # ================================================================== #
    #  Stage 3: Post-processing                                           #
    # ================================================================== #

    def run_postprocess(
        self,
        requirements_path: str | Path,
        extracted_records_path: str | Path,
        output_dir: str | Path,
        *,
        config_path: str | Path | None = None,
        preset_dir: str | Path | None = None,
        use_presets: bool = True,
    ) -> dict:
        """
        Stage 3 Post-processing 主流程。

        Flow:
          1. 加载 user_requirements、extracted_records
          2. 读取 postprocess_config.yaml（显式 config > preset > 空配置）
          3. 通用线路：空值处理、数值解析、严格去重、CSV 导出
          4. 领域 preset 线路：同义词/标准词表/单位规则/有效性规则
          5. 输出 postprocessed_records.jsonl + records.csv + summary

        对应 ALLMAT 的工程思想：
          - 用规则 normalizer 统一实体字段
          - 用严格 partition/merge 去重
          - fuzzy/LLM entity resolution 留作后续扩展
        """
        from datetime import datetime, timezone

        import yaml

        from .preset_manager import find_preset_file
        from .tools.postprocess import (
            load_postprocess_config,
            postprocess_records,
            resolve_numeric_fields,
            write_records_csv,
        )
        from .user_requirements import load_user_requirements

        requirements_path = Path(requirements_path)
        extracted_records_path = Path(extracted_records_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Post-processing Stage")
        logger.info("=" * 60)

        logger.info("\n[Step 1] Loading user requirements and records...")
        req = load_user_requirements(requirements_path)
        assert req.record is not None, "user_requirements.yaml must have 'record' block"
        raw_records = self._read_jsonl(extracted_records_path)
        logger.info(f"  Input records: {len(raw_records)}")

        logger.info("\n[Step 2] Loading postprocess config...")
        used_config_path = None
        if config_path is not None:
            used_config_path = Path(config_path)
        elif use_presets:
            used_config_path = find_preset_file(
                requirements_path, "postprocess_config.yaml", preset_dir=preset_dir
            )

        config = load_postprocess_config(used_config_path)
        if used_config_path:
            logger.info(f"  Using config: {used_config_path}")
        else:
            logger.info("  No config found; using generic defaults")

        logger.info("\n[Step 3] Cleaning, standardizing, and deduplicating...")
        records, counts = postprocess_records(raw_records, req.record.fields, config)
        numeric_fields = resolve_numeric_fields(req.record.fields, config)
        logger.info(f"  Output records: {len(records)}")

        logger.info("\n[Step 4] Writing outputs...")
        self._write_jsonl(output_dir / "postprocessed_records.jsonl", records)
        write_records_csv(output_dir / "records.csv", records, req.record.fields, numeric_fields)

        with open(output_dir / "postprocess_config.yaml", "w", encoding="utf-8") as fh:
            yaml.dump(config, fh, allow_unicode=True, sort_keys=False)

        summary = {
            **counts,
            "requirements_path": str(requirements_path),
            "input_path": str(extracted_records_path),
            "postprocess_config": str(used_config_path) if used_config_path else None,
            "output_jsonl": str(output_dir / "postprocessed_records.jsonl"),
            "output_csv": str(output_dir / "records.csv"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(output_dir / "postprocessing_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)

        logger.info("  ✓ postprocessed_records.jsonl")
        logger.info("  ✓ records.csv")
        logger.info("  ✓ postprocessing_summary.json")
        logger.info("\n" + "=" * 60)
        logger.info("Post-processing Stage - COMPLETED")
        logger.info("=" * 60)

        return summary
