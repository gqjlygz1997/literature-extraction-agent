"""Parser for Web of Science tagged plain-text exports.

Web of Science "savedrecs.txt" files are bibliographic metadata, not full text.
This module converts them into a JSONL-friendly candidate-paper format that can
be sent through the paper filter before full-text XML acquisition.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


FIELD_RE = re.compile(r"^([A-Z0-9]{2})\s(.*)$")


LIST_FIELDS = {"AU", "AF", "DE", "ID", "C1", "C3", "EM", "OI", "RI", "WC", "SC"}


def parse_wos_file(path: str | Path) -> list[dict]:
    """Parse one Web of Science savedrecs text file."""

    path = Path(path)
    records: list[dict] = []
    current: dict[str, list[str]] = {}
    current_field: str | None = None

    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n\r")
            if not line.strip():
                continue

            if line == "ER":
                if current:
                    records.append(_normalize_record(current, source_file=path))
                current = {}
                current_field = None
                continue

            match = FIELD_RE.match(line)
            if match:
                field, value = match.groups()
                current.setdefault(field, []).append(value.strip())
                current_field = field
                continue

            if current_field and line.startswith(" "):
                continuation = line.strip()
                if continuation:
                    current[current_field].append(continuation)

    if current:
        records.append(_normalize_record(current, source_file=path))

    return records


def parse_wos_files(paths: Iterable[str | Path]) -> list[dict]:
    """Parse and deduplicate multiple savedrecs files."""

    seen: set[tuple[str, str]] = set()
    rows: list[dict] = []
    for path in paths:
        for row in parse_wos_file(path):
            key = _dedupe_key(row)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def discover_wos_files(inputs: Iterable[str | Path]) -> list[Path]:
    """Expand input files/directories into sorted .txt paths."""

    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.txt")))
        elif path.exists():
            paths.append(path)
        else:
            raise FileNotFoundError(f"WOS input not found: {path}")
    return sorted(dict.fromkeys(paths))


def _normalize_record(fields: dict[str, list[str]], *, source_file: Path) -> dict:
    title = _text(fields, "TI")
    abstract = _text(fields, "AB")
    doi = _text(fields, "DI")
    pmid = _text(fields, "PM")
    wos_uid = _text(fields, "UT")

    paper_id = _make_paper_id(wos_uid=wos_uid, pmid=pmid, doi=doi, title=title)

    return {
        "paper_id": paper_id,
        "metadata_source": "wos",
        "source_path": "",
        "source_file": str(source_file),
        "file_type": "metadata",
        "title": title,
        "abstract": abstract,
        "text_for_filter": abstract,
        "doi": doi,
        "pmid": pmid,
        "pmcid": "",
        "wos_uid": wos_uid,
        "authors": _list(fields, "AU"),
        "journal": _text(fields, "SO"),
        "year": _text(fields, "PY"),
        "document_type": _text(fields, "DT"),
        "keywords": _list(fields, "DE") + _list(fields, "ID"),
        "abstract_available": bool(abstract),
        "front_matter_used": False,
        "metadata_quality": "external_metadata" if title or abstract else "title_only",
    }


def _text(fields: dict[str, list[str]], key: str) -> str:
    return " ".join(part.strip() for part in fields.get(key, []) if part.strip()).strip()


def _list(fields: dict[str, list[str]], key: str) -> list[str]:
    parts = fields.get(key, [])
    if key in LIST_FIELDS:
        values: list[str] = []
        for part in parts:
            values.extend(v.strip() for v in part.split(";") if v.strip())
        return values
    text = _text(fields, key)
    return [text] if text else []


def _make_paper_id(*, wos_uid: str, pmid: str, doi: str, title: str) -> str:
    if wos_uid:
        return _safe_id(wos_uid.replace("WOS:", "WOS_"))
    if pmid:
        return f"PMID_{_safe_id(pmid)}"
    if doi:
        return f"DOI_{_safe_id(doi)}"
    return f"WOS_TITLE_{_safe_id(title[:80])}"


def _safe_id(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "unknown"


def _dedupe_key(row: dict) -> tuple[str, str]:
    for key in ("doi", "pmid", "wos_uid"):
        value = str(row.get(key, "")).strip().lower()
        if value:
            return key, value
    return "title", str(row.get("title", "")).strip().lower()
