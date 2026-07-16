"""Resolve DOI/PMID metadata to PMC JATS XML when available."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
PMC_XML_URL = "https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/?report=xml"
EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPEPMC_XML_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"


class XMLUnavailableError(RuntimeError):
    """Raised when a PMCID exists but no JATS XML is available from providers."""


def acquire_pmc_xml(
    passed_rows: list[dict],
    output_dir: str | Path,
    *,
    email: str = "",
    tool: str = "literature-extraction-agent",
    sleep_seconds: float = 0.34,
    limit: int | None = None,
    resume: bool = True,
) -> dict:
    """Resolve passed metadata rows to PMCID and download available PMC XML."""

    output_dir = Path(output_dir)
    xml_dir = output_dir / "pmc_xml"
    xml_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "fulltext_acquisition_results.jsonl"
    downloaded_path = output_dir / "downloaded_papers.jsonl"
    existing_results = _read_jsonl_if_exists(results_path) if resume else []
    existing_downloaded_rows = _read_jsonl_if_exists(downloaded_path) if resume else []
    done_index = _identity_index(
        [row for row in existing_results if row.get("status") != "error"]
    )

    rows = list(passed_rows)
    if resume and done_index:
        rows = [row for row in rows if not (_identity_values(row) & done_index)]
    if limit:
        rows = rows[:limit]
    pending_index = _identity_index(rows)
    if pending_index:
        existing_results = [
            row for row in existing_results
            if not (_identity_values(row) & pending_index)
        ]

    results: list[dict] = list(existing_results)
    new_results: list[dict] = []
    downloaded_rows: list[dict] = list(existing_downloaded_rows)

    for row in rows:
        result = _process_one(
            row,
            xml_dir=xml_dir,
            email=email,
            tool=tool,
        )
        new_results.append(result)
        results.append(result)
        if result["status"] == "downloaded" or result["status"] == "already_exists":
            downloaded_rows.append(_to_downloaded_paper_row(row, result))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    downloaded_rows = _dedupe_rows(downloaded_rows)

    _write_jsonl(results_path, results)
    _write_jsonl(downloaded_path, downloaded_rows)

    summary = {
        "total": len(results),
        "processed_this_run": len(new_results),
        "previously_processed": len(existing_results),
        "downloaded": sum(1 for r in results if r["status"] == "downloaded"),
        "already_exists": sum(1 for r in results if r["status"] == "already_exists"),
        "no_pmcid": sum(1 for r in results if r["status"] == "no_pmcid"),
        "xml_unavailable": sum(1 for r in results if r["status"] == "xml_unavailable"),
        "error": sum(1 for r in results if r["status"] == "error"),
        "downloaded_papers_path": str(downloaded_path.resolve()),
        "xml_dir": str(xml_dir.resolve()),
    }
    with open(output_dir / "fulltext_acquisition_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    return summary


def resolve_pmcid(row: dict, *, email: str = "", tool: str = "literature-extraction-agent") -> str:
    """Return PMCID for one row if Europe PMC or NCBI can map DOI/PMID."""

    if row.get("pmcid"):
        return _normalize_pmcid(str(row["pmcid"]))

    pmid = str(row.get("pmid") or "").strip()
    doi = str(row.get("doi") or "").strip()
    if not pmid and not doi:
        return ""

    pmcid = _resolve_pmcid_europepmc(pmid=pmid, doi=doi)
    if pmcid:
        return pmcid

    identifier = pmid or doi
    params = {
        "ids": identifier,
        "format": "json",
        "tool": tool,
    }
    if email:
        params["email"] = email
    url = f"{IDCONV_URL}?{urllib.parse.urlencode(params)}"
    data = _get_json(url)
    records = data.get("records", [])
    if not records:
        return ""

    pmcid = str(records[0].get("pmcid", "")).strip()
    return _normalize_pmcid(pmcid)


def download_pmc_xml(pmcid: str, output_path: str | Path) -> None:
    """Download PMC XML to output_path."""

    pmcid = _normalize_pmcid(pmcid)
    urls = [
        EUROPEPMC_XML_URL.format(pmcid=urllib.parse.quote(pmcid)),
        PMC_XML_URL.format(pmcid=urllib.parse.quote(pmcid)),
    ]

    last_error: Exception | None = None
    text = ""
    for url in urls:
        try:
            text = _get_text(url)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    else:
        raise RuntimeError(f"Could not download XML for {pmcid}: {last_error}")

    if "<article" not in text[:2000] and "<?xml" not in text[:200]:
        raise XMLUnavailableError(f"No JATS XML available for {pmcid}")

    output_path = Path(output_path)
    output_path.write_text(text, encoding="utf-8")


def _process_one(row: dict, *, xml_dir: Path, email: str, tool: str) -> dict:
    pmcid = ""
    base = {
        "paper_id": row.get("paper_id", ""),
        "title": row.get("title", ""),
        "doi": row.get("doi", ""),
        "pmid": row.get("pmid", ""),
        "pmcid": row.get("pmcid", ""),
    }

    try:
        pmcid = resolve_pmcid(row, email=email, tool=tool)
        if not pmcid:
            return {**base, "status": "no_pmcid", "reason": "No PMCID found for DOI/PMID"}

        xml_path = xml_dir / f"{pmcid}.xml"
        if xml_path.exists() and xml_path.stat().st_size > 0:
            return {
                **base,
                "pmcid": pmcid,
                "status": "already_exists",
                "source_path": str(xml_path),
                "reason": "",
            }

        download_pmc_xml(pmcid, xml_path)
        return {
            **base,
            "pmcid": pmcid,
            "status": "downloaded",
            "source_path": str(xml_path),
            "reason": "",
        }
    except XMLUnavailableError as exc:
        logger.info("JATS XML unavailable for %s: %s", row.get("paper_id"), exc)
        return {
            **base,
            "pmcid": pmcid or base.get("pmcid", ""),
            "status": "xml_unavailable",
            "source_path": "",
            "reason": str(exc),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Full-text acquisition failed for %s: %s", row.get("paper_id"), exc)
        return {
            **base,
            "pmcid": pmcid or base.get("pmcid", ""),
            "status": "error",
            "source_path": "",
            "reason": str(exc),
        }


def _resolve_pmcid_europepmc(*, pmid: str = "", doi: str = "") -> str:
    """Resolve PMCID via Europe PMC, which is stable for WOS PMID/DOI inputs."""

    queries: list[str] = []
    if pmid:
        queries.append(f"EXT_ID:{pmid} AND SRC:MED")
    if doi:
        queries.append(f'DOI:"{doi}"')

    for query in queries:
        params = {
            "query": query,
            "format": "json",
            "pageSize": "1",
        }
        url = f"{EUROPEPMC_SEARCH_URL}?{urllib.parse.urlencode(params)}"
        try:
            data = _get_json(url)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Europe PMC PMCID lookup failed for %s: %s", query, exc)
            continue

        results = data.get("resultList", {}).get("result", [])
        if not results:
            continue

        row = results[0]
        pmcid = str(row.get("pmcid", "") or "").strip()
        if pmcid:
            return _normalize_pmcid(pmcid)

        for full_text_id in row.get("fullTextIdList", {}).get("fullTextId", []):
            full_text_id = str(full_text_id).strip()
            if full_text_id.upper().startswith("PMC"):
                return _normalize_pmcid(full_text_id)

    return ""


def _to_downloaded_paper_row(original: dict, result: dict) -> dict:
    return {
        "paper_id": result["pmcid"],
        "source_path": result["source_path"],
        "file_type": "xml",
        "doi": original.get("doi", ""),
        "pmid": original.get("pmid", ""),
        "pmcid": result["pmcid"],
        "title": original.get("title", ""),
        "abstract_available": bool(original.get("abstract") or original.get("text_for_filter")),
        "metadata_quality": "external_metadata",
        "metadata_source": original.get("metadata_source", ""),
        "wos_uid": original.get("wos_uid", ""),
    }


def _normalize_pmcid(pmcid: str) -> str:
    pmcid = pmcid.strip()
    if not pmcid:
        return ""
    if pmcid.upper().startswith("PMC"):
        return "PMC" + pmcid[3:]
    if pmcid.isdigit():
        return f"PMC{pmcid}"
    return pmcid


def _get_json(url: str) -> dict:
    return json.loads(_get_text(url))


def _get_text(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "literature-extraction-agent/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} for {url}") from exc


def _read_jsonl_if_exists(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _identity_values(row: dict) -> set[str]:
    values: set[str] = set()
    for key in ("paper_id", "pmcid", "pmid", "doi", "wos_uid", "source_path", "source_file"):
        text = str(row.get(key) or "").strip()
        if text:
            values.add(text)
    return values


def _identity_index(rows: list[dict]) -> set[str]:
    values: set[str] = set()
    for row in rows:
        values.update(_identity_values(row))
    return values


def _dedupe_rows(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []
    for row in rows:
        identities = _identity_values(row)
        key = next(iter(sorted(identities)), "")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(row)
    return deduped


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
