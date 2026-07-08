"""Parse local scientific papers into metadata and text chunks.

Supported formats: XML (JATS/PMC), HTML.
PDF is explicitly not supported — callers receive an error entry in results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

MetadataQuality = Literal[
    "structured_xml",     # JATS XML with title + abstract
    "html_abstract",      # HTML with title + extracted abstract
    "html_front_matter",  # HTML with title but only front-matter fallback
    "external_metadata",  # title/abstract imported from WOS or another index
    "title_only",         # only a title was found
    "parse_error",        # parsing failed entirely
]


@dataclass(frozen=True)
class ArticleMeta:
    """Lightweight metadata produced by parse_metadata_light.

    text_for_filter is what gets sent to the LLM paper classifier:
      - abstract text when available
      - front-matter fallback (first N long paragraphs) when abstract is missing
      - empty string when nothing could be extracted
    """

    source_path: Path
    file_type: Literal["xml", "html", "txt", "metadata", "unknown"]
    title: str = ""
    abstract: str = ""
    text_for_filter: str = ""       # abstract or front_matter fallback
    paper_id: str = ""
    doi: str = ""
    abstract_available: bool = False
    front_matter_used: bool = False
    metadata_quality: MetadataQuality = "parse_error"
    error: str = ""                  # non-empty only when metadata_quality == "parse_error"


@dataclass(frozen=True)
class DocumentChunk:
    """A paragraph, table, or caption after full-text parsing (phase 2+)."""

    paper_id: str
    chunk_id: str
    text: str
    chunk_type: Literal["paragraph", "table", "caption", "title", "abstract"]
    section_path: list[str] = field(default_factory=list)
    # metadata is dict[str, Any]: table chunks store list/int values
    # (headers, rows, n_rows, ...), not just strings.
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class DocumentParser:
    """Parser facade for XML and HTML files.

    PDF is not supported.  Callers should check ArticleMeta.metadata_quality
    and ArticleMeta.error to detect parse failures.
    """

    # ---- public API --------------------------------------------------------

    def detect_file_type(self, path: str | Path) -> str:
        suffix = Path(path).suffix.lower()
        if suffix in {".xml", ".nxml"}:
            return "xml"
        if suffix in {".html", ".htm"}:
            return "html"
        if suffix == ".pdf":
            return "pdf"
        if suffix == ".txt":
            return "txt"
        return "unknown"

    def parse_metadata_light(self, path: str | Path) -> ArticleMeta:
        """Read only title and abstract (or front-matter fallback) for paper filtering.

        PDF files are not supported and return an ArticleMeta with
        metadata_quality="parse_error" and error="PDF is not supported".
        """
        path = Path(path)
        file_type = self.detect_file_type(path)

        if file_type == "pdf":
            return ArticleMeta(
                source_path=path,
                file_type="unknown",   # we store "unknown" so downstream code
                                       # never has to handle a "pdf" literal
                paper_id=path.stem,
                metadata_quality="parse_error",
                error="PDF is not supported",
            )

        if file_type not in {"xml", "html"}:
            return ArticleMeta(
                source_path=path,
                file_type=file_type,
                paper_id=path.stem,
                metadata_quality="parse_error",
                error=f"Unsupported file type: {path.suffix!r}",
            )

        try:
            if file_type == "xml":
                return self._parse_xml_light(path)
            else:
                return self._parse_html_light(path)
        except Exception as exc:  # noqa: BLE001
            return ArticleMeta(
                source_path=path,
                file_type=file_type,
                paper_id=path.stem,
                metadata_quality="parse_error",
                error=str(exc),
            )

    def parse_full_text(self, path: str | Path) -> list[DocumentChunk]:
        """Parse JATS/PMC XML full text into section-aware DocumentChunks.

        Facade: reuses the light metadata pass to resolve paper_id/doi/title,
        then delegates the heavy walk to xml_full_text_parser.

        Only XML is supported.  HTML/PDF raise ValueError so callers can record
        an error entry and continue with the next paper.
        """
        path = Path(path)
        file_type = self.detect_file_type(path)
        if file_type != "xml":
            raise ValueError(f"parse_full_text supports XML only, got {file_type!r}")

        chunks, _skips = self.parse_full_text_with_skips(path)
        return chunks

    def parse_full_text_with_skips(
        self, path: str | Path
    ) -> tuple[list[DocumentChunk], dict[str, int]]:
        """Like parse_full_text but also returns skipped-section counts.

        Used by the preprocessing workflow to aggregate a run summary.
        """
        path = Path(path)
        file_type = self.detect_file_type(path)
        if file_type != "xml":
            raise ValueError(f"parse_full_text supports XML only, got {file_type!r}")

        meta = self._parse_xml_light(path)
        paper_id = meta.paper_id or path.stem

        from .xml_full_text_parser import parse_jats_full_text
        return parse_jats_full_text(
            path,
            paper_id=paper_id,
            doi=meta.doi,
            title=meta.title,
        )

    # ---- XML (JATS/PMC) ----------------------------------------------------

    def _parse_xml_light(self, path: Path) -> ArticleMeta:
        from bs4 import BeautifulSoup

        with open(path, encoding="utf-8") as fh:
            soup = BeautifulSoup(fh.read(), "lxml-xml")

        # --- IDs ---
        doi = pmcid = None
        for aid in soup.find_all("article-id"):
            pub_type = aid.get("pub-id-type", "")
            if pub_type == "doi" and not doi:
                doi = aid.get_text(strip=True)
            elif pub_type == "pmcid" and not pmcid:
                pmcid = aid.get_text(strip=True)
        paper_id = pmcid or path.stem

        # --- title ---
        title_el = soup.find("article-title")
        title = title_el.get_text(" ", strip=True) if title_el else ""

        # --- abstract ---
        abstract_parts: list[str] = []
        front = soup.find("front")
        for abstract_el in (front.find_all("abstract") if front else []):
            for p in abstract_el.find_all("p"):
                text = p.get_text(" ", strip=True)
                if text:
                    abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        abstract_available = bool(abstract)
        text_for_filter = abstract

        # quality assessment
        if title and abstract_available:
            quality: MetadataQuality = "structured_xml"
        elif title:
            quality = "title_only"
        else:
            quality = "parse_error"

        return ArticleMeta(
            source_path=path,
            file_type="xml",
            title=title,
            abstract=abstract,
            text_for_filter=text_for_filter,
            paper_id=paper_id,
            doi=doi or "",
            abstract_available=abstract_available,
            front_matter_used=False,
            metadata_quality=quality,
            error="" if quality != "parse_error" else "No title found in XML",
        )

    # ---- HTML --------------------------------------------------------------

    def _parse_html_light(self, path: Path) -> ArticleMeta:
        from bs4 import BeautifulSoup

        with open(path, encoding="utf-8", errors="replace") as fh:
            soup = BeautifulSoup(fh.read(), "lxml")

        paper_id = path.stem
        title = self._extract_html_title(soup)
        abstract, abstract_available = self._extract_html_abstract(soup)

        if abstract_available:
            text_for_filter = abstract
            front_matter_used = False
            quality: MetadataQuality = "html_abstract"
        else:
            front_matter = self._extract_html_front_matter(soup, max_chars=2000)
            if front_matter:
                text_for_filter = front_matter
                front_matter_used = True
                quality = "html_front_matter"
            else:
                text_for_filter = ""
                front_matter_used = False
                quality = "title_only" if title else "parse_error"

        return ArticleMeta(
            source_path=path,
            file_type="html",
            title=title,
            abstract=abstract,
            text_for_filter=text_for_filter,
            paper_id=paper_id,
            doi=self._extract_html_doi(soup),
            abstract_available=abstract_available,
            front_matter_used=front_matter_used,
            metadata_quality=quality,
            error="" if quality != "parse_error" else "No title or text found in HTML",
        )

    # ---- HTML extraction helpers -------------------------------------------

    @staticmethod
    def _extract_html_title(soup) -> str:
        """Extract title by priority: citation meta → DC meta → OG meta
        → h1 with article-title class/id → cleaned <title> tag."""

        # 1. citation_title meta
        for name in ("citation_title", "dc.title", "DC.title"):
            el = soup.find("meta", attrs={"name": name})
            if el and el.get("content", "").strip():
                return el["content"].strip()

        # 2. og:title
        el = soup.find("meta", property="og:title")
        if el and el.get("content", "").strip():
            return el["content"].strip()

        # 3. h1 or heading element with article-title in class/id
        for tag in ("h1", "h2", "h3"):
            for el in soup.find_all(tag):
                attrs = " ".join([
                    " ".join(el.get("class", [])),
                    el.get("id", ""),
                ])
                if re.search(r"article[-_]?title|headline", attrs, re.I):
                    text = el.get_text(" ", strip=True)
                    if text:
                        return text

        # 4. <title> tag with publisher suffix stripped
        title_el = soup.find("title")
        if title_el:
            raw = title_el.get_text(" ", strip=True)
            # strip " | Journal Name" or " - Publisher" suffixes
            cleaned = re.sub(r"\s*[|\-–—]\s*[^|\-–—]{4,}$", "", raw).strip()
            if cleaned:
                return cleaned

        return ""

    @staticmethod
    def _extract_html_abstract(soup) -> tuple[str, bool]:
        """Extract abstract text.  Returns (text, found_bool).

        Priority:
        1. meta citation_abstract
        2. meta dc.description / description
        3. element with id/class/name containing 'abstract'
        4. heading 'Abstract' or 'Summary' followed by paragraphs
        """

        # 1. citation_abstract meta
        el = soup.find("meta", attrs={"name": "citation_abstract"})
        if el and el.get("content", "").strip():
            return el["content"].strip(), True

        # 2. dc.description / description meta
        for name in ("dc.description", "DC.description", "description"):
            el = soup.find("meta", attrs={"name": name})
            if el:
                content = el.get("content", "").strip()
                # description metas can be very short (site descriptions);
                # only use if reasonably long
                if len(content) > 100:
                    return content, True

        # 3. element with 'abstract' in id / class / name attribute
        for tag in ("section", "div", "article", "p", "blockquote"):
            for el in soup.find_all(tag):
                attrs = " ".join([
                    " ".join(el.get("class", [])),
                    el.get("id", ""),
                    el.get("name", ""),
                ])
                if re.search(r"\babstract\b", attrs, re.I):
                    text = el.get_text(" ", strip=True)
                    if len(text) > 80:
                        return text[:3000], True

        # 4. heading 'Abstract' or 'Summary' → collect following paragraphs
        for heading_tag in ("h2", "h3", "h4", "h1"):
            for heading in soup.find_all(heading_tag):
                heading_text = heading.get_text(strip=True).lower()
                if heading_text in {"abstract", "summary", "抄录", "摘要"}:
                    parts: list[str] = []
                    for sib in heading.find_next_siblings():
                        if sib.name in {"h1", "h2", "h3", "h4"}:
                            break
                        text = sib.get_text(" ", strip=True)
                        if text:
                            parts.append(text)
                        if sum(len(p) for p in parts) > 2000:
                            break
                    abstract_text = " ".join(parts).strip()
                    if len(abstract_text) > 80:
                        return abstract_text[:3000], True

        return "", False

    @staticmethod
    def _extract_html_front_matter(soup, max_chars: int = 2000) -> str:
        """Collect the first long paragraphs from body as a front-matter proxy.

        Skips navigation, header, footer, and short boilerplate paragraphs.
        Returns at most max_chars characters.
        """
        MIN_LEN = 80  # characters; shorter paragraphs are likely nav/boilerplate

        # Remove known non-content elements in-place (soup is already parsed)
        for tag in soup.find_all(["nav", "header", "footer", "script", "style", "aside"]):
            tag.decompose()

        collected: list[str] = []
        total = 0
        for p in soup.find_all("p"):
            text = p.get_text(" ", strip=True)
            if len(text) < MIN_LEN:
                continue
            collected.append(text)
            total += len(text)
            if len(collected) >= 5 or total >= max_chars:
                break

        combined = " ".join(collected)
        return combined[:max_chars]

    @staticmethod
    def _extract_html_doi(soup) -> str:
        """Best-effort DOI extraction from HTML meta tags."""
        for name in ("citation_doi", "dc.identifier", "DC.identifier"):
            el = soup.find("meta", attrs={"name": name})
            if el and el.get("content", "").strip():
                val = el["content"].strip()
                # strip leading "doi:" prefix if present
                return re.sub(r"^doi:\s*", "", val, flags=re.I)
        return ""
