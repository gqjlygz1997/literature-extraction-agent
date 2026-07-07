"""JATS/PMC XML full-text parser producing DocumentChunk objects.

This is the phase-2 (preprocessing) core.  It parses a PMC/JATS XML article
into a flat stream of content chunks (abstract / paragraph / table) while
preserving the section hierarchy in ``section_path``.

Design (see docs/preprocessing_mvp.md):
  - Depth-first walk of <body>, maintaining section_path as a stack (list).
  - Abstract from <front>/<abstract>; structured abstracts keep sub-titles.
  - <p> may contain inline <table-wrap>; those are stripped out and emitted
    as separate table chunks (JATS common case; mirrors ALLMAT
    xml_section_extract_acs).
  - Tables keep their real section_path (improvement over ALLMAT which drops
    it to the literal "table") plus structured label/caption/header_rows/
    headers/rows/footnotes/markdown_text.
  - Paragraph chunking mirrors ALLMAT loader.chunk_text: <=400 tokens stays
    whole; >400 splits by sentence into 200-600 token chunks.

This module intentionally supports **only JATS XML** — no multi-publisher
HTML parsers (unlike ALLMAT's chempp).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .document_parser import DocumentChunk


# ---------------------------------------------------------------------------
# Chunking constants (hard-coded in v1; not exposed via config)
# ---------------------------------------------------------------------------

MAX_CHUNK_TOKENS = 400      # paragraphs longer than this get split
MIN_TAIL_TOKENS = 200       # avoid trailing chunks smaller than this
TOKEN_UPPER_BOUND = 600     # informational target ceiling
PARSER_VERSION = "preprocessing_mvp_v1"
SUPPORTED_FORMAT = "jats_xml_only"
CHUNKING_CONFIG = {
    "max_chunk_tokens": MAX_CHUNK_TOKENS,
    "min_tail_tokens": MIN_TAIL_TOKENS,
    "token_upper_bound": TOKEN_UPPER_BOUND,
    "overlap_tokens": 0,
    "split_by_sentence": True,
}

# Back-matter section tags that carry no domain evidence -> skipped.
# NOTE: supplementary-material is deliberately NOT here — it often holds data
# tables that are core evidence.  See docs/preprocessing_mvp.md §5.
SKIP_BACK_TAGS = {"ref-list", "ack", "fn-group", "notes"}

# Sentence splitter: split on . ! ? followed by whitespace + capital/quote/digit.
# Lightweight alternative to nltk (no data download); good enough for sci text.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'“(\[]?[A-Z0-9])")


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_ENCODER = None


def _count_tokens(text: str) -> int:
    """Count tokens with tiktoken (cl100k_base); fall back to chars/4."""
    global _ENCODER
    if _ENCODER is None:
        try:
            import tiktoken
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001 - tiktoken missing or offline
            _ENCODER = False  # sentinel: use char approximation
    if _ENCODER is False:
        return max(1, len(text) // 4)
    return len(_ENCODER.encode(text))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_jats_full_text(
    path,
    paper_id: str,
    doi: str,
    title: str,
) -> tuple[list["DocumentChunk"], dict[str, int]]:
    """Parse one JATS/PMC XML file.

    Returns ``(chunks, skipped_sections)`` where skipped_sections maps a
    skipped back-matter tag (ref-list, ack, ...) to how many times it was
    dropped — surfaced in the run summary.

    Raises a plain Exception on structural failure (no <body>); callers in the
    workflow catch per-paper so one bad file doesn't stop the batch.
    """
    from bs4 import BeautifulSoup

    with open(path, encoding="utf-8") as fh:
        soup = BeautifulSoup(fh.read(), "lxml-xml")

    body = soup.find("body")
    if body is None:
        raise ValueError("no <body> found in JATS XML")

    ctx = _Ctx(paper_id=paper_id, doi=doi, title=title, source_path=str(path))

    front = soup.find("front")
    if front is not None:
        _emit_abstract(front, ctx)

    _walk(body, [], ctx)

    # Also account for back-matter dropped outside <body>.
    back = soup.find("back")
    if back is not None:
        for tag in SKIP_BACK_TAGS:
            for _ in back.find_all(tag):
                ctx.note_skip(tag)

    return ctx.chunks, ctx.skipped


# ---------------------------------------------------------------------------
# Internal traversal context
# ---------------------------------------------------------------------------

class _Ctx:
    """Mutable state threaded through the recursive walk."""

    def __init__(self, paper_id: str, doi: str, title: str, source_path: str):
        self.paper_id = paper_id
        self.doi = doi
        self.title = title
        self.source_path = source_path
        self.chunks: list = []
        self.index = 0                      # global chunk_index within paper
        self.skipped: dict[str, int] = {}   # tag -> count, for summary

    def _common_meta(self, sec_type, token_count: int) -> dict[str, Any]:
        return {
            "doi": self.doi,
            "title": self.title,
            "source_path": self.source_path,
            "sec_type": sec_type,
            "chunk_index": self.index,
            "token_count": token_count,
        }

    def add_paragraph(self, text: str, section_path: list[str], sec_type):
        from .document_parser import DocumentChunk
        cid = f"{self.paper_id}::p{self.index:04d}"
        self.chunks.append(DocumentChunk(
            paper_id=self.paper_id,
            chunk_id=cid,
            text=text,
            chunk_type="paragraph",
            section_path=list(section_path),
            metadata=self._common_meta(sec_type, _count_tokens(text)),
        ))
        self.index += 1

    def add_abstract(self, text: str, section_path: list[str]):
        from .document_parser import DocumentChunk
        cid = f"{self.paper_id}::a{self.index:04d}"
        self.chunks.append(DocumentChunk(
            paper_id=self.paper_id,
            chunk_id=cid,
            text=text,
            chunk_type="abstract",
            section_path=list(section_path),
            metadata=self._common_meta(None, _count_tokens(text)),
        ))
        self.index += 1

    def add_table(self, text: str, section_path: list[str], sec_type, extra: dict):
        from .document_parser import DocumentChunk
        cid = f"{self.paper_id}::t{self.index:04d}"
        meta = self._common_meta(sec_type, _count_tokens(text))
        meta.update(extra)
        self.chunks.append(DocumentChunk(
            paper_id=self.paper_id,
            chunk_id=cid,
            text=text,
            chunk_type="table",
            section_path=list(section_path),
            metadata=meta,
        ))
        self.index += 1

    def note_skip(self, tag: str):
        self.skipped[tag] = self.skipped.get(tag, 0) + 1


# ---------------------------------------------------------------------------
# Abstract
# ---------------------------------------------------------------------------

def _emit_abstract(front, ctx: _Ctx) -> None:
    """Emit abstract chunks. Structured abstracts keep sub-section titles."""
    for abstract_el in front.find_all("abstract"):
        # skip graphical / teaser abstracts if labelled as such
        atype = (abstract_el.get("abstract-type") or "").lower()
        if atype in {"graphical", "teaser", "precis"}:
            continue

        sub_secs = abstract_el.find_all("sec", recursive=False)
        if sub_secs:
            for sec in sub_secs:
                title_el = sec.find("title", recursive=False)
                sub_title = _clean(title_el.get_text(" ", strip=True)) if title_el else ""
                path = ["Abstract", sub_title] if sub_title else ["Abstract"]
                parts = [_clean(p.get_text(" ", strip=True)) for p in sec.find_all("p")]
                text = " ".join(t for t in parts if t)
                if text:
                    ctx.add_abstract(text, path)
        else:
            parts = [_clean(p.get_text(" ", strip=True)) for p in abstract_el.find_all("p")]
            text = " ".join(t for t in parts if t)
            if text:
                ctx.add_abstract(text, ["Abstract"])


# ---------------------------------------------------------------------------
# Body walk
# ---------------------------------------------------------------------------

def _walk(sec, section_path: list[str], ctx: _Ctx) -> None:
    """Depth-first traversal of a <body> or <sec>, emitting chunks."""
    for child in sec.children:
        name = getattr(child, "name", None)
        if name is None:
            continue  # NavigableString

        if name in SKIP_BACK_TAGS:
            ctx.note_skip(name)
            continue

        if name == "sec":
            title_el = child.find("title", recursive=False)
            title_txt = _clean(title_el.get_text(" ", strip=True)) if title_el else ""
            cur_path = section_path + [title_txt] if title_txt else list(section_path)
            _walk(child, cur_path, ctx)

        elif name == "p":
            _emit_paragraph(child, section_path, _sec_type_of(sec), ctx)

        elif name == "table-wrap":
            _emit_table(child, section_path, _sec_type_of(sec), ctx)

        # fig / disp-formula / list / etc. are skipped in v1.
        # But recurse into wrapper-like containers that may hold sec/p.
        elif name in {"boxed-text", "app", "app-group"}:
            _walk(child, section_path, ctx)


def _emit_paragraph(p_tag, section_path: list[str], sec_type, ctx: _Ctx) -> None:
    """Emit a paragraph. Inline <table-wrap> is stripped out and emitted
    as its own table chunk first (JATS common case).

    We work on a re-parsed clone so decompose() never mutates the live tree.
    """
    inline_tables = p_tag.find_all("table-wrap")

    if inline_tables:
        from bs4 import BeautifulSoup
        p_clone = BeautifulSoup(str(p_tag), "lxml-xml").find("p")
        if p_clone is not None:
            for tw in p_clone.find_all("table-wrap"):
                _emit_table(tw, section_path, sec_type, ctx)
                tw.decompose()
            text = _clean(p_clone.get_text(" ", strip=True))
        else:
            # fallback: emit inline tables off the live tag, then take text
            for tw in inline_tables:
                _emit_table(tw, section_path, sec_type, ctx)
            text = _clean(p_tag.get_text(" ", strip=True))
    else:
        text = _clean(p_tag.get_text(" ", strip=True))

    if not text or _is_noise(text):
        return

    for chunk_text in _chunk_paragraph(text):
        ctx.add_paragraph(chunk_text, section_path, sec_type)


# ---------------------------------------------------------------------------
# Table parsing
# ---------------------------------------------------------------------------

def _emit_table(table_wrap, section_path: list[str], sec_type, ctx: _Ctx) -> None:
    label_el = table_wrap.find("label")
    label = _clean(label_el.get_text(" ", strip=True)) if label_el else ""

    caption_el = table_wrap.find("caption")
    caption = _clean(caption_el.get_text(" ", strip=True)) if caption_el else ""

    table_el = table_wrap.find("table")
    header_rows, rows = _parse_jats_table(table_el) if table_el else ([], [])
    headers = _flatten_headers(header_rows)

    footnotes = []
    foot_el = table_wrap.find("table-wrap-foot")
    if foot_el:
        for p in foot_el.find_all(["p", "fn"]):
            t = _clean(p.get_text(" ", strip=True))
            if t:
                footnotes.append(t)
        if not footnotes:
            t = _clean(foot_el.get_text(" ", strip=True))
            if t:
                footnotes.append(t)

    markdown = _to_markdown(headers, rows)

    caption_line = ". ".join(x for x in (label, caption) if x)
    text = f"{caption_line}\n{markdown}" if caption_line else markdown

    n_cols = max((len(r) for r in ([headers] if headers else []) + rows), default=0)
    extra = {
        "label": label,
        "caption": caption,
        "header_rows": header_rows,
        "headers": headers,
        "rows": rows,
        "n_rows": len(rows),
        "n_cols": n_cols,
        "footnotes": footnotes,
        "markdown_text": markdown,
    }
    ctx.add_table(text, section_path, sec_type, extra)


def _parse_jats_table(table_el) -> tuple[list[list[str]], list[list[str]]]:
    """Parse a JATS <table> into (header_rows, body_rows).

    Uses the rowspan/colspan grid-fill algorithm (mirrors ALLMAT
    loader.parse_html_table_to_json): allocate a grid, fill cells honoring
    rowspan/colspan, put merged-cell overflow as empty strings.

    thead rows -> header_rows; tbody (or leftover) rows -> body_rows.
    If no thead, the first row is treated as the header.
    """
    thead = table_el.find("thead")
    tbody = table_el.find("tbody")

    head_trs = thead.find_all("tr") if thead else []
    if tbody:
        body_trs = tbody.find_all("tr")
    else:
        all_trs = table_el.find_all("tr")
        # drop any rows already counted in thead
        head_set = set(id(tr) for tr in head_trs)
        body_trs = [tr for tr in all_trs if id(tr) not in head_set]

    header_rows = _grid_fill(head_trs)
    body_rows = _grid_fill(body_trs)

    if not header_rows and body_rows:
        # no thead: promote first body row to header
        header_rows = [body_rows[0]]
        body_rows = body_rows[1:]

    return header_rows, body_rows


def _grid_fill(trs) -> list[list[str]]:
    """Fill a 2D grid from <tr> list honoring rowspan/colspan."""
    if not trs:
        return []
    grid: list[list] = [[] for _ in range(len(trs))]

    for r, tr in enumerate(trs):
        cells = tr.find_all(["td", "th"])
        col = 0
        for cell in cells:
            # advance past cells already filled by a spanning cell above/left
            while col < len(grid[r]) and grid[r][col] is not None:
                col += 1
            try:
                rowspan = int(cell.get("rowspan", 1) or 1)
            except ValueError:
                rowspan = 1
            try:
                colspan = int(cell.get("colspan", 1) or 1)
            except ValueError:
                colspan = 1
            content = _clean(cell.get_text(" ", strip=True))

            for dr in range(rowspan):
                rr = r + dr
                if rr >= len(grid):
                    break
                while len(grid[rr]) < col + colspan:
                    grid[rr].append(None)
                for dc in range(colspan):
                    grid[rr][col + dc] = content if (dr == 0 and dc == 0) else ""
            col += colspan

    return [[c if c is not None else "" for c in row] for row in grid]


def _flatten_headers(header_rows: list[list[str]]) -> list[str]:
    """Flatten multi-row headers into single column labels.

    Joins non-empty cell text top-to-bottom per column with a space.
    """
    if not header_rows:
        return []
    n_cols = max(len(r) for r in header_rows)
    out: list[str] = []
    for c in range(n_cols):
        parts = []
        for row in header_rows:
            if c < len(row) and row[c]:
                if not parts or parts[-1] != row[c]:
                    parts.append(row[c])
        out.append(" ".join(parts))
    return out


def _to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    """Render a Markdown table. Falls back gracefully when headers absent."""
    if not headers and not rows:
        return ""
    n_cols = max(
        [len(headers)] + [len(r) for r in rows]
    ) if (headers or rows) else 0
    if n_cols == 0:
        return ""

    def _row(cells: list[str]) -> str:
        padded = list(cells) + [""] * (n_cols - len(cells))
        return "| " + " | ".join(c.replace("|", "\\|") for c in padded) + " |"

    lines = []
    if headers:
        lines.append(_row(headers))
    else:
        lines.append(_row([""] * n_cols))
    lines.append("|" + "|".join(["---"] * n_cols) + "|")
    for r in rows:
        lines.append(_row(r))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Paragraph chunking (mirrors ALLMAT loader.chunk_text)
# ---------------------------------------------------------------------------

def _chunk_paragraph(text: str) -> list[str]:
    """Split a paragraph into 200-600 token chunks.

    <=400 tokens: keep whole. Otherwise accumulate sentences up to ~400
    tokens, avoiding a trailing chunk smaller than MIN_TAIL_TOKENS.
    """
    if _count_tokens(text) <= MAX_CHUNK_TOKENS:
        return [text]

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text]

    tok = [_count_tokens(s) for s in sentences]
    chunks: list[str] = []
    start = 0
    accum = 0
    for i in range(len(sentences)):
        if i == len(sentences) - 1:
            chunks.append(" ".join(sentences[start:]))
            break
        accum += tok[i]
        tokens_left = sum(tok[i + 1:])
        if accum <= MAX_CHUNK_TOKENS:
            continue
        if tokens_left >= MIN_TAIL_TOKENS:
            chunks.append(" ".join(sentences[start:i + 1]))
            start = i + 1
            accum = 0
        else:
            chunks.append(" ".join(sentences[start:]))
            break
    return [c for c in chunks if c.strip()]


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    if not text:
        return ""
    return _WS_RE.sub(" ", text).strip()


def _is_noise(text: str) -> bool:
    """Drop empty / pure-symbol / trivial residual paragraphs."""
    if len(text) < 2:
        return True
    if not re.search(r"[A-Za-z0-9]", text):
        return True
    return False


def _sec_type_of(sec) -> str | None:
    """Return the sec-type attribute of a <sec>, or None (body has none)."""
    if getattr(sec, "name", None) == "sec":
        return sec.get("sec-type") or None
    return None
