"""Text cleaner and MediaWiki section splitter for Wikipedia astronomy articles."""

from __future__ import annotations

import re

# Sections that contain bibliography/navigation noise rather than astronomical content
STOP_SECTIONS: set[str] = {
    "see also",
    "references",
    "external links",
    "notes",
    "further reading",
    "gallery",
    "citations",
    "sources",
    "bibliography",
}

_CITATION_RE = re.compile(r"\[(?:\d+|citation needed|nb \d+|note \d+)\]", flags=re.IGNORECASE)
_HEADING_RE = re.compile(r"^(={2,4})\s*(.+?)\s*\1\s*$", flags=re.MULTILINE)
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def clean_wiki_text(text: str) -> str:
    """Remove citation brackets, redundant whitespace, and formatting artifacts."""
    cleaned = _CITATION_RE.sub("", text)
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
    cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def split_into_sections(raw_extract: str) -> list[tuple[str, str]]:
    """Split a plain-text Wikipedia extract into `(section_title, cleaned_body)` pairs.

    The introductory lead paragraph(s) before the first `== Heading ==` are labeled `"Overview"`.
    Boilerplate sections (`References`, `See also`, `External links`, etc.) are discarded.
    """
    if not raw_extract or not raw_extract.strip():
        return []

    matches = list(_HEADING_RE.finditer(raw_extract))
    sections: list[tuple[str, str]] = []

    # 1. Lead / Overview section before the first heading
    first_pos = matches[0].start() if matches else len(raw_extract)
    lead_text = clean_wiki_text(raw_extract[:first_pos])
    if len(lead_text) >= 40:
        sections.append(("Overview", lead_text))

    # 2. Subsequent sections
    for idx, match in enumerate(matches):
        heading_title = match.group(2).strip()
        start_pos = match.end()
        end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_extract)

        if heading_title.lower() in STOP_SECTIONS:
            continue

        body = clean_wiki_text(raw_extract[start_pos:end_pos])
        if len(body) >= 40:
            sections.append((heading_title, body))

    return sections
