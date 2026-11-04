"""Unified inline citation extractor for single (`[M031]`) and grouped (`[M031, CON_AND]`) bracket citations."""

from __future__ import annotations

import re

# Matches bracket groups `[...]` NOT immediately followed by `(` (excludes Markdown links `[C100](https://...)`)
_BRACKET_GROUP_RE = re.compile(r"\[([^\[\]\n]{2,120})\](?!\()")

# Valid individual catalog/guide/constellation token inside `[...]`
_SINGLE_TOKEN_RE = re.compile(
    r"^(?:(?:M|C)\d{1,3}|Messier\s*\d{1,3}|Caldwell\s*\d{1,3}|NGC\s*\d{1,4}|IC\s*\d{1,4}|CON_[A-Z]{3}|GUIDE_[A-Z0-9_]+)$",
    flags=re.IGNORECASE,
)


def extract_inline_citations(text: str) -> list[str]:
    """Extract all raw inline catalog/constellation/guide citations from `text` in appearance order.

    Supports both single `[M031]` and comma/semicolon-separated `[M031, M032, CON_AND]` brackets,
    while ignoring Markdown links `[label](url)` and non-citation brackets.
    """
    if not text:
        return []
    found: list[str] = []
    for group_match in _BRACKET_GROUP_RE.finditer(text):
        inner = group_match.group(1).strip()
        parts = [p.strip() for p in re.split(r"[,;/]+", inner) if p.strip()]
        if not parts:
            continue
        # Only treat the bracket group as a citation block when ALL tokens inside are valid astro IDs
        # (prevents false positives on prose notes while catching `[M031, M999]` hallucinations!)
        if all(_SINGLE_TOKEN_RE.match(p) for p in parts):
            found.extend(parts)
    return found
