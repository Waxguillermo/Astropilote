"""Section-aware chunker with prepended object context header."""

from __future__ import annotations

import re

from astropilote.agent.schemas import Chunk
from astropilote.ingest.catalog import CatalogTarget

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def slugify_section(section: str) -> str:
    """Convert a section title into a clean URL/ID slug, e.g. 'Observation' -> 'observation'."""
    slug = _SLUG_RE.sub("-", section.lower()).strip("-")
    return slug or "section"


def format_context_header(target: CatalogTarget) -> str:
    """Build the self-contained context header prepended to every chunk of `target`."""
    parts = [
        target.doc_id,
        target.title,
        f"Type: {target.object_type}",
        f"Constellation: {target.constellation}",
    ]
    if target.magnitude is not None:
        parts.append(f"Mag: {target.magnitude}")
    if target.size_major_arcmin is not None:
        if target.size_minor_arcmin is not None:
            parts.append(f"Size: {target.size_major_arcmin}'x{target.size_minor_arcmin}'")
        else:
            parts.append(f"Size: {target.size_major_arcmin}'")
    if target.surface_brightness is not None:
        parts.append(f"SurfBr: {target.surface_brightness}")
    parts.append(f"Narrowband: {'yes' if target.narrowband_friendly else 'no'}")
    if target.aliases:
        parts.append(f"Aliases: {', '.join(target.aliases[:4])}")
    return "[" + " | ".join(parts) + "]"


def estimate_tokens(text: str) -> int:
    """Estimate subword token count (~1.33 tokens per whitespace word)."""
    words = len(text.split())
    return int(words * 1.33)


def chunk_target_sections(
    target: CatalogTarget,
    sections: list[tuple[str, str]],
    max_chunk_tokens: int = 400,
    overlap_words: int = 35,
) -> list[Chunk]:
    """Turn `(section_title, section_body)` pairs into self-contained `Chunk` models.

    Every chunk begins with `format_context_header(target)` + `Section: {section_title}`
    so neither BM25 nor dense embeddings lose the parent object identity.
    """
    header = format_context_header(target)
    metadata = target.to_chunk_metadata()
    chunks: list[Chunk] = []

    # Reserve ~55 tokens for the context header and section prefix
    body_word_budget = max(80, int((max_chunk_tokens - 55) / 1.33))

    for section_title, body in sections:
        slug = slugify_section(section_title)
        sentences = _SENTENCE_SPLIT_RE.split(body.strip())

        current_words: list[str] = []
        sub_idx = 0

        for sentence in sentences:
            s_words = sentence.split()
            if not s_words:
                continue

            if current_words and len(current_words) + len(s_words) > body_word_budget:
                body_str = " ".join(current_words)
                full_text = f"{header}\nSection: {section_title}\n{body_str}"
                chunks.append(
                    Chunk(
                        chunk_id=f"{target.doc_id}#{slug}-{sub_idx}",
                        doc_id=target.doc_id,
                        title=target.title,
                        section=section_title,
                        text=full_text,
                        metadata=metadata,
                    )
                )
                sub_idx += 1
                # Keep trailing overlap words for context continuity across long sections
                current_words = current_words[-overlap_words:] if overlap_words > 0 else []

            current_words.extend(s_words)

        if current_words:
            body_str = " ".join(current_words)
            full_text = f"{header}\nSection: {section_title}\n{body_str}"
            chunks.append(
                Chunk(
                    chunk_id=f"{target.doc_id}#{slug}-{sub_idx}",
                    doc_id=target.doc_id,
                    title=target.title,
                    section=section_title,
                    text=full_text,
                    metadata=metadata,
                )
            )

    return chunks
