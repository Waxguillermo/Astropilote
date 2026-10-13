"""Catalog & RAG retrieval tools (`search_catalog`, `get_object`)."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Literal

from astropilote.config import settings
from astropilote.retrieval.hybrid import HybridRetriever, MetadataFilter
from astropilote.tools.base import adk_tool

_retriever_singleton: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """Return a lazy singleton `HybridRetriever` instance."""
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = HybridRetriever()
    return _retriever_singleton


def normalize_object_id(raw_id: str) -> str:
    """Normalize shorthand designations like 'M31' -> 'M031', 'C14' -> 'C014', 'NGC 7000' -> 'NGC7000'."""
    s = raw_id.strip().upper().replace(" ", "")
    m_match = re.match(r"^(?:MESSIER|M)(\d{1,3})$", s)
    if m_match:
        return f"M{int(m_match.group(1)):03d}"
    c_match = re.match(r"^(?:CALDWELL|C)(\d{1,3})$", s)
    if c_match:
        return f"C{int(c_match.group(1)):03d}"
    ngc_match = re.match(r"^NGC0*(\d{1,4})$", s)
    if ngc_match:
        return f"NGC{int(ngc_match.group(1)):04d}"
    return raw_id.strip().upper()


@lru_cache(maxsize=1)
def load_documents_lookup() -> tuple[dict[str, dict], dict[str, str]]:
    """Load `data/processed/documents.jsonl` and build `(by_doc_id, alias_to_doc_id)` maps."""
    docs_path = settings.data_dir / "processed" / "documents.jsonl"
    by_id: dict[str, dict] = {}
    alias_map: dict[str, str] = {}

    if not docs_path.exists():
        return by_id, alias_map

    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_id = doc["doc_id"]
            by_id[doc_id] = doc
            alias_map[doc_id.upper()] = doc_id
            alias_map[doc["title"].upper()] = doc_id
            for alias in doc.get("metadata", {}).get("aliases", []):
                alias_map[alias.strip().upper()] = doc_id
                alias_map[alias.strip().upper().replace(" ", "")] = doc_id

    return by_id, alias_map


def resolve_object_record(identifier: str) -> dict | None:
    """Resolve an object ID, NGC number, or common name to its document record."""
    by_id, alias_map = load_documents_lookup()
    norm = normalize_object_id(identifier)
    if norm in by_id:
        return by_id[norm]
    key = identifier.strip().upper()
    if key in alias_map:
        return by_id[alias_map[key]]
    compact = key.replace(" ", "")
    if compact in alias_map:
        return by_id[alias_map[compact]]
    return None


@adk_tool
def search_catalog(
    query: str,
    catalog: str | None = None,
    object_type: str | None = None,
    constellation: str | None = None,
    max_magnitude: float | None = None,
    narrowband_only: bool = False,
    top_k: int = 5,
) -> list[dict]:
    """Search the 506-document astronomy & astrophotography corpus using hybrid search + reranking.

    Args:
        query: Natural language query or object names (e.g. "bright planetary nebula for small refractor").
        catalog: Optional filter ("Messier", "Caldwell", "Herschel", or "Constellation").
        object_type: Optional filter ("galaxy", "emission_nebula", "planetary_nebula", "globular_cluster", "open_cluster", "supernova_remnant").
        constellation: Optional 3-letter IAU constellation filter (e.g. "CYG", "ORI", "AND", "HER").
        max_magnitude: Optional upper limit on visual magnitude (lower means brighter, e.g. 8.5).
        narrowband_only: Set True to restrict to H-alpha / OIII emission & planetary nebulae or supernova remnants.
        top_k: Number of chunks to return (default 5).
    """
    retriever = get_retriever()
    valid_cat: Literal["Messier", "Caldwell", "Herschel", "Constellation"] | None = (
        catalog  # type: ignore[assignment]
        if catalog in ("Messier", "Caldwell", "Herschel", "Constellation")
        else None
    )
    filters = MetadataFilter(
        catalog=valid_cat,
        object_type=object_type,
        constellation=constellation,
        max_magnitude=max_magnitude,
        narrowband_only=narrowband_only,
    )
    chunks = retriever.search(query=query, top_k=top_k, filters=filters)
    return [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "title": c.title,
            "section": c.section,
            "score": c.score,
            "text": c.text,
            "metadata": c.metadata.model_dump(),
        }
        for c in chunks
    ]


@adk_tool
def get_object(object_id: str, focal_length_mm: float | None = None) -> dict:
    """Retrieve exact coordinates, magnitude, angular size, and FOV framing for a specific object.

    Args:
        object_id: Canonical ID or alias, e.g. 'M031', 'M31', 'C014', 'NGC 7000', 'Pleiades'.
        focal_length_mm: Optional telescope/lens focal length in mm to compute APS-C camera FOV framing.
    """
    record = resolve_object_record(object_id)
    if record is None:
        return {
            "found": False,
            "object_id": object_id,
            "message": f"Object '{object_id}' is not in the AstroPilote 506-document corpus.",
        }

    meta = dict(record.get("metadata", {}))
    framing_advice = None
    if focal_length_mm and focal_length_mm > 0 and meta.get("size_major_arcmin"):
        # Standard APS-C sensor (23.5 mm x 15.6 mm) field of view in arcminutes
        fov_w = round((57.2958 * 60.0 * 23.5) / focal_length_mm, 1)
        fov_h = round((57.2958 * 60.0 * 15.6) / focal_length_mm, 1)
        obj_size = float(meta["size_major_arcmin"])
        fill_ratio = round((obj_size / fov_w) * 100.0, 1)
        if obj_size > fov_w * 1.1:
            framing_advice = f"Target ({obj_size}') exceeds APS-C FOV ({fov_w}'x{fov_h}'); requires a mosaic or shorter focal length."
        elif fill_ratio >= 25.0:
            framing_advice = f"Ideal framing: target ({obj_size}') fills {fill_ratio}% of APS-C width ({fov_w}'x{fov_h}')."
        else:
            framing_advice = f"Compact target ({obj_size}') fills {fill_ratio}% of APS-C width ({fov_w}'x{fov_h}'); crop or use longer focal length."

    return {
        "found": True,
        "doc_id": record["doc_id"],
        "title": record["title"],
        "catalog": record["catalog"],
        **meta,
        "framing_advice": framing_advice,
    }
