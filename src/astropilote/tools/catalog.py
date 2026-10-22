"""Catalog & RAG retrieval tools (`search_catalog`, `get_object`)."""

from __future__ import annotations

import json
import re
from functools import lru_cache

from astropilote.agent.schemas import CATALOG_NAMES, CatalogName
from astropilote.config import settings
from astropilote.retrieval.hybrid import HybridRetriever, MetadataFilter

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
    ic_match = re.match(r"^IC0*(\d{1,4})$", s)
    if ic_match:
        return f"IC{int(ic_match.group(1)):04d}"
    return raw_id.strip().upper()


@lru_cache(maxsize=1)
def load_documents_lookup() -> tuple[dict[str, dict], dict[str, str]]:
    """Load `data/processed/documents.jsonl` and build `(by_doc_id, alias_to_doc_id)` maps."""
    docs_path = settings.data_dir / "processed" / "documents.jsonl"
    by_id: dict[str, dict] = {}
    alias_map: dict[str, str] = {}

    if not docs_path.exists() or docs_path.stat().st_size == 0:
        from astropilote.ingest.run_ingest import run_pipeline

        run_pipeline(offline_fallback=True)
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


def search_catalog(
    query: str,
    catalog: str | None = None,
    object_type: str | None = None,
    constellation: str | None = None,
    max_magnitude: float | None = None,
    narrowband_only: bool = False,
    top_k: int = 5,
    exclude_guides: bool = False,
) -> list[dict]:
    """Search the astronomy corpus (objects, constellations, practical guides) with hybrid search + reranking.

    Pass `catalog="Guide"` to search only practical observing / astrophotography guides.
    """
    retriever = get_retriever()
    valid_cat: CatalogName | None = (
        catalog  # type: ignore[assignment]
        if catalog in CATALOG_NAMES
        else None
    )
    filters = MetadataFilter(
        catalog=valid_cat,
        object_type=object_type,
        constellation=constellation,
        max_magnitude=max_magnitude,
        narrowband_only=narrowband_only,
        exclude_catalog="Guide" if exclude_guides else None,
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


def get_object(
    object_id: str,
    focal_length_mm: float | None = None,
    sensor_format: str = "APS-C",
) -> dict:
    """Retrieve exact coordinates, magnitude, angular size, and FOV framing for a specific object.

    Args:
        object_id: Canonical ID or alias, e.g. 'M031', 'M31', 'C014', 'NGC 7000', 'Pleiades'.
        focal_length_mm: Optional telescope/lens focal length in mm to compute camera FOV framing.
        sensor_format: Camera sensor size ('APS-C' = 23.5x15.6 mm or 'full-frame' = 36x24 mm).
    """
    record = resolve_object_record(object_id)
    if record is None:
        return {
            "found": False,
            "object_id": object_id,
            "message": f"Object '{object_id}' is not in the AstroPilote corpus.",
        }

    meta = dict(record.get("metadata", {}))
    framing_advice = None
    if focal_length_mm and focal_length_mm > 0 and meta.get("size_major_arcmin"):
        is_ff = str(sensor_format).lower().startswith("full")
        sensor_w = 36.0 if is_ff else 23.5
        sensor_h = 24.0 if is_ff else 15.6
        sensor_label = "full-frame (36 x 24 mm)" if is_ff else "APS-C"
        fov_w = round((57.2958 * 60.0 * sensor_w) / focal_length_mm, 1)
        fov_h = round((57.2958 * 60.0 * sensor_h) / focal_length_mm, 1)
        obj_size = float(meta["size_major_arcmin"])
        fill_ratio = round((obj_size / fov_w) * 100.0, 1)
        if obj_size > fov_w * 1.1:
            framing_advice = (
                f"No, the whole target ({obj_size:.0f} arcminutes / {obj_size}') does not fit in the frame: "
                f"with a {focal_length_mm:.0f} mm focal length, a {sensor_label} sensor covers only about "
                f"{fov_w} x {fov_h} arcminutes ({fov_w}'x{fov_h}'), smaller than the target's roughly "
                f"{obj_size:.0f} arcminutes, so a mosaic or a much shorter focal length is required."
            )
        elif fill_ratio >= 25.0:
            framing_advice = f"Ideal framing: target ({obj_size}') fills {fill_ratio}% of {sensor_label} width ({fov_w}'x{fov_h}')."
        else:
            framing_advice = (
                f"Compact target ({obj_size}') fills {fill_ratio}% of {sensor_label} width ({fov_w}'x{fov_h}'); "
                "crop or use longer focal length."
            )

    return {
        "found": True,
        "doc_id": record["doc_id"],
        "title": record["title"],
        "catalog": record["catalog"],
        **meta,
        "framing_advice": framing_advice,
    }
