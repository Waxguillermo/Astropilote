"""Catalog & RAG retrieval tools (`search_catalog`, `get_object`)."""

from __future__ import annotations

import json
import re
from functools import lru_cache

from astropilote.agent.schemas import CATALOG_NAMES, CatalogName
from astropilote.config import settings
from astropilote.ingest.catalog import IAU_CONSTELLATIONS
from astropilote.retrieval.hybrid import HybridRetriever, MetadataFilter

_retriever_singleton: HybridRetriever | None = None

_CONSTELLATION_NAMES: dict[str, str] = {code: cname for code, cname, _, _, _ in IAU_CONSTELLATIONS}

# (width_mm, height_mm, label_for_framing, crop_factor)
SENSOR_SPECS: dict[str, tuple[float, float, str, float]] = {
    "full-frame": (36.0, 24.0, "full-frame (36 x 24 mm)", 1.0),
    "aps-c": (23.5, 15.6, "APS-C", 1.5),
    "micro-4/3": (17.3, 13.0, "Micro 4/3 (17.3 x 13.0 mm)", 2.0),
    "1-inch": (13.2, 8.8, "1-inch (13.2 x 8.8 mm)", 2.7),
    "smartphone": (7.0, 5.2, "Smartphone (7.0 x 5.2 mm)", 5.1),
}

_APPEARANCE_BY_TYPE: dict[str, str] = {
    "galaxy": (
        "Appears visually and on single short frames as a soft, milky oval glow with a condensed bright core; "
        "stacking multiple exposures reveals faint spiral arms, dust lanes, and outer halo structure."
    ),
    "galaxy_pair": (
        "Two interacting galactic cores visible in the same field as twin fuzzy condensations; "
        "long stacked exposures bring out tidal tails and faint connecting bridges of stars."
    ),
    "galaxy_triplet": (
        "Three distinct galactic nuclei sharing the field of view, each showing a bright core and differently angled disk."
    ),
    "emission_nebula": (
        "Glowing cloud of ionized hydrogen (H-alpha) and oxygen (OIII). Visually subtle or greyish-misty, "
        "but in long-exposure photography it reveals rich crimson/magenta H-alpha filaments, dark Bok globules, and teal OIII arcs."
    ),
    "planetary_nebula": (
        "Compact, high-surface-brightness glowing shell or ring of ionized gas around a hot central white dwarf, "
        "showing vivid teal-green (OIII 500.7 nm) and red (H-alpha) hues on camera."
    ),
    "reflection_nebula": (
        "Delicate, ghostly blue dust cloud reflecting starlight from nearby hot young stars; "
        "requires broadband (no narrowband filter) exposures to capture its subtle azure filaments."
    ),
    "supernova_remnant": (
        "Intricate, lacy network of shock-excited glowing filaments (crimson H-alpha and electric-blue OIII) "
        "spanning a wide area of sky."
    ),
    "dark_nebula": (
        "Dense silhouette of interstellar dust blocking background Milky Way starlight; "
        "appears as a striking dark void carved against a rich starfield."
    ),
    "globular_cluster": (
        "Tightly packed, symmetrical sphere of tens to hundreds of thousands of ancient stars; "
        "looks like a bright unresolved fuzzball at low power, resolving into a glittering 'sugar-pile' of golden and blue stars on properly exposed photos."
    ),
    "open_cluster": (
        "Loose, sparkling grouping of dozens to hundreds of young stars of contrasting blue-white and golden spectral colors "
        "against the background sky."
    ),
    "cluster_with_nebulosity": (
        "Bright open star cluster embedded inside a glowing emission or reflection nebula — combining pinpoint jewel-like stars "
        "with surrounding gaseous filaments."
    ),
}


def resolve_sensor_specs(sensor_format: str | None) -> tuple[float, float, str, float]:
    """Return `(width_mm, height_mm, label, crop_factor)` for a sensor format string."""
    raw = str(sensor_format or "APS-C").strip().lower()
    if raw.startswith("full"):
        return SENSOR_SPECS["full-frame"]
    if raw in SENSOR_SPECS:
        return SENSOR_SPECS[raw]
    return SENSOR_SPECS["aps-c"]


def _format_ra_hms(ra_deg: float | None) -> str | None:
    if ra_deg is None:
        return None
    total_hours = (float(ra_deg) % 360.0) / 15.0
    h = int(total_hours)
    m = (total_hours - h) * 60.0
    return f"{h:02d}h {m:04.1f}m"


def _format_dec_dms(dec_deg: float | None) -> str | None:
    if dec_deg is None:
        return None
    val = float(dec_deg)
    sign = "+" if val >= 0 else "-"
    abs_val = abs(val)
    d = int(abs_val)
    m = int(round((abs_val - d) * 60.0))
    if m == 60:
        d += 1
        m = 0
    return f"{sign}{d:02d}° {m:02d}'"


def compute_shooting_guide(
    record: dict,
    focal_length_mm: float | None = None,
    aperture_mm: float | None = None,
    sensor_format: str = "APS-C",
    mount_type: str = "equatorial_goto",
    has_autoguiding: bool = False,
    filters: list[str] | None = None,
) -> dict:
    """Compute a deterministic orientation, visual/photo appearance, and camera settings guide for `record`."""
    meta = dict(record.get("metadata") or {})
    doc_id = str(record.get("doc_id") or "")
    otype = str(meta.get("object_type") or "deep_sky_object")
    const_code = str(meta.get("constellation") or "").upper()
    const_name = _CONSTELLATION_NAMES.get(const_code, const_code or "Sky")
    ra_deg = meta.get("ra_deg")
    dec_deg = meta.get("dec_deg")
    mag = meta.get("magnitude")
    size_maj = meta.get("size_major_arcmin")
    size_min = meta.get("size_minor_arcmin")
    nb_friendly = bool(meta.get("narrowband_friendly"))

    # 1. Orientation & Finding
    ra_hms = _format_ra_hms(ra_deg)
    dec_dms = _format_dec_dms(dec_deg)
    if size_maj:
        moon_ratio = round(float(size_maj) / 30.0, 2)
        dim_str = f"{size_maj}' × {size_min}'" if size_min else f"{size_maj}'"
        if moon_ratio >= 1.0:
            size_vs_moon = f"{dim_str} (~{moon_ratio}× the Full Moon's 30' diameter)"
        else:
            size_vs_moon = (
                f"{dim_str} (~{int(round(moon_ratio * 100))}% of the Full Moon's 30' diameter)"
            )
    else:
        size_vs_moon = "Compact / point-like or unmeasured angular size"

    if mag is not None and float(mag) <= 5.5:
        finding_hint = (
            f"Bright target (mag {mag}): visible to the naked eye under dark skies and unmistakable in binoculars "
            f"or a finder scope within {const_name} ({const_code}). Take a 2–5s high-ISO test shot to center the core."
        )
    elif mag is not None and float(mag) <= 8.5:
        finding_hint = (
            f"Medium brightness (mag {mag}): easily spotted in 10×50 binoculars or a 50mm finder scope in {const_name} ({const_code}); "
            "use star-hopping from bright constellation stars or plate-solving / GoTo to center."
        )
    else:
        mag_lbl = f"mag {mag}" if mag is not None else "faint diffuse surface brightness"
        finding_hint = (
            f"Faint photographic target ({mag_lbl}) in {const_name} ({const_code}): invisible to the naked eye; "
            "use GoTo + plate-solving (or careful star-hopping + a 10–20s ISO 3200 test exposure) to verify framing on the bright star pattern."
        )

    # 2. Appearance
    appearance_desc = _APPEARANCE_BY_TYPE.get(
        otype,
        "Deep-sky astronomical target showing stellar or nebulous structure across stacked exposures.",
    )
    if doc_id in ("M042", "M031"):
        appearance_desc += (
            " Note: Has a very high-dynamic-range core that can saturate on long exposures — combine short (10s–30s) "
            "and long exposures (HDR) to preserve core detail."
        )

    # 3. Camera, FOV & Exposure Settings
    eff_focal = float(focal_length_mm) if (focal_length_mm and focal_length_mm > 0) else 480.0
    eff_aperture = float(aperture_mm) if (aperture_mm and aperture_mm > 0) else None
    f_ratio = round(eff_focal / eff_aperture, 1) if eff_aperture else None
    sensor_w, sensor_h, sensor_label, crop_factor = resolve_sensor_specs(sensor_format)
    fov_w = round((57.2958 * 60.0 * sensor_w) / eff_focal, 1)
    fov_h = round((57.2958 * 60.0 * sensor_h) / eff_focal, 1)
    fill_pct = round((float(size_maj) / fov_w) * 100.0, 1) if size_maj else None

    # Exposure & ISO based on mount_type
    m_type = str(mount_type or "equatorial_goto").lower()
    if m_type == "tripod":
        npf_sec = round(max(0.5, min(15.0, 500.0 / (eff_focal * crop_factor))), 1)
        sub_exp = (
            f"{npf_sec}s max per frame (untracked tripod 500-rule limit at {eff_focal:.0f}mm on {sensor_label} "
            "to prevent star trails; shoot 200–500 short frames in RAW and recenter the target every 25–30 shots)"
        )
        iso_gain = "ISO 1600–6400 (wide-open aperture or stopped down 0.5 stop)"
    elif m_type == "altaz_goto":
        sub_exp = (
            "10s – 20s per sub-exposure (Alt-Az GoTo tracking; keeps individual subs short to avoid field rotation "
            "at the edges while stacking 100–300 frames)"
        )
        iso_gain = "ISO 800–1600 or high-conversion gain (Gain 100–200)"
    elif m_type == "star_tracker":
        rec_s = (
            "60s – 120s" if eff_focal <= 150 else ("30s – 60s" if eff_focal <= 350 else "20s – 45s")
        )
        sub_exp = (
            f"{rec_s} per sub-exposure on portable Star Tracker (verify polar alignment carefully; "
            f"{'use autoguiding in RA' if has_autoguiding else 'discard any wind-trailed frames'})"
        )
        iso_gain = "ISO 800–1600 (or Unity Gain 100)"
    else:
        if has_autoguiding:
            rec_s = "180s – 300s (or 120s in broadband)"
            guide_lbl = "Equatorial GoTo with autoguiding + dithering every 3 frames"
        else:
            rec_s = "60s – 120s" if eff_focal <= 600 else "30s – 60s"
            guide_lbl = "Equatorial GoTo (unguided — refine polar alignment)"
        if doc_id in ("M042", "M031", "M013"):
            rec_s += " + 15s–30s short subs to avoid saturating the bright core"
        sub_exp = f"{rec_s} ({guide_lbl})"
        iso_gain = "ISO 800 (DSLR/Mirrorless) or Unity Gain 100 at -10°C (cooled astro CMOS)"

    # Filter advice
    user_filters = {str(f).lower() for f in (filters or [])}
    has_dual = any(
        k in user_filters for k in ("dual_band", "dual-band", "l-extreme", "l-enhance", "ha_oiii")
    )
    has_sho = any(k in user_filters for k in ("narrowband_sho", "sho", "h-alpha", "ha"))
    has_uhc = any(k in user_filters for k in ("uhc", "uhc_cls", "cls"))

    if nb_friendly:
        if has_dual:
            filter_adv = (
                "Use your Dual-Band (Hα + OIII) filter: isolates the 656.3 nm H-alpha and 500.7 nm OIII emission lines, "
                "dramatically boosting nebula contrast against light pollution and moonlight."
            )
        elif has_sho:
            filter_adv = "Use your Narrowband (Hα / OIII / SII) filters for maximum emission-line contrast, even under bright Moon."
        elif has_uhc:
            filter_adv = "Use your UHC / CLS filter to suppress urban skyglow while passing H-beta, OIII, and H-alpha emission."
        else:
            filter_adv = (
                "Narrowband-friendly emission target (Hα / OIII): can be imaged in broadband RGB under dark skies, "
                "or with a Dual-Band (Hα+OIII) / UHC filter to strongly boost nebulosity contrast."
            )
    else:
        if has_dual or has_sho:
            filter_adv = (
                "Broadband continuous-spectrum target: DO NOT use your Dual-Band or narrowband filter (it would block "
                "most starlight/galaxy continuum). Shoot unfiltered / UV-IR cut only (or mild broadband L-Pro)."
            )
        else:
            filter_adv = (
                "Broadband continuous-spectrum target: shoot with UV/IR-cut (no narrowband filter) to preserve natural "
                "star colors and galactic/reflection continuum; prioritize moonless dark hours."
            )

    return {
        "orientation": {
            "constellation_code": const_code,
            "constellation_name": const_name,
            "ra_hms": ra_hms,
            "dec_dms": dec_dms,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "size_vs_full_moon": size_vs_moon,
            "how_to_find": finding_hint,
        },
        "appearance": {
            "object_type": otype,
            "visual_and_photo_look": appearance_desc,
        },
        "camera_settings": {
            "focal_length_mm": eff_focal,
            "aperture_mm": eff_aperture,
            "f_ratio": f"f/{f_ratio}" if f_ratio else None,
            "sensor_format": sensor_label,
            "fov_arcmin": f"{fov_w} x {fov_h} arcminutes ({fov_w}'x{fov_h}')",
            "fov_deg": f"{round(fov_w / 60.0, 2)}° x {round(fov_h / 60.0, 2)}°",
            "target_fill_pct": fill_pct,
            "mount_type": m_type,
            "recommended_sub_exposure": sub_exp,
            "recommended_iso_or_gain": iso_gain,
            "filter_recommendation": filter_adv,
            "calibration_tip": "Focus on a bright star using Live View 10× zoom or a Bahtinov mask, and capture 20 Darks, 25 Flats, and 30 Bias frames.",
        },
    }


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
    aperture_mm: float | None = None,
    mount_type: str = "equatorial_goto",
    has_autoguiding: bool = False,
    filters: list[str] | None = None,
) -> dict:
    """Retrieve exact coordinates, magnitude, angular size, FOV framing, and shooting guide for a specific object.

    Args:
        object_id: Canonical ID or alias, e.g. 'M031', 'M31', 'C014', 'NGC 7000', 'Pleiades'.
        focal_length_mm: Optional telescope/lens focal length in mm to compute camera FOV framing.
        sensor_format: Camera sensor size ('APS-C', 'full-frame', 'micro-4/3', '1-inch', 'smartphone').
        aperture_mm: Optional telescope/lens aperture diameter in mm.
        mount_type: Mount tracking type ('equatorial_goto', 'star_tracker', 'altaz_goto', 'tripod').
        has_autoguiding: Whether autoguiding is active.
        filters: Optional list of available filters.
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
        sensor_w, sensor_h, sensor_label, _ = resolve_sensor_specs(sensor_format)
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

    shooting_guide = compute_shooting_guide(
        record=record,
        focal_length_mm=focal_length_mm,
        aperture_mm=aperture_mm,
        sensor_format=sensor_format,
        mount_type=mount_type,
        has_autoguiding=has_autoguiding,
        filters=filters,
    )

    return {
        "found": True,
        "doc_id": record["doc_id"],
        "title": record["title"],
        "catalog": record["catalog"],
        **meta,
        "framing_advice": framing_advice,
        "shooting_guide": shooting_guide,
    }
