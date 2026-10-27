"""FastAPI backend for AstroPilote (Anthropic API + Local Ollama + WCS Sky Vision).

Usage:
    uv run python -m astropilote.app.main
"""

from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from astropilote import __version__
from astropilote.agent.graph import run_agent
from astropilote.agent.schemas import EquipmentProfile
from astropilote.config import settings
from astropilote.ingest.catalog import CALDWELL_TO_OPENNGC, IAU_CONSTELLATIONS
from astropilote.llm import (
    LLMUnavailableError,
    check_llm_status,
    get_torch_device,
    save_anthropic_key,
)
from astropilote.tools.catalog import (
    get_object,
    get_retriever,
    load_documents_lookup,
    resolve_object_record,
)
from astropilote.tools.sky import compute_night_ephemeris, visibility
from astropilote.tools.world import geocode, weather
from astropilote.vision.service import (
    VISION_PRESETS,
    PhotoError,
    PhotoMetadata,
    get_or_create_preset_image,
    run_vision_pipeline,
)

STATIC_DIR = Path(__file__).parent / "static"

PRESET_SCENARIOS: list[dict[str, str]] = [
    {
        "id": "fontainebleau_session",
        "label": "Tonight's Sky Guide (Fontainebleau)",
        "subtitle": "Beginner session · What to see & where to look",
        "mode": "agent",
        "query": (
            "Saturday night, October 17, 2026, I will be in Fontainebleau as a beginner with binoculars "
            "or a small telescope. What are the best objects to see in the sky tonight and where should I look?"
        ),
    },
    {
        "id": "m31_satellites",
        "label": "Andromeda & Companion Galaxies (M31 · M32 · M110)",
        "subtitle": "Closest giant spiral · Naked eye & binoculars",
        "mode": "rag",
        "query": (
            "What is the Andromeda Galaxy (M31), what are its two brightest companion galaxies (M32 and M110), "
            "and can a beginner see them with the naked eye or binoculars?"
        ),
    },
    {
        "id": "chamonix_narrowband",
        "label": "Glowing Nebulae in Chamonix (Cygnus Wide-Field)",
        "subtitle": "Aug 15, 2026 · Star-forming clouds in Cygnus",
        "mode": "agent",
        "query": (
            "I am in Chamonix on August 15, 2026. Which bright glowing nebulae or star clouds in the constellation "
            "Cygnus can I observe tonight?"
        ),
    },
    {
        "id": "southern_trap_c080",
        "label": "Why Latitude Matters (Omega Centauri from Paris)",
        "subtitle": "Declination geometry · Can we see C80 at 48.8°N?",
        "mode": "agent",
        "query": "Can I observe Omega Centauri (Caldwell 80) from Paris on May 15, 2026 at midnight? Explain simply why or why not.",
    },
    {
        "id": "out_of_corpus_ic1101",
        "label": "Out-of-Corpus Guardrail (IC 1101 in Abell 2029)",
        "subtitle": "Tests zero-hallucination abstention",
        "mode": "agent",
        "query": (
            "What can you observe of the supergiant elliptical galaxy IC 1101 in Abell 2029 with an amateur telescope?"
        ),
    },
    {
        "id": "impossible_forecast_2028",
        "label": "2028 Weather Forecast Guardrail",
        "subtitle": "Refuses impossible >14-day meteorological prediction",
        "mode": "agent",
        "query": "Will the sky be completely cloud-free in Fontainebleau on March 3, 2028 at 23:00?",
    },
]

EQUIPMENT_PRESETS: list[dict[str, Any]] = [
    {
        "preset_id": "apo_80ed_eq",
        "rig_name": "80mm f/6 APO Refractor + APS-C + Equatorial GoTo",
        "telescope_or_lens": "80/480mm f/6 Apochromatic Refractor",
        "focal_length_mm": 480.0,
        "aperture_mm": 80.0,
        "camera_model": "Astro-modified APS-C Mirrorless / DSLR",
        "sensor_format": "APS-C",
        "mount_type": "equatorial_goto",
        "has_autoguiding": True,
        "filters": ["broadband", "dual_band"],
        "bortle_class": 4,
    },
    {
        "preset_id": "tele_135_tracker",
        "rig_name": "135mm f/2 Telephoto + Full-Frame + Star Tracker",
        "telescope_or_lens": "Samyang 135mm f/2 ED Telephoto Lens",
        "focal_length_mm": 135.0,
        "aperture_mm": 67.5,
        "camera_model": "Full-Frame 36×24 mm Camera",
        "sensor_format": "full-frame",
        "mount_type": "star_tracker",
        "has_autoguiding": False,
        "filters": ["broadband", "dual_band"],
        "bortle_class": 4,
    },
    {
        "preset_id": "newton_150_750",
        "rig_name": "150/750 f/5 Newtonian + Cooled APS-C Camera",
        "telescope_or_lens": "150/750mm f/5 Newtonian Reflector with Coma Corrector",
        "focal_length_mm": 750.0,
        "aperture_mm": 150.0,
        "camera_model": "ZWO ASI2600MC Pro (Cooled APS-C)",
        "sensor_format": "APS-C",
        "mount_type": "equatorial_goto",
        "has_autoguiding": True,
        "filters": ["broadband", "dual_band", "narrowband_sho"],
        "bortle_class": 5,
    },
    {
        "preset_id": "sct_c8_1280",
        "rig_name": 'Celestron C8 (1280mm f/6.3) + 1" Astro Camera',
        "telescope_or_lens": "Celestron C8 203mm + f/6.3 Focal Reducer (1280mm)",
        "focal_length_mm": 1280.0,
        "aperture_mm": 203.0,
        "camera_model": 'ZWO ASI533MC Pro (1" Sensor)',
        "sensor_format": "1-inch",
        "mount_type": "equatorial_goto",
        "has_autoguiding": True,
        "filters": ["broadband", "uhc_cls"],
        "bortle_class": 5,
    },
    {
        "preset_id": "dslr_50mm_tripod",
        "rig_name": "50mm f/1.8 Lens + APS-C Camera on Fixed Tripod",
        "telescope_or_lens": "50mm f/1.8 Prime Lens",
        "focal_length_mm": 50.0,
        "aperture_mm": 28.0,
        "camera_model": "APS-C Mirrorless / DSLR",
        "sensor_format": "APS-C",
        "mount_type": "tripod",
        "has_autoguiding": False,
        "filters": ["broadband"],
        "bortle_class": 4,
    },
    {
        "preset_id": "seestar_s50",
        "rig_name": "Smart Telescope 50/250mm (Seestar / Alt-Az GoTo)",
        "telescope_or_lens": "Integrated 50/250mm f/5 APO Triplet (Seestar / Dwarf)",
        "focal_length_mm": 250.0,
        "aperture_mm": 50.0,
        "camera_model": "Built-in IMX462 CMOS Sensor",
        "sensor_format": "1-inch",
        "mount_type": "altaz_goto",
        "has_autoguiding": False,
        "filters": ["broadband", "dual_band"],
        "bortle_class": 6,
    },
    {
        "preset_id": "smartphone_night",
        "rig_name": "Smartphone on Tripod (Night / Astro Mode)",
        "telescope_or_lens": "Smartphone Main Wide Lens (24mm eq. f/1.8)",
        "focal_length_mm": 24.0,
        "aperture_mm": 13.3,
        "camera_model": "Smartphone Night / RAW Sensor",
        "sensor_format": "smartphone",
        "mount_type": "tripod",
        "has_autoguiding": False,
        "filters": ["broadband"],
        "bortle_class": 5,
    },
]


class ChatRequest(BaseModel):
    """Request body for `/api/chat`."""

    question: str = Field(..., min_length=2)
    mode: Literal["agent", "rag"] = Field(default="agent")
    llm_mode: Literal["api", "local"] | None = Field(default=None)
    date_override: str | None = Field(default=None)
    focal_length_mm: float | None = Field(default=480.0)
    aperture_mm: float | None = Field(default=80.0)
    equipment_profile: EquipmentProfile | None = Field(default=None)


class ApiKeyRequest(BaseModel):
    """Request body for `/api/config/key`."""

    api_key: str = Field(..., min_length=5)


class EphemerisPreviewRequest(BaseModel):
    """Request body for `/api/ephemeris`."""

    place_name: str = Field(..., min_length=1)
    obs_date: str = Field(default_factory=lambda: settings.get_reference_date().isoformat())
    focal_length_mm: float = 480.0
    aperture_mm: float = 80.0
    object_ids: list[str] = Field(default_factory=lambda: ["M031", "M013", "M027", "C014", "M045"])


class VisionIdentifyRequest(BaseModel):
    """Request body for `/api/vision/identify`."""

    preset_id: str | None = Field(default=None)
    image_base64: str | None = Field(default=None)
    filename: str | None = Field(default="sky_capture.png")
    ra_hint_deg: float | None = Field(default=None)
    dec_hint_deg: float | None = Field(default=None)
    focal_length_mm: float | None = Field(default=None)
    sensor_width_mm: float | None = Field(default=None)
    field_width_deg: float | None = Field(default=None)
    pixel_scale_arcsec: float | None = Field(default=None)
    telescope: str | None = Field(default=None)
    camera: str | None = Field(default=None)
    exposure_seconds: float | None = Field(default=None)
    trust_fits_wcs: bool = Field(default=False)


_CONSTELLATION_WIKI_MAP: dict[str, str] = {
    f"CON_{code}": wtitle for code, _cname, wtitle, _ra, _dec in IAU_CONSTELLATIONS
}


def _resolve_wikipedia_info(rec: dict[str, Any]) -> tuple[str, str]:
    """Return `(wikipedia_page_title, wikipedia_url)` for a corpus document record."""
    doc_id = str(rec.get("doc_id") or "")
    title = str(rec.get("title") or doc_id)
    if doc_id in _CONSTELLATION_WIKI_MAP:
        wpage = _CONSTELLATION_WIKI_MAP[doc_id]
    elif doc_id == "M045":
        wpage = "Pleiades"
    elif doc_id.startswith("M") and doc_id[1:].isdigit():
        m_num = int(doc_id[1:])
        wpage = f"Messier {m_num}"
    elif doc_id.startswith("C") and doc_id[1:].isdigit():
        c_num = int(doc_id[1:])
        ngc_key, fallback_common = CALDWELL_TO_OPENNGC.get(c_num, ("", title))
        if ngc_key.startswith("NGC"):
            wpage = f"NGC {ngc_key[3:].lstrip('0')}"
        elif ngc_key.startswith("IC"):
            wpage = f"IC {ngc_key[2:].lstrip('0')}"
        else:
            wpage = fallback_common
    elif doc_id.startswith("NGC") and doc_id[3:].isdigit():
        wpage = f"NGC {int(doc_id[3:])}"
    elif doc_id.startswith("IC") and doc_id[2:].isdigit():
        wpage = f"IC {int(doc_id[2:])}"
    else:
        wpage = str(rec.get("wikipedia_title") or title).split("(")[0].strip() or title

    slug = quote(wpage.replace(" ", "_"), safe="()_:-'")
    return wpage, f"https://en.wikipedia.org/wiki/{slug}"


def _get_doc_chunks(doc_id: str) -> list[dict[str, Any]]:
    """Return all indexed chunks for `doc_id` in reading order."""
    retriever = get_retriever()
    retriever._ensure_bm25_loaded()
    out: list[dict[str, Any]] = []
    for ch in retriever._chunks:
        if ch.doc_id == doc_id:
            out.append(
                {
                    "chunk_id": ch.chunk_id,
                    "section": ch.section,
                    "text": ch.text,
                }
            )
    return out


def _serialize_catalog_source(
    object_id: str,
    *,
    is_cited: bool = True,
    focal_length_mm: float = 480.0,
    aperture_mm: float | None = 80.0,
    sensor_format: str = "APS-C",
    mount_type: str = "equatorial_goto",
    has_autoguiding: bool = False,
    filters: list[str] | None = None,
    vision_pixel: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Resolve a catalog document into a rich source inspector dictionary with Wikipedia links & chunks."""
    rec = resolve_object_record(object_id)
    if rec is None:
        return None
    doc_id = rec["doc_id"]
    obj_info = get_object(
        object_id=doc_id,
        focal_length_mm=focal_length_mm,
        sensor_format=sensor_format,
        aperture_mm=aperture_mm,
        mount_type=mount_type,
        has_autoguiding=has_autoguiding,
        filters=filters,
    )
    cat_name = str(obj_info.get("catalog") or "Deep-Sky Catalog")
    const_name = str(obj_info.get("constellation") or "Sky")
    otype = str(obj_info.get("object_type") or "object").replace("_", " ").title()
    wpage, wurl = _resolve_wikipedia_info(rec)
    doc_chunks = _get_doc_chunks(doc_id)
    full_text = rec.get("text") or "\n\n".join(
        f"### {c['section']}\n{c['text']}" for c in doc_chunks
    )
    hierarchy = f"{cat_name} › {const_name} › {otype} › Wikipedia ({wpage})"
    return {
        **obj_info,
        "hierarchy": hierarchy,
        "wikipedia_page": wpage,
        "wikipedia_url": wurl,
        "num_chunks": rec.get("num_chunks", len(doc_chunks)),
        "chunks": doc_chunks,
        "full_text": full_text,
        "is_cited": is_cited,
        "vision_pixel": vision_pixel,
    }


app = FastAPI(
    title="AstroPilote API",
    description="Open-source Hybrid RAG, Ephemeris Planner & WCS Sky Vision Copilot.",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    """Return system status, active LLM mode (Anthropic vs Local Ollama), and corpus metrics."""
    by_id, _ = load_documents_lookup()
    retriever = get_retriever()
    retriever._ensure_bm25_loaded()
    llm_info = check_llm_status()
    localizable_count = sum(
        1
        for rec in by_id.values()
        if (rec.get("metadata") or {}).get("ra_deg") is not None
        and (rec.get("metadata") or {}).get("dec_deg") is not None
    )
    return {
        "status": "ok",
        "llm": llm_info,
        "accelerator_device": get_torch_device(),
        "corpus_documents": len(by_id),
        "localizable_documents": localizable_count,
        "indexed_chunks": len(retriever._chunks),
        "has_dense_qdrant": retriever.has_dense_index(),
    }


@app.post("/api/config/key")
def api_save_key(req: ApiKeyRequest) -> dict[str, Any]:
    """Save `ANTHROPIC_API_KEY` into `.env` and return updated LLM status."""
    llm_info = save_anthropic_key(req.api_key)
    return {"status": "ok", "llm": llm_info}


@app.get("/api/presets")
def api_presets() -> dict[str, Any]:
    """Return curated preset questions, vision presets, and photography equipment presets."""
    return {
        "presets": PRESET_SCENARIOS,
        "vision_presets": VISION_PRESETS,
        "equipment_presets": EQUIPMENT_PRESETS,
    }


@app.get("/api/equipment/presets")
def api_equipment_presets() -> dict[str, Any]:
    """Return curated photography equipment profile presets."""
    return {"presets": EQUIPMENT_PRESETS}


@app.get("/api/corpus")
def api_corpus() -> dict[str, Any]:
    """Return all 567 Wikipedia & OpenNGC corpus documents with Wikipedia page titles and links."""
    by_id, _ = load_documents_lookup()
    retriever = get_retriever()
    retriever._ensure_bm25_loaded()

    first_excerpt_by_doc: dict[str, str] = {}
    for ch in retriever._chunks:
        if ch.doc_id not in first_excerpt_by_doc:
            first_excerpt_by_doc[ch.doc_id] = ch.text[:220]

    docs_out: list[dict[str, Any]] = []
    breakdown: dict[str, int] = {
        "Messier": 0,
        "Caldwell": 0,
        "Herschel": 0,
        "Astrophoto": 0,
        "Constellation": 0,
        "Guide": 0,
    }
    for doc_id, rec in by_id.items():
        cat = str(rec.get("catalog") or "Messier")
        breakdown[cat] = breakdown.get(cat, 0) + 1
        meta = rec.get("metadata", {}) or {}
        wpage, wurl = _resolve_wikipedia_info(rec)
        docs_out.append(
            {
                "doc_id": doc_id,
                "title": rec.get("title", doc_id),
                "catalog": cat,
                "object_type": meta.get("object_type", "deep_sky_object"),
                "constellation": meta.get("constellation", ""),
                "ra_deg": meta.get("ra_deg"),
                "dec_deg": meta.get("dec_deg"),
                "magnitude": meta.get("magnitude"),
                "size_major_arcmin": meta.get("size_major_arcmin"),
                "narrowband_friendly": bool(meta.get("narrowband_friendly")),
                "num_chunks": rec.get("num_chunks", 4),
                "wikipedia_page": wpage,
                "wikipedia_url": wurl,
                "excerpt": first_excerpt_by_doc.get(doc_id, ""),
            }
        )

    return {
        "total_documents": len(docs_out),
        "total_chunks": len(retriever._chunks),
        "catalog_breakdown": breakdown,
        "sources": {
            "wikipedia": "https://en.wikipedia.org/w/api.php (CC BY-SA 4.0)",
            "openngc": "https://github.com/mattiaverga/OpenNGC (CC-BY-SA-4.0)",
        },
        "documents": docs_out,
    }


@app.get("/api/vision/presets")
def api_vision_presets() -> dict[str, Any]:
    """Return curated sky images with WCS metadata for 1-click Vision testing."""
    return {"presets": VISION_PRESETS}


@app.post("/api/vision/identify")
def api_vision_identify(req: VisionIdentifyRequest) -> dict[str, Any]:
    """Identify catalog objects inside an uploaded sky photo or preset image using Alessandro's tool."""
    try:
        if req.preset_id:
            preset, img_path = get_or_create_preset_image(req.preset_id)
            ra_hint = req.ra_hint_deg if req.ra_hint_deg is not None else preset["ra_hint_deg"]
            dec_hint = req.dec_hint_deg if req.dec_hint_deg is not None else preset["dec_hint_deg"]
            focal = (
                req.focal_length_mm
                if req.focal_length_mm is not None
                else preset["focal_length_mm"]
            )
            sensor_w = (
                req.sensor_width_mm
                if req.sensor_width_mm is not None
                else preset["sensor_width_mm"]
            )
            fov_w = (
                req.field_width_deg
                if req.field_width_deg is not None
                else preset["field_width_deg"]
            )
            meta = PhotoMetadata(
                ra_hint_deg=ra_hint,
                dec_hint_deg=dec_hint,
                focal_length_mm=focal,
                sensor_width_mm=sensor_w,
                field_width_deg=fov_w if req.pixel_scale_arcsec is None else None,
                pixel_scale_arcsec=req.pixel_scale_arcsec,
                telescope=req.telescope or preset.get("telescope"),
                camera=req.camera or preset.get("camera"),
                exposure_seconds=req.exposure_seconds or preset.get("exposure_seconds"),
            )
            out = run_vision_pipeline(
                img_path, meta, trust_fits_wcs=req.trust_fits_wcs, is_preset=True
            )
            preset_Meta = preset
        elif req.image_base64:
            raw_b64 = req.image_base64
            if "," in raw_b64 and raw_b64.strip().startswith("data:"):
                raw_b64 = raw_b64.split(",", 1)[1]
            img_bytes = base64.b64decode(raw_b64)
            safe_name = Path(req.filename or "uploaded_sky.png").name
            meta = PhotoMetadata(
                ra_hint_deg=req.ra_hint_deg,
                dec_hint_deg=req.dec_hint_deg,
                focal_length_mm=req.focal_length_mm,
                sensor_width_mm=req.sensor_width_mm,
                field_width_deg=req.field_width_deg if req.pixel_scale_arcsec is None else None,
                pixel_scale_arcsec=req.pixel_scale_arcsec,
                telescope=req.telescope,
                camera=req.camera,
                exposure_seconds=req.exposure_seconds,
            )
            with tempfile.TemporaryDirectory(prefix="astropilote-upload-") as tmp_dir:
                upload_path = Path(tmp_dir) / safe_name
                upload_path.write_bytes(img_bytes)
                out = run_vision_pipeline(upload_path, meta, trust_fits_wcs=req.trust_fits_wcs)
            preset_Meta = None
        else:
            raise PhotoError("Provide either preset_id or image_base64.")
    except (PhotoError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    res = out["result"]
    focal_for_framing = float(
        res.get("input_metadata", {}).get("focal_length_mm") or req.focal_length_mm or 480.0
    )
    sources_full: list[dict[str, Any]] = []
    for obj in res.get("objects", []):
        src = _serialize_catalog_source(
            obj["doc_id"],
            is_cited=True,
            focal_length_mm=focal_for_framing,
            vision_pixel=obj.get("pixel"),
        )
        if src:
            sources_full.append(src)

    return {
        "status": "ok",
        "preset": preset_Meta,
        "vision": res,
        "original_image_data_url": out["original_image_data_url"],
        "annotated_image_data_url": out["annotated_image_data_url"],
        "csv_content": out["csv_content"],
        "trace_steps": out["trace_steps"],
        "sources_full": sources_full,
    }


@app.get("/api/catalog/{object_id}")
def api_get_catalog_object(
    object_id: str,
    focal_length_mm: float = 480.0,
    aperture_mm: float = 80.0,
    sensor_format: str = "APS-C",
    mount_type: str = "equatorial_goto",
    has_autoguiding: bool = False,
) -> dict[str, Any]:
    """Return full document record, metadata, FOV framing, and shooting guide for a specific catalog ID."""
    src = _serialize_catalog_source(
        object_id,
        is_cited=True,
        focal_length_mm=focal_length_mm,
        aperture_mm=aperture_mm,
        sensor_format=sensor_format,
        mount_type=mount_type,
        has_autoguiding=has_autoguiding,
    )
    if src is None:
        raise HTTPException(status_code=404, detail=f"Object '{object_id}' not found in corpus.")
    return src


@app.post("/api/ephemeris")
def api_ephemeris_preview(req: EphemerisPreviewRequest) -> dict[str, Any]:
    """Compute night twilight window, weather, and target altitude curves."""
    geo = geocode(place_name=req.place_name)
    if not geo.get("found"):
        raise HTTPException(
            status_code=404,
            detail=geo.get("message") or f"Location '{req.place_name}' could not be geocoded.",
        )
    lat = float(geo["lat"])
    lon = float(geo["lon"])

    night_info = compute_night_ephemeris(lat=lat, lon=lon, obs_date_str=req.obs_date)
    vis_info = visibility(
        object_ids=req.object_ids,
        lat=lat,
        lon=lon,
        obs_date=req.obs_date,
        focal_length_mm=req.focal_length_mm,
    )
    weather_info = weather(lat=lat, lon=lon, obs_date=req.obs_date)
    return {
        "location": geo,
        "night": night_info,
        "weather": weather_info,
        "visibility": vis_info,
    }


@app.post("/api/chat")
def api_chat(req: ChatRequest) -> dict[str, Any]:
    """Execute the AstroPilote workflow on `req.question`."""
    prof = (
        req.equipment_profile if (req.equipment_profile and req.equipment_profile.enabled) else None
    )
    eff_focal = prof.focal_length_mm if prof else req.focal_length_mm
    eff_aperture = prof.aperture_mm if prof else req.aperture_mm
    try:
        answer, state, eval_row = run_agent(
            req.question,
            question_id="web_session",
            llm_mode=req.llm_mode,
            mode=req.mode,
            date_override=req.date_override,
            focal_length_mm=eff_focal,
            aperture_mm=eff_aperture,
            equipment_profile=prof,
        )
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    vis_objects: dict[str, Any] = {}
    vis_date: str | None = None
    night_payload: dict[str, Any] = {}
    weather_payload: dict[str, Any] = {}
    tool_outputs_list: list[str] = []

    for msg in state.get("messages", []):
        if getattr(msg, "role", "") == "tool":
            tool_outputs_list.append(str(getattr(msg, "content", ""))[:360])
            try:
                data = json.loads(msg.content)
                if msg.name == "visibility":
                    vis_objects = data.get("objects", {})
                    vis_date = data.get("date")
                elif msg.name == "night":
                    night_payload = data
                elif msg.name == "weather":
                    weather_payload = data
            except (json.JSONDecodeError, TypeError):
                pass

    if vis_date:
        daily_n = night_payload.get("daily_nights")
        if isinstance(daily_n, list):
            matched_n = next((n for n in daily_n if n.get("date") == vis_date), None)
            if isinstance(matched_n, dict):
                night_payload = {**night_payload, **matched_n, "daily_nights": daily_n}
        daily_w = weather_payload.get("daily_forecasts")
        if isinstance(daily_w, list):
            matched_w = next((w for w in daily_w if w.get("date") == vis_date), None)
            if isinstance(matched_w, dict):
                weather_payload = {**weather_payload, **matched_w, "daily_forecasts": daily_w}

    intent = state.get("intent")
    retrieved_chunks = [c.model_dump(mode="json") for c in state.get("retrieved", [])]

    cited_ids = list(answer.sources or [])
    retrieved_doc_ids = list(
        dict.fromkeys(c["doc_id"] for c in retrieved_chunks if c.get("doc_id"))
    )
    ordered_ids = list(dict.fromkeys(cited_ids + retrieved_doc_ids))
    cited_set = set(cited_ids)

    focal_mm = float(
        (intent.focal_length_mm if intent and intent.focal_length_mm else None)
        or eff_focal
        or 480.0
    )
    aperture_mm = float(
        (intent.aperture_mm if intent and intent.aperture_mm else None) or eff_aperture or 80.0
    )
    sensor_fmt = (
        intent.sensor_format
        if intent and intent.sensor_format
        else (prof.sensor_format if prof else "APS-C")
    )
    mount_type = prof.mount_type if prof else (intent.mount_type if intent else "equatorial_goto")
    has_autoguiding = bool(
        prof.has_autoguiding if prof else (intent.has_autoguiding if intent else False)
    )
    filters_list = list(prof.filters if prof else (intent.filters if intent else []))

    sources_full: list[dict[str, Any]] = []
    for did in ordered_ids:
        src = _serialize_catalog_source(
            did,
            is_cited=(did in cited_set),
            focal_length_mm=focal_mm,
            aperture_mm=aperture_mm,
            sensor_format=sensor_fmt,
            mount_type=mount_type,
            has_autoguiding=has_autoguiding,
            filters=filters_list,
        )
        if src:
            sources_full.append(src)

    tool_observations: list[dict[str, Any]] = []
    for idx, tc in enumerate(eval_row.tool_calls, start=1):
        obs_preview = tool_outputs_list[idx - 1] if (idx - 1) < len(tool_outputs_list) else ""
        tool_observations.append(
            {
                "step": idx,
                "name": tc.name,
                "arguments": tc.arguments,
                "latency_ms": round(tc.duration_s * 1000.0, 1),
                "result": obs_preview,
            }
        )

    return {
        "mode": req.mode,
        "answer": answer.model_dump(mode="json"),
        "telemetry": eval_row.model_dump(mode="json"),
        "tool_observations": tool_observations,
        "intent": intent.model_dump(mode="json") if intent else None,
        "equipment_profile": prof.model_dump(mode="json") if prof else None,
        "night": night_payload,
        "weather": weather_payload,
        "visibility_objects": vis_objects,
        "retrieved_chunks": retrieved_chunks,
        "sources_full": sources_full,
    }


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/favicon.ico", include_in_schema=False)
def serve_favicon() -> Response:
    """Return 204 No Content for browser favicon requests."""
    return Response(status_code=204)


@app.get("/")
def serve_index() -> FileResponse:
    """Serve the clean AstroPilote Web UI."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Static frontend index.html not found.")
    return FileResponse(
        str(index_path),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


def main() -> None:
    """Launch the AstroPilote server on http://127.0.0.1:8000."""
    print("🔭 AstroPilote running at http://127.0.0.1:8000")
    uvicorn.run("astropilote.app.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
