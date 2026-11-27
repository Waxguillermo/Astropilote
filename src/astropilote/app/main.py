"""FastAPI backend for AstroPilote (Anthropic API + Local Ollama + WCS Sky Vision).

Usage:
    uv run python -m astropilote.app.main
"""

from __future__ import annotations

import base64
import inspect
import json
import re
import tempfile
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import astropilote.agent.graph as graph_mod
from astropilote import __version__
from astropilote.agent.graph import get_astro_graph, run_agent
from astropilote.agent.schemas import ConversationTurn, EquipmentProfile
from astropilote.app.routers import auth_router, logbook_router
from astropilote.auth import (
    enforce_rate_limit,
    get_current_user,
    resolve_astrometry_api_key,
    resolve_llm_api_key,
)
from astropilote.backend.approvals import PendingApprovals
from astropilote.config import settings
from astropilote.db import User, get_db, init_db
from astropilote.device import get_compute_device
from astropilote.export import (
    export_booking_ics,
    export_plan_csv,
    export_plan_ics,
    safe_download_filename,
)
from astropilote.ingest.catalog import CALDWELL_TO_OPENNGC, IAU_CONSTELLATIONS
from astropilote.llm import (
    LLMUnavailableError,
    check_llm_status,
    save_anthropic_key,
    use_request_api_key,
)
from astropilote.logbook import append_chat_message
from astropilote.optics import EQUIPMENT_PRESETS
from astropilote.tools.catalog import (
    get_object,
    get_retriever,
    load_documents_lookup,
    resolve_object_record,
)
from astropilote.tools.sky import compute_night_ephemeris, visibility
from astropilote.tools.world import geocode, weather
from astropilote.tracing import setup_tracing
from astropilote.vision.service import (
    MAX_BYTES,
    VISION_PRESETS,
    PhotoError,
    PhotoMetadata,
    get_or_create_preset_image,
    run_vision_pipeline,
)
from astropilote.vision.solvers import detect_available_solvers

_DEFAULT_RUN_AGENT = run_agent
_DEFAULT_RUN_AGENT_STREAM = getattr(graph_mod, "run_agent_stream", None)
run_agent_stream = _DEFAULT_RUN_AGENT_STREAM

STATIC_DIR = Path(__file__).parent / "static"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SEEDS_DIR = PROJECT_ROOT / "seeds"
_OBSERVATORY_SEEDS = ("observatory_demo.json", "two_bookings.json")
OBSERVATORY_DB_PATH = PROJECT_ROOT / "data" / "backend" / "observatory.sqlite3"
OBSERVATORY_CHECKPOINT_PATH = PROJECT_ROOT / "data" / "backend" / "agentic_checkpoints.sqlite3"
OBSERVATORY_MEMORY_PATH = PROJECT_ROOT / "data" / "backend" / "user_memory.sqlite3"
_APPROVAL_TTL_S = 600
_APPROVAL_CLAIM_ERRORS = {
    "missing": (404, "No approval is pending for this thread."),
    "forbidden": (403, "This approval request belongs to another user."),
    "expired": (410, "This approval request expired; submit the cancellation again."),
}
_OBSERVATORY_GATEWAY_LOCK = threading.Lock()
_observatory_gateway: Any | None = None

_DEFAULT_CORS_ORIGINS = ["http://127.0.0.1:8000", "http://localhost:8000"]
ALLOWED_CORS_ORIGINS: list[str] = [
    origin.strip()
    for origin in settings.cors_origins.split(",")
    if origin.strip() and origin.strip() != "*"
] or list(_DEFAULT_CORS_ORIGINS)

_LOCAL_CLIENT_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}
_ALLOWED_VISION_SUFFIXES = {".png", ".jpg", ".jpeg", ".fits", ".fit"}
_CHUNK_HEADER_RE = re.compile(r"^\[[^\]]*\]\s*(?:Section:[^\n]*\n?)?")
_STREAM_END = object()

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


class ChatRequest(BaseModel):
    """Request body for `/api/chat` and `/api/chat/stream`."""

    question: str = Field(..., min_length=2, max_length=4000)
    mode: Literal["agent", "rag"] = Field(default="agent")
    llm_mode: Literal["api", "local"] | None = Field(default=None)
    date_override: str | None = Field(default=None, max_length=32)
    focal_length_mm: float | None = Field(default=None, gt=0)
    aperture_mm: float | None = Field(default=None, gt=0)
    equipment_profile: EquipmentProfile | None = Field(default=None)
    history: list[ConversationTurn] = Field(default_factory=list, max_length=10)
    thread_id: str | None = Field(default=None, max_length=64)


class ApiKeyRequest(BaseModel):
    """Request body for `/api/config/key`."""

    api_key: str = Field(
        ...,
        min_length=12,
        max_length=256,
        pattern=r"^sk-(?:ant|or)-[A-Za-z0-9_-]+$",
    )


class EphemerisPreviewRequest(BaseModel):
    """Request body for `/api/ephemeris`."""

    place_name: str = Field(..., min_length=1, max_length=200)
    obs_date: str = Field(default_factory=lambda: settings.get_reference_date().isoformat())
    focal_length_mm: float = 480.0
    aperture_mm: float = 80.0
    object_ids: list[str] = Field(
        default_factory=lambda: ["M031", "M013", "M027", "C014", "M045"],
        max_length=25,
    )


class VisionIdentifyRequest(BaseModel):
    """Request body for `/api/vision/identify`."""

    preset_id: str | None = Field(default=None, max_length=100)
    image_base64: str | None = Field(default=None, max_length=45_000_000)
    filename: str | None = Field(default="sky_capture.png", max_length=255)
    ra_hint_deg: float | None = Field(default=None)
    dec_hint_deg: float | None = Field(default=None)
    focal_length_mm: float | None = Field(default=None)
    sensor_width_mm: float | None = Field(default=None)
    field_width_deg: float | None = Field(default=None)
    pixel_scale_arcsec: float | None = Field(default=None)
    telescope: str | None = Field(default=None, max_length=200)
    camera: str | None = Field(default=None, max_length=200)
    exposure_seconds: float | None = Field(default=None)
    trust_fits_wcs: bool = Field(default=False)


class ObservatoryResetRequest(BaseModel):
    """Request body for `/api/observatory/reset`."""

    seed_name: Literal["observatory_demo.json", "two_bookings.json"] = Field(
        default="observatory_demo.json"
    )
    user_id: str = Field(default="astro-club", min_length=1, max_length=64)
    clear_memory: bool = Field(default=False)


class ObservatoryRunRequest(BaseModel):
    """Request body for `/api/observatory/run`."""

    text: str = Field(..., min_length=1, max_length=2000)
    user_id: str = Field(default="astro-club", min_length=1, max_length=64)
    llm_mode: Literal["offline", "local", "api"] = Field(default="offline")
    reference_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class ObservatoryApproveRequest(BaseModel):
    """Request body for `/api/observatory/approve`."""

    thread_id: str = Field(..., min_length=3, max_length=128)
    approved: bool
    user_id: str = Field(default="astro-club", min_length=1, max_length=64)


class ExportPlanItem(BaseModel):
    """Target entry submitted to `/api/export/session`."""

    object_id: str = Field(default="", max_length=64)
    name: str = Field(default="", max_length=200)
    start: str = Field(default="", max_length=16)
    end: str = Field(default="", max_length=16)
    altitude_deg: float | None = Field(default=None, ge=0.0, le=90.0)
    direction: str = Field(default="", max_length=32)
    imaging_notes: str | None = Field(default=None, max_length=1000)


class SessionExportRequest(BaseModel):
    """Request body for `/api/export/session` (`.ics` calendar or N.I.N.A. `.csv`)."""

    format: Literal["ics", "csv"] = Field(default="ics")
    obs_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    location_name: str = Field(default="Observation Site", max_length=200)
    timezone: str | None = Field(default=None, max_length=64)
    night: dict[str, Any] = Field(default_factory=dict)
    weather: dict[str, Any] = Field(default_factory=dict)
    plan: list[ExportPlanItem] = Field(default_factory=list, max_length=50)


class BookingExportRequest(BaseModel):
    """Request body for `/api/export/booking` (`.ics` invite for an observatory reservation)."""

    booking_id: str = Field(..., min_length=1, max_length=64)
    telescope_name: str = Field(..., min_length=1, max_length=200)
    start_utc: str = Field(..., min_length=10, max_length=64)
    end_utc: str = Field(..., min_length=10, max_length=64)
    site: str = Field(default="Lyon Observatory", max_length=200)
    aperture_mm: float | None = Field(default=None, gt=0.0)
    price_cents: int | None = Field(default=None, ge=0)


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
    get_fn = getattr(retriever, "get_doc_chunks", None)
    if callable(get_fn):
        raw_chunks = get_fn(doc_id)
        out_fast: list[dict[str, Any]] = []
        for ch in raw_chunks:
            if isinstance(ch, dict):
                out_fast.append(
                    {
                        "chunk_id": ch["chunk_id"],
                        "section": ch.get("section", ""),
                        "text": ch.get("text", ""),
                    }
                )
            else:
                out_fast.append(
                    {
                        "chunk_id": ch.chunk_id,
                        "section": ch.section,
                        "text": ch.text,
                    }
                )
        return out_fast

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


def _to_json_dict(obj: Any) -> Any:
    """Convert a Pydantic model instance to a JSON-compatible dict, or return `obj` as-is."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from either a mapping or an object attribute."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _accepts_kwarg(fn: Any, kwarg_name: str) -> bool:
    """Return True if `fn` accepts `kwarg_name` explicitly or via `**kwargs`."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == kwarg_name and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def _prepare_chat_call(
    req: ChatRequest, fn: Any
) -> tuple[EquipmentProfile | None, float | None, float | None, dict[str, Any]]:
    """Build normalized equipment settings and keyword arguments for `run_agent` / `run_agent_stream`."""
    prof = (
        req.equipment_profile if (req.equipment_profile and req.equipment_profile.enabled) else None
    )
    eff_focal = prof.focal_length_mm if prof else req.focal_length_mm
    eff_aperture = prof.aperture_mm if prof else req.aperture_mm
    kwargs: dict[str, Any] = {
        "question_id": "web_session",
        "llm_mode": req.llm_mode,
        "mode": req.mode,
        "date_override": req.date_override,
        "focal_length_mm": eff_focal,
        "aperture_mm": eff_aperture,
        "equipment_profile": prof,
    }
    if _accepts_kwarg(fn, "conversation_history"):
        kwargs["conversation_history"] = req.history
    return prof, eff_focal, eff_aperture, kwargs


def _invoke_run_agent(
    req: ChatRequest,
) -> tuple[Any, dict[str, Any], Any, EquipmentProfile | None, float | None, float | None]:
    """Execute synchronous `run_agent` with optional conversation history support."""
    prof, eff_focal, eff_aperture, kwargs = _prepare_chat_call(req, run_agent)
    answer, state, eval_row = run_agent(req.question, **kwargs)
    return answer, state, eval_row, prof, eff_focal, eff_aperture


def _build_chat_response_payload(
    *,
    req: ChatRequest,
    answer: Any,
    state: dict[str, Any],
    eval_row: Any,
    prof: EquipmentProfile | None,
    eff_focal: float | None,
    eff_aperture: float | None,
) -> dict[str, Any]:
    """Assemble the complete JSON response dictionary shared by `/api/chat` and `/api/chat/stream`."""
    vis_objects: dict[str, Any] = {}
    vis_date: str | None = None
    night_payload: dict[str, Any] = {}
    weather_payload: dict[str, Any] = {}
    tool_outputs_list: list[str] = []

    for msg in state.get("messages", []):
        role = _get_field(msg, "role", "")
        if role == "tool":
            content = str(_get_field(msg, "content", ""))
            msg_name = _get_field(msg, "name", "")
            tool_outputs_list.append(content[:360])
            try:
                data = json.loads(content)
                if msg_name == "visibility":
                    vis_objects = data.get("objects", {})
                    vis_date = data.get("date")
                elif msg_name == "night":
                    night_payload = data
                elif msg_name == "weather":
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
    retrieved_chunks = [_to_json_dict(c) for c in state.get("retrieved", [])]

    answer_dict = _to_json_dict(answer) if answer is not None else {}
    cited_ids = list(_get_field(answer_dict, "sources") or _get_field(answer, "sources") or [])
    retrieved_doc_ids = list(
        dict.fromkeys(
            c["doc_id"] for c in retrieved_chunks if isinstance(c, dict) and c.get("doc_id")
        )
    )
    ordered_ids = list(dict.fromkeys(cited_ids + retrieved_doc_ids))
    cited_set = set(cited_ids)

    intent_focal = _get_field(intent, "focal_length_mm") if intent else None
    intent_aperture = _get_field(intent, "aperture_mm") if intent else None
    intent_sensor = _get_field(intent, "sensor_format") if intent else None
    intent_mount = _get_field(intent, "mount_type") if intent else None
    intent_guiding = _get_field(intent, "has_autoguiding") if intent else None
    intent_filters = _get_field(intent, "filters") if intent else None

    focal_mm = float(intent_focal or eff_focal or 480.0)
    aperture_mm = float(intent_aperture or eff_aperture or 80.0)
    sensor_fmt = intent_sensor or (prof.sensor_format if prof else "APS-C")
    mount_type = prof.mount_type if prof else (intent_mount or "equatorial_goto")
    has_autoguiding = bool(
        prof.has_autoguiding if prof else (intent_guiding if intent_guiding is not None else False)
    )
    filters_list = list(prof.filters if prof else (intent_filters or []))

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

    eval_dict = _to_json_dict(eval_row) if eval_row is not None else {}
    raw_tool_calls = (_get_field(eval_row, "tool_calls") if eval_row is not None else []) or []

    tool_observations: list[dict[str, Any]] = []
    for idx, tc in enumerate(raw_tool_calls, start=1):
        obs_preview = tool_outputs_list[idx - 1] if (idx - 1) < len(tool_outputs_list) else ""
        tool_observations.append(
            {
                "step": idx,
                "name": _get_field(tc, "name", ""),
                "arguments": _get_field(tc, "arguments", {}) or {},
                "status": _get_field(tc, "status", "ok") or "ok",
                "latency_ms": round(float(_get_field(tc, "duration_s", 0.0) or 0.0) * 1000.0, 1),
                "result": obs_preview,
            }
        )

    harness_payload = {
        "engine": "langgraph",
        "steps": int(state.get("steps", 0) or 0),
        "react_thoughts": list(state.get("react_thoughts") or []),
        "events": list(state.get("harness_events") or []),
        "budget_tripped": state.get("harness_budget_tripped"),
    }

    return {
        "mode": req.mode,
        "answer": answer_dict,
        "telemetry": eval_dict,
        "harness": harness_payload,
        "tool_observations": tool_observations,
        "intent": _to_json_dict(intent) if intent else None,
        "equipment_profile": prof.model_dump(mode="json") if prof else None,
        "night": night_payload,
        "weather": weather_payload,
        "visibility_objects": vis_objects,
        "retrieved_chunks": retrieved_chunks,
        "sources_full": sources_full,
    }


def _format_sse_event(
    ev: dict[str, Any],
    *,
    req: ChatRequest,
    prof: EquipmentProfile | None,
    eff_focal: float | None,
    eff_aperture: float | None,
) -> str:
    """Serialize a streaming agent event dictionary into a Server-Sent Events `data:` frame."""
    if ev.get("event") == "done":
        if (
            "sources_full" in ev
            and "tool_observations" in ev
            and "telemetry" in ev
            and "answer" in ev
        ):
            done_payload = {"event": "done", **ev}
        else:
            answer_obj = ev.get("answer")
            state_obj = (
                ev.get("state") if ev.get("state") is not None else (ev.get("final_state") or {})
            )
            eval_obj = ev.get("eval_row") if ev.get("eval_row") is not None else ev.get("telemetry")
            final_payload = _build_chat_response_payload(
                req=req,
                answer=answer_obj,
                state=state_obj,
                eval_row=eval_obj,
                prof=prof,
                eff_focal=eff_focal,
                eff_aperture=eff_aperture,
            )
            done_payload = {"event": "done", **final_payload}
        return f"data: {json.dumps(jsonable_encoder(done_payload), ensure_ascii=False)}\n\n"
    return f"data: {json.dumps(jsonable_encoder(ev), ensure_ascii=False)}\n\n"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Initialize SQLAlchemy DB, preload corpus lookup and hybrid index on startup."""
    setup_tracing()
    init_db()
    load_documents_lookup()
    retriever = get_retriever()
    ensure_bm25 = getattr(retriever, "_ensure_bm25_loaded", None)
    if callable(ensure_bm25):
        ensure_bm25()
    try:
        yield
    finally:
        _close_observatory_gateway()
        close_fn = getattr(get_retriever(), "close", None)
        if callable(close_fn):
            close_fn()


app = FastAPI(
    title="AstroPilote API",
    description="Open-source Hybrid RAG, Ephemeris Planner, Observing Logbook & Blind Plate-Solving Copilot.",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)
app.include_router(auth_router)
app.include_router(logbook_router)


@app.middleware("http")
async def add_security_headers(request: Request, call_next: Any) -> Response:
    """Attach standard HTTP security headers to all API and static responses."""
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    """Return system status, active LLM mode (Anthropic vs Local Ollama), and corpus metrics."""
    by_id, _ = load_documents_lookup()
    retriever = get_retriever()
    retriever._ensure_bm25_loaded()
    llm_info = check_llm_status()
    solvers = detect_available_solvers()
    has_local_astrometry = solvers["astap"] or solvers["solve_field"]
    localizable_count = sum(
        1
        for rec in by_id.values()
        if (rec.get("metadata") or {}).get("ra_deg") is not None
        and (rec.get("metadata") or {}).get("dec_deg") is not None
    )
    return {
        "status": "ok",
        "version": __version__,
        "auth_mode": settings.auth_mode,
        "llm": {**llm_info, "has_local_astrometry": has_local_astrometry},
        "has_local_astrometry": has_local_astrometry,
        "vision_solvers": solvers,
        "accelerator_device": get_compute_device(),
        "corpus_documents": len(by_id),
        "localizable_documents": localizable_count,
        "indexed_chunks": len(retriever._chunks),
        "has_dense_index": retriever.has_dense_index(),
        "has_dense_qdrant": retriever.has_dense_index(),
    }


@app.post("/api/config/key")
def api_save_key(req: ApiKeyRequest, request: Request) -> dict[str, Any]:
    """Save `ANTHROPIC_API_KEY` into `.env` (restricted to local solo mode + loopback)."""
    client_host = request.client.host if request.client else None
    if settings.auth_mode == "cloud" or client_host not in _LOCAL_CLIENT_HOSTS:
        raise HTTPException(
            status_code=403,
            detail="Server .env API key configuration is disabled in cloud mode; use /api/auth/keys/{provider}.",
        )
    try:
        llm_info = save_anthropic_key(req.api_key)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "llm": llm_info}


@app.get("/api/presets")
async def api_presets() -> dict[str, Any]:
    """Return curated preset questions, vision presets, and photography equipment presets."""
    return {
        "presets": PRESET_SCENARIOS,
        "vision_presets": VISION_PRESETS,
        "equipment_presets": EQUIPMENT_PRESETS,
    }


@app.get("/api/equipment/presets")
async def api_equipment_presets() -> dict[str, Any]:
    """Return curated photography equipment profile presets."""
    return {"presets": EQUIPMENT_PRESETS}


@app.get("/api/graph")
def api_graph() -> dict[str, Any]:
    """Return the compiled LangGraph `StateGraph` topology and Mermaid diagram."""
    workflow = get_astro_graph()
    return {
        "engine": "langgraph",
        "nodes": ["guard", "refuse", "agent", "tools", "compose", "validate", "scrub"],
        "mermaid": workflow.get_mermaid(),
    }


@app.get("/api/corpus")
def api_corpus() -> dict[str, Any]:
    """Return all 567 Wikipedia & OpenNGC corpus documents with Wikipedia page titles and links."""
    by_id, _ = load_documents_lookup()
    retriever = get_retriever()
    retriever._ensure_bm25_loaded()

    first_excerpt_by_doc: dict[str, str] = {}
    for ch in retriever._chunks:
        if ch.doc_id not in first_excerpt_by_doc:
            cleaned = _CHUNK_HEADER_RE.sub("", ch.text or "").strip()
            first_excerpt_by_doc[ch.doc_id] = cleaned[:220]

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
async def api_vision_presets() -> dict[str, Any]:
    """Return curated sky images with WCS metadata for 1-click Vision testing."""
    return {"presets": VISION_PRESETS}


@app.post("/api/vision/identify")
def api_vision_identify(
    req: VisionIdentifyRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Identify catalog objects inside an uploaded sky photo or preset image."""
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    astro_key = resolve_astrometry_api_key(db, user_id=user.id)
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
            preset_meta = preset
        elif req.image_base64:
            raw_b64 = req.image_base64
            if "," in raw_b64 and raw_b64.strip().startswith("data:"):
                raw_b64 = raw_b64.split(",", 1)[1]
            img_bytes = base64.b64decode(raw_b64)
            if not img_bytes:
                raise PhotoError("Image file is empty.")
            if len(img_bytes) > MAX_BYTES:
                raise PhotoError(f"Image file is too large: maximum {MAX_BYTES} bytes.")

            raw_suffix = Path(req.filename or "").suffix.lower()
            ext = raw_suffix if raw_suffix in _ALLOWED_VISION_SUFFIXES else ".png"

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
                upload_path = Path(tmp_dir) / f"upload{ext}"
                upload_path.write_bytes(img_bytes)
                out = run_vision_pipeline(
                    upload_path,
                    meta,
                    trust_fits_wcs=req.trust_fits_wcs,
                    astrometry_api_key=astro_key,
                )
            preset_meta = None
        else:
            raise PhotoError("Provide either preset_id or image_base64.")
    except (PhotoError, ValueError, OSError) as exc:
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
        "preset": preset_meta,
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


@app.post("/api/export/session")
def api_export_session(req: SessionExportRequest) -> Response:
    """Export a night observation plan as an RFC 5545 `.ics` calendar or N.I.N.A./Telescopius `.csv`."""
    items = [item.model_dump() for item in req.plan]
    raw_date = req.obs_date or str(req.night.get("date") or "")[:10]
    try:
        obs_d = date.fromisoformat(raw_date) if raw_date else settings.get_reference_date()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid observation date.") from exc

    if req.format == "csv":
        body = export_plan_csv(items)
        filename = safe_download_filename(f"astropilote_nina_{obs_d.isoformat()}", "csv")
        return Response(
            content=body,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    body = export_plan_ics(
        items,
        obs_date=obs_d,
        location_name=req.location_name,
        tz_name=req.timezone,
        night=req.night,
        weather=req.weather,
    )
    filename = safe_download_filename(f"astropilote_session_{obs_d.isoformat()}", "ics")
    return Response(
        content=body,
        media_type="text/calendar; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/export/booking")
def api_export_booking(req: BookingExportRequest) -> Response:
    """Export a confirmed remote observatory booking as an RFC 5545 `.ics` calendar file."""
    try:
        body = export_booking_ics(
            booking_id=req.booking_id,
            telescope_name=req.telescope_name,
            start_utc=req.start_utc,
            end_utc=req.end_utc,
            site=req.site,
            aperture_mm=req.aperture_mm,
            price_cents=req.price_cents,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    filename = safe_download_filename(f"astropilote_{req.booking_id}", "ics")
    return Response(
        content=body,
        media_type="text/calendar; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/chat")
def api_chat(
    req: ChatRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Execute the AstroPilote workflow on `req.question` with hybrid auth, rate-limit, and BYOK scoping."""
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    scoped_key = resolve_llm_api_key(
        db, user_id=user.id, header_api_key=request.headers.get("X-API-Key")
    )
    try:
        with use_request_api_key(scoped_key):
            answer, state, eval_row, prof, eff_focal, eff_aperture = _invoke_run_agent(req)
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    payload = _build_chat_response_payload(
        req=req,
        answer=answer,
        state=state,
        eval_row=eval_row,
        prof=prof,
        eff_focal=eff_focal,
        eff_aperture=eff_aperture,
    )
    if req.thread_id is not None:
        saved_u = append_chat_message(
            db, user_id=user.id, thread_id=req.thread_id or None, role="user", content=req.question
        )
        append_chat_message(
            db,
            user_id=user.id,
            thread_id=saved_u["thread_id"],
            role="assistant",
            content=str(payload.get("answer") or ""),
            metadata={
                "citations": payload.get("citations", []),
                "verified": payload.get("verified"),
            },
        )
        payload["thread_id"] = saved_u["thread_id"]
    return payload


@app.post("/api/chat/stream")
def api_chat_stream(
    req: ChatRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Stream live stage, tool, token, and final verified response events via Server-Sent Events."""
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    scoped_key = resolve_llm_api_key(
        db, user_id=user.id, header_api_key=request.headers.get("X-API-Key")
    )
    sse_headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    graph_stream_fn = getattr(graph_mod, "run_agent_stream", None)
    if run_agent_stream is not _DEFAULT_RUN_AGENT_STREAM:
        stream_fn = run_agent_stream
    elif graph_stream_fn is not _DEFAULT_RUN_AGENT_STREAM:
        stream_fn = graph_stream_fn
    elif (
        run_agent is not _DEFAULT_RUN_AGENT
        or getattr(graph_mod, "run_agent", None) is not _DEFAULT_RUN_AGENT
    ):
        stream_fn = None
    else:
        stream_fn = run_agent_stream

    if callable(stream_fn):
        prof, eff_focal, eff_aperture, kwargs = _prepare_chat_call(req, stream_fn)
        try:
            with use_request_api_key(scoped_key):
                raw_stream = stream_fn(req.question, **kwargs)
        except LLMUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        if inspect.isasyncgen(raw_stream):

            async def async_event_generator() -> AsyncIterator[str]:
                with use_request_api_key(scoped_key):
                    try:
                        async for ev in raw_stream:
                            if isinstance(ev, dict):
                                yield _format_sse_event(
                                    ev,
                                    req=req,
                                    prof=prof,
                                    eff_focal=eff_focal,
                                    eff_aperture=eff_aperture,
                                )
                    except LLMUnavailableError as exc:
                        err_frame = {"event": "error", "status": 503, "detail": str(exc)}
                        yield f"data: {json.dumps(err_frame, ensure_ascii=False)}\n\n"

            return StreamingResponse(
                async_event_generator(),
                media_type="text/event-stream",
                headers=sse_headers,
            )

        stream_iter = iter(raw_stream)
        try:
            with use_request_api_key(scoped_key):
                first_ev = next(stream_iter, _STREAM_END)
        except LLMUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        def sync_event_generator() -> Iterator[str]:
            try:
                if first_ev is not _STREAM_END and isinstance(first_ev, dict):
                    yield _format_sse_event(
                        first_ev,
                        req=req,
                        prof=prof,
                        eff_focal=eff_focal,
                        eff_aperture=eff_aperture,
                    )
                while True:
                    with use_request_api_key(scoped_key):
                        ev = next(stream_iter, _STREAM_END)
                    if ev is _STREAM_END:
                        break
                    if isinstance(ev, dict):
                        yield _format_sse_event(
                            ev,
                            req=req,
                            prof=prof,
                            eff_focal=eff_focal,
                            eff_aperture=eff_aperture,
                        )
            except LLMUnavailableError as exc:
                err_frame = {"event": "error", "status": 503, "detail": str(exc)}
                yield f"data: {json.dumps(err_frame, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            sync_event_generator(),
            media_type="text/event-stream",
            headers=sse_headers,
        )

    def fallback_event_generator() -> Iterator[str]:
        initial_stage = {
            "event": "stage",
            "stage": "guard",
            "title": "Analyzing query & retrieving catalog sources",
        }
        yield f"data: {json.dumps(initial_stage, ensure_ascii=False)}\n\n"
        try:
            with use_request_api_key(scoped_key):
                answer, state, eval_row, prof, eff_focal, eff_aperture = _invoke_run_agent(req)
        except LLMUnavailableError as exc:
            err_frame = {"event": "error", "status": 503, "detail": str(exc)}
            yield f"data: {json.dumps(err_frame, ensure_ascii=False)}\n\n"
            return
        final_payload = _build_chat_response_payload(
            req=req,
            answer=answer,
            state=state,
            eval_row=eval_row,
            prof=prof,
            eff_focal=eff_focal,
            eff_aperture=eff_aperture,
        )
        done_payload = {"event": "done", **final_payload}
        yield f"data: {json.dumps(jsonable_encoder(done_payload), ensure_ascii=False)}\n\n"

    return StreamingResponse(
        fallback_event_generator(),
        media_type="text/event-stream",
        headers=sse_headers,
    )


def _require_local_client(request: Request) -> None:
    """Restrict legacy Observatory MCP studio operations to local solo mode + loopback."""
    client_host = request.client.host if request.client else None
    if settings.auth_mode == "cloud" or client_host not in _LOCAL_CLIENT_HOSTS:
        raise HTTPException(
            status_code=403,
            detail="Legacy Observatory studio is restricted to local solo mode.",
        )


def _ensure_observatory_seeded() -> Any:
    """Ensure the SQLite observatory backend exists and is seeded from `observatory_demo.json`."""
    from astropilote.backend.store import ObservatoryStore

    store = ObservatoryStore(OBSERVATORY_DB_PATH)
    if not OBSERVATORY_DB_PATH.exists() or not store.list_telescopes():
        store.reset_backend(SEEDS_DIR / "observatory_demo.json")
    return store


def _get_observatory_gateway() -> Any:
    """Return the shared stdio MCP gateway, respawned only if the backend database changed.

    Spawning the two MCP server subprocesses costs far more than a tool call, so the
    session is kept for the lifetime of the app and closed by the lifespan handler.
    """
    global _observatory_gateway
    from astropilote.agentic.workflow import StdioMCPGateway

    db_path = str(Path(OBSERVATORY_DB_PATH).resolve())
    with _OBSERVATORY_GATEWAY_LOCK:
        if _observatory_gateway is None or _observatory_gateway.db_path != db_path:
            if _observatory_gateway is not None:
                _observatory_gateway.close()
            _observatory_gateway = StdioMCPGateway(
                db_path, offline_sky=settings.observatory_offline_sky
            )
        return _observatory_gateway


def _close_observatory_gateway() -> None:
    global _observatory_gateway
    with _OBSERVATORY_GATEWAY_LOCK:
        if _observatory_gateway is not None:
            _observatory_gateway.close()
            _observatory_gateway = None


def _pending_approvals() -> PendingApprovals:
    return PendingApprovals(OBSERVATORY_CHECKPOINT_PATH)


_OBSERVATORY_MAX_STEPS = 6
_SUMMARY_KEYS = ("id", "status", "slot_id", "found", "place_name", "available")
_SUMMARY_MAX_CHARS = 160
_APPROVAL_REQUEST_KEYS = ("action", "booking_id", "risk", "question", "paid")


def _summarize_tool_result(data: Any) -> str:
    """Return a concise summary of a successful MCP tool result for the chat trace."""
    if isinstance(data, list):
        return f"{len(data)} row{'' if len(data) == 1 else 's'}"
    if isinstance(data, dict):
        parts = [f"{k}={data[k]}" for k in _SUMMARY_KEYS if data.get(k) not in (None, "")]
        return (", ".join(parts) or f"{len(data)} fields")[:_SUMMARY_MAX_CHARS]
    return str(data)[:_SUMMARY_MAX_CHARS]


class _OutcomeRecorder:
    """Wrap the shared MCP gateway and record each call outcome for the UI.

    Placed underneath `TracedGateway` (which only records `status`), it observes the
    same calls in the same order. It intentionally omits `close()` so the shared
    gateway remains alive for the lifetime of the app.
    """

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.outcomes: list[dict[str, str]] = []

    @property
    def last_attempts(self) -> Any:
        """Forward retry-layer attempts so `TracedGateway` above still records them."""
        return getattr(self.delegate, "last_attempts", None)

    def call(self, server: str, tool: str, args: dict[str, Any]) -> Any:
        try:
            data = self.delegate.call(server, tool, args)
        except Exception as exc:
            # `ToolFailure` carries `payload.error`; `MCPTransportError` carries `code`.
            payload = getattr(exc, "payload", None)
            error = payload.get("error") if isinstance(payload, dict) else None
            error = error if isinstance(error, dict) else {}
            self.outcomes.append(
                {
                    "error_code": str(
                        error.get("code") or getattr(exc, "code", None) or type(exc).__name__
                    ),
                    "summary": str(error.get("message") or "")[:_SUMMARY_MAX_CHARS],
                }
            )
            raise
        self.outcomes.append({"summary": _summarize_tool_result(data)})
        return data


def _with_outcomes(
    calls: list[dict[str, Any]], outcomes: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Merge `TracedGateway` calls with recorded outcomes, falling back to raw calls if misaligned."""
    if len(calls) != len(outcomes):
        return list(calls)
    return [{**call, **outcome} for call, outcome in zip(calls, outcomes, strict=True)]


def _approval_request(graph: Any, thread_id: str) -> dict[str, Any] | None:
    """Read the pending `interrupt()` payload (booking, risk, question) from the checkpoint."""
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    for pending in snapshot.interrupts:
        value = getattr(pending, "value", None)
        if isinstance(value, dict):
            return {k: value[k] for k in _APPROVAL_REQUEST_KEYS if k in value}
    return None


def _observatory_snapshot(store: Any, memory: Any, user_id: str) -> dict[str, Any]:
    """State fields shared by every Observatory studio response."""
    return {
        "state": store.get_state(),
        "preferred_telescope_id": memory.preferred_telescope(user_id),
        "preferred_site": memory.get_preference(user_id, "preferred_site"),
        "recent_history": [event.model_dump() for event in memory.recent_events(user_id)],
        "simulation": {
            "reference_date": settings.observatory_reference_date.isoformat(),
            "offline_sky": settings.observatory_offline_sky,
        },
    }


_MEMORY_WRITE_TOOLS = {
    ("set", "preferred_telescope_id"): ("set_preferred_telescope", "telescope_id"),
    ("clear", "preferred_telescope_id"): ("clear_preferred_telescope", None),
    ("set", "preferred_site"): ("set_preferred_site", "site"),
    ("clear", "preferred_site"): ("clear_preferred_site", None),
    ("clear", "history"): ("clear_history", None),
}


_HISTORY_READ_CUE = re.compile(
    r"\b(?:historique|history|derniere fois|last time|dernieres reservations|booking history)\b"
)


def _memory_trace_entry(text: str, response: dict[str, Any]) -> dict[str, Any]:
    """Trace row for a memory turn (memory tools run in-process, outside the MCP gateway).

    A write is named after the change the trusted text asked for; a read checks prompt cues
    first so a telescope recall is not mislabelled `get_history` when the user has history.
    """
    from astropilote.agentic.conversation import (
        _SITE_WORD,
        _TELESCOPE_WORD,
        _normalized,
        requested_memory_write,
    )

    write = requested_memory_write(text)
    if write is not None:
        tool, arg = _MEMORY_WRITE_TOOLS[(write.op, write.key)]
        args = {arg: write.value} if arg else {}
        trace_status = "ok" if response.get("status") in ("saved", "forgotten") else "error"
        return {"server": "memory", "tool": tool, "args": args, "status": trace_status}

    normalized = _normalized(text)
    if _HISTORY_READ_CUE.search(normalized) and not _TELESCOPE_WORD.search(normalized):
        tool = "get_history"
    elif _SITE_WORD.search(normalized) and not _TELESCOPE_WORD.search(normalized):
        tool = "get_preferred_site"
    elif _TELESCOPE_WORD.search(normalized):
        tool = "get_preferred_telescope"
    elif response.get("history"):
        tool = "get_history"
    elif response.get("preferred_site") and not response.get("preferred_telescope_id"):
        tool = "get_preferred_site"
    else:
        tool = "get_preferred_telescope"
    trace_status = "error" if response.get("status") == "tool_error" else "ok"
    return {"server": "memory", "tool": tool, "args": {}, "status": trace_status}


def _synthesize_offline_plan(
    text: str, response: dict[str, Any], tool_calls: list[dict[str, Any]]
) -> list[str]:
    """Produce concise plan steps for deterministic offline Observatory runs."""
    route = str(response.get("route") or "clarify")
    status = str(response.get("status") or "completed")
    intent = response.get("intent")
    steps = [
        f"Route classified as '{route}'"
        + (f" (intent='{intent}')" if intent else "")
        + f" for: {text.strip()[:90]}"
    ]
    if tool_calls:
        names = ", ".join(f"{c['server']}.{c['tool']}" for c in tool_calls)
        steps.append(f"Executed MCP tool sequence: {names}")
    if response.get("pending_approval"):
        bk = response.get("pending_booking_id") or "reservation"
        steps.append(
            f"Paused at checkpointed LangGraph interrupt() awaiting human operator approval for {bk}."
        )
    else:
        steps.append(f"Finished with status='{status}'.")
    return steps


@app.get("/api/observatory/state")
def api_observatory_state(request: Request, user_id: str = "astro-club") -> dict[str, Any]:
    """Return current SQLite observatory state and user telescope preference (loopback only)."""
    _require_local_client(request)
    from astropilote.agentic.memory import UserMemory

    store = _ensure_observatory_seeded()
    clean_user = user_id.strip() or "astro-club"
    return {
        "status": "ok",
        "user_id": clean_user,
        "seeds": list(_OBSERVATORY_SEEDS),
        **_observatory_snapshot(store, UserMemory(OBSERVATORY_MEMORY_PATH), clean_user),
    }


@app.post("/api/observatory/reset")
def api_observatory_reset(req: ObservatoryResetRequest, request: Request) -> dict[str, Any]:
    """Reset the SQLite observatory database to a canonical seed fixture (loopback only)."""
    _require_local_client(request)
    from astropilote.agentic.memory import UserMemory
    from astropilote.agentic.workflow import discard_checkpointed_threads
    from astropilote.backend.store import ObservatoryStore

    seed_path = SEEDS_DIR / req.seed_name
    if not seed_path.is_file():
        raise HTTPException(status_code=404, detail=f"Seed '{req.seed_name}' not found.")

    store = ObservatoryStore(OBSERVATORY_DB_PATH)
    store.reset_backend(seed_path)
    # Approvals paused before the reset target the previous dataset and must not be resumable.
    discard_checkpointed_threads(OBSERVATORY_CHECKPOINT_PATH, _pending_approvals().drain())
    memory = UserMemory(OBSERVATORY_MEMORY_PATH)
    if req.clear_memory:
        for actor in ("astro-club", "alice", "bob", req.user_id.strip()):
            if actor:
                memory.forget_user(actor)
    clean_user = req.user_id.strip() or "astro-club"
    return {
        "status": "ok",
        "seed": req.seed_name,
        "user_id": clean_user,
        **_observatory_snapshot(store, memory, clean_user),
    }


@app.post("/api/observatory/run")
def api_observatory_run(req: ObservatoryRunRequest, request: Request) -> dict[str, Any]:
    """Run one stateful Observatory + Sky MCP turn with plan, tool trace, and HITL interrupt."""
    _require_local_client(request)
    from astropilote.agentic.conversation import resolve_instrument_reference
    from astropilote.agentic.history import HistoryRecordingGateway
    from astropilote.agentic.llm_agent import LLMObservatoryAgent
    from astropilote.agentic.memory import UserMemory
    from astropilote.agentic.memory_conversation import MemoryConversationalAgent
    from astropilote.agentic.resilience import RetryingGateway
    from astropilote.agentic.scenario_eval import TracedGateway
    from astropilote.agentic.workflow import ObservatoryWorkflow

    ref_date = settings.observatory_reference_date
    if req.reference_date is not None:
        try:
            ref_date = date.fromisoformat(req.reference_date)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid ISO reference_date.") from exc
    store = _ensure_observatory_seeded()
    clean_user = req.user_id.strip()
    effective_text = resolve_instrument_reference(
        req.text, store.get_state(), reference_date=ref_date
    )
    memory = UserMemory(OBSERVATORY_MEMORY_PATH)
    # Per-request retry layer: transient MCP failures are replayed, and its circuit resets
    # with each request. It sits under the recorder, so the UI trace shows the final outcome.
    # Verified booking writes are appended to the actor's long-term history on the way back.
    recorder = _OutcomeRecorder(
        HistoryRecordingGateway(
            RetryingGateway(_get_observatory_gateway()), memory, store.get_state
        )
    )
    gateway = TracedGateway(recorder)
    approval_request: dict[str, Any] | None = None

    if req.llm_mode in ("local", "api"):
        with LLMObservatoryAgent(
            db_path=OBSERVATORY_DB_PATH,
            checkpoint_path=OBSERVATORY_CHECKPOINT_PATH,
            memory=memory,
            gateway=gateway,
            llm_mode=req.llm_mode,
        ) as llm_agent:
            try:
                response = llm_agent.handle(
                    effective_text,
                    user_id=clean_user,
                    reference_date=ref_date,
                    max_steps=_OBSERVATORY_MAX_STEPS,
                )
            except LLMUnavailableError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            if response.get("pending_approval") is True:
                approval_request = _approval_request(llm_agent.graph, response["thread_id"])
        plan_steps = list(response.get("thoughts") or [])
    else:

        def make_workflow() -> ObservatoryWorkflow:
            return ObservatoryWorkflow(
                db_path=OBSERVATORY_DB_PATH,
                checkpoint_path=OBSERVATORY_CHECKPOINT_PATH,
                gateway=gateway,
            )

        def astronomy_notice(_: str) -> dict[str, Any]:
            return {
                "status": "astronomy_redirect",
                "message": (
                    "General deep-sky encyclopedia & ephemeris question detected — "
                    "use the main AstroPilote chat for 567-object RAG & WCS queries."
                ),
            }

        controller = MemoryConversationalAgent(
            db_path=OBSERVATORY_DB_PATH,
            checkpoint_path=OBSERVATORY_CHECKPOINT_PATH,
            memory=memory,
            gateway=gateway,
            workflow_factory=make_workflow,
            astronomy_runner=astronomy_notice,
        )
        response = controller.handle(
            effective_text,
            user_id=clean_user,
            reference_date=ref_date,
            max_steps=_OBSERVATORY_MAX_STEPS,
        )
        if response.get("pending_approval") is True:
            with make_workflow() as workflow:
                approval_request = _approval_request(workflow.graph, response["thread_id"])
        plan_steps = _synthesize_offline_plan(effective_text, response, gateway.calls)

    tool_calls = _with_outcomes(gateway.calls, recorder.outcomes)
    if response.get("route") == "memory":
        tool_calls.append(_memory_trace_entry(effective_text, response))

    thread_id = response.get("thread_id")
    if response.get("pending_approval") is True and isinstance(thread_id, str) and thread_id:
        _pending_approvals().register(
            thread_id=thread_id,
            user_id=clean_user,
            context={"llm_mode": req.llm_mode, "prior_calls": tool_calls, "prior_plan": plan_steps},
            ttl_seconds=_APPROVAL_TTL_S,
        )

    return {
        "status": "ok",
        "llm_mode": req.llm_mode,
        **({"interpreted_text": effective_text} if effective_text != req.text else {}),
        "response": response,
        "plan": plan_steps,
        "max_steps": _OBSERVATORY_MAX_STEPS,
        "approval_request": approval_request,
        "tool_calls": tool_calls,
        **_observatory_snapshot(store, memory, clean_user),
    }


@app.post("/api/observatory/approve")
def api_observatory_approve(req: ObservatoryApproveRequest, request: Request) -> dict[str, Any]:
    """Resume a paused LangGraph cancellation `interrupt()` with explicit human approval/rejection."""
    _require_local_client(request)
    from astropilote.agentic.history import HistoryRecordingGateway
    from astropilote.agentic.llm_agent import LLMObservatoryAgent
    from astropilote.agentic.memory import UserMemory
    from astropilote.agentic.resilience import RetryingGateway
    from astropilote.agentic.scenario_eval import TracedGateway
    from astropilote.agentic.workflow import ObservatoryWorkflow, discard_checkpointed_threads

    store = _ensure_observatory_seeded()
    clean_user = req.user_id.strip()
    claim, pending = _pending_approvals().claim(thread_id=req.thread_id, user_id=clean_user)
    if pending is None:
        if claim == "expired":
            discard_checkpointed_threads(OBSERVATORY_CHECKPOINT_PATH, [req.thread_id])
        status_code, detail = _APPROVAL_CLAIM_ERRORS[claim]
        raise HTTPException(status_code=status_code, detail=detail)

    llm_mode = pending["llm_mode"]
    memory = UserMemory(OBSERVATORY_MEMORY_PATH)
    recorder = _OutcomeRecorder(
        HistoryRecordingGateway(
            RetryingGateway(_get_observatory_gateway()), memory, store.get_state
        )
    )
    gateway = TracedGateway(recorder)
    try:
        if llm_mode in ("local", "api"):
            with LLMObservatoryAgent(
                db_path=OBSERVATORY_DB_PATH,
                checkpoint_path=OBSERVATORY_CHECKPOINT_PATH,
                memory=memory,
                gateway=gateway,
                llm_mode=llm_mode,
            ) as llm_agent:
                resumed = llm_agent.resume(thread_id=req.thread_id, approved=req.approved)
        else:
            with ObservatoryWorkflow(
                db_path=OBSERVATORY_DB_PATH,
                checkpoint_path=OBSERVATORY_CHECKPOINT_PATH,
                gateway=gateway,
            ) as wf:
                resumed = wf.resume(thread_id=req.thread_id, approved=req.approved)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    decision_label = "APPROVED" if req.approved else "REJECTED"
    return {
        "status": "ok",
        "llm_mode": llm_mode,
        "response": resumed,
        "plan": [
            *pending["prior_plan"],
            f"Human operator {decision_label} checkpointed interrupt() on thread {req.thread_id}.",
        ],
        "max_steps": _OBSERVATORY_MAX_STEPS,
        "approval_request": None,
        "tool_calls": [*pending["prior_calls"], *_with_outcomes(gateway.calls, recorder.outcomes)],
        **_observatory_snapshot(store, memory, clean_user),
    }


FRONTEND_DIST_DIR = Path(__file__).resolve().parents[3] / "frontend" / "dist"
if (FRONTEND_DIST_DIR / "assets").exists():
    app.mount(
        "/assets", StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")), name="frontend-assets"
    )
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/favicon.ico", include_in_schema=False)
async def serve_favicon() -> Response:
    """Return 204 No Content for browser favicon requests."""
    return Response(status_code=204)


@app.get("/")
async def serve_index() -> FileResponse:
    """Serve the compiled React SPA when built, falling back to legacy static UI."""
    spa_index = FRONTEND_DIST_DIR / "index.html"
    index_path = spa_index if spa_index.exists() else (STATIC_DIR / "index.html")
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
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
