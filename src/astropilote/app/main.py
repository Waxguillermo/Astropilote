"""FastAPI backend for AstroPilote (Anthropic API mode + Local Ollama mode).

Usage:
    uv run python -m astropilote.app.main
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from astropilote.agent.graph import run_agent
from astropilote.llm import check_llm_status, get_torch_device
from astropilote.tools.catalog import get_object, get_retriever, resolve_object_record
from astropilote.tools.sky import compute_night_ephemeris, visibility
from astropilote.tools.world import geocode, weather

STATIC_DIR = Path(__file__).parent / "static"

PRESET_SCENARIOS: list[dict[str, str]] = [
    {
        "id": "fontainebleau_session",
        "label": "Fontainebleau Night Plan",
        "subtitle": "Oct 17, 2026 · 80mm Refractor",
        "query": (
            "Saturday night, October 17, 2026, I will be in Fontainebleau with an 80 mm refractor "
            "(480 mm focal length). What targets should I observe or image?"
        ),
    },
    {
        "id": "chamonix_narrowband",
        "label": "Chamonix Narrowband Nebulae",
        "subtitle": "Aug 15, 2026 · Ha/OIII Filter",
        "query": (
            "I am in Chamonix on August 15, 2026 with a wide-field astrophotography rig (250 mm focal length) "
            "and a dual-band Ha/OIII filter. Which emission nebulae or supernova remnants in Cygnus can I shoot tonight?"
        ),
    },
    {
        "id": "m31_satellites",
        "label": "Andromeda Satellite Galaxies",
        "subtitle": "Factual Catalog Q&A (M31 / M32 / M110)",
        "query": (
            "What are the two brightest satellite galaxies of the Andromeda Galaxy (M31) that can be seen "
            "in the same wide field of view, and what are their visual magnitudes?"
        ),
    },
    {
        "id": "southern_trap_c080",
        "label": "Omega Centauri from Paris",
        "subtitle": "Horizon & Ephemeris Check (C080)",
        "query": "Can I observe Omega Centauri (Caldwell 80) from Paris on May 15, 2026 at midnight?",
    },
    {
        "id": "out_of_corpus_ic1101",
        "label": "Out-of-Corpus Guard (IC 1101)",
        "subtitle": "Tests zero-hallucination abstention",
        "query": (
            "What can you observe of the supergiant elliptical galaxy IC 1101 in Abell 2029 with an amateur telescope?"
        ),
    },
    {
        "id": "impossible_forecast_2028",
        "label": "2028 Weather Forecast Guard",
        "subtitle": "Refuses >14-day weather prediction",
        "query": "Will the sky be completely cloud-free in Fontainebleau on March 3, 2028 at 23:00?",
    },
]


class ChatRequest(BaseModel):
    """Request body for `/api/chat`."""

    question: str = Field(..., min_length=2)
    llm_mode: Literal["api", "local"] | None = Field(default=None)
    date_override: str | None = Field(default=None)
    focal_length_mm: float | None = Field(default=480.0)
    aperture_mm: float | None = Field(default=80.0)


class EphemerisPreviewRequest(BaseModel):
    """Request body for `/api/ephemeris`."""

    place_name: str = "Fontainebleau"
    obs_date: str = "2026-10-17"
    focal_length_mm: float = 480.0
    aperture_mm: float = 80.0
    object_ids: list[str] = Field(
        default_factory=lambda: ["M031", "M013", "M027", "C014", "M045"]
    )


app = FastAPI(
    title="AstroPilote API",
    description="Open-source Hybrid RAG & Astronomical Session Copilot.",
    version="1.0.0",
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
    retriever = get_retriever()
    retriever._ensure_bm25_loaded()
    llm_info = check_llm_status()
    return {
        "status": "ok",
        "llm": llm_info,
        "accelerator_device": get_torch_device(),
        "corpus_documents": 506,
        "indexed_chunks": len(retriever._chunks),
        "has_dense_qdrant": retriever.has_dense_index(),
    }


@app.get("/api/presets")
def api_presets() -> dict[str, Any]:
    """Return curated preset questions."""
    return {"presets": PRESET_SCENARIOS}


@app.get("/api/catalog/{object_id}")
def api_get_catalog_object(object_id: str, focal_length_mm: float = 480.0) -> dict[str, Any]:
    """Return full document record, metadata, and FOV framing for a specific catalog ID."""
    rec = resolve_object_record(object_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Object '{object_id}' not found in corpus.")
    obj_info = get_object(object_id=object_id, focal_length_mm=focal_length_mm)
    return {
        **obj_info,
        "full_text": rec.get("text", ""),
    }


@app.post("/api/ephemeris")
def api_ephemeris_preview(req: EphemerisPreviewRequest) -> dict[str, Any]:
    """Compute night twilight window, weather, and target altitude curves."""
    geo = geocode(place_name=req.place_name)
    lat = float(geo.get("lat", 48.4047))
    lon = float(geo.get("lon", 2.7016))

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
    answer, state, eval_row = run_agent(
        req.question,
        question_id="web_session",
        llm_mode=req.llm_mode,
    )

    vis_objects: dict[str, Any] = {}
    night_payload: dict[str, Any] = {}
    weather_payload: dict[str, Any] = {}
    for msg in state.get("messages", []):
        if getattr(msg, "role", "") == "tool":
            try:
                data = json.loads(msg.content)
                if msg.name == "visibility":
                    vis_objects = data.get("objects", {})
                elif msg.name == "night":
                    night_payload = data
                elif msg.name == "weather":
                    weather_payload = data
            except Exception:
                pass

    intent = state.get("intent")
    retrieved_chunks = [c.model_dump(mode="json") for c in state.get("retrieved", [])]

    return {
        "answer": answer.model_dump(mode="json"),
        "telemetry": eval_row.model_dump(mode="json"),
        "intent": intent.model_dump(mode="json") if intent else None,
        "night": night_payload,
        "weather": weather_payload,
        "visibility_objects": vis_objects,
        "retrieved_chunks": retrieved_chunks,
    }


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index() -> FileResponse:
    """Serve the clean AstroPilote Web UI."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Static frontend index.html not found.")
    return FileResponse(str(index_path))


def main() -> None:
    """Launch the AstroPilote server on http://127.0.0.1:8000."""
    print("🔭 AstroPilote running at http://127.0.0.1:8000")
    uvicorn.run("astropilote.app.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
