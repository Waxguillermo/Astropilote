"""Modular FastAPI router for the Observing Logbook, Rigs, Sites, Messier/Caldwell Progress, and Exports."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from astropilote.auth import enforce_rate_limit, get_current_user
from astropilote.db import User, get_db
from astropilote.export import (
    export_logbook_session_ics,
    export_logbook_session_nina_csv,
    export_logbook_session_stellarium_json,
    safe_download_filename,
)
from astropilote.logbook import (
    add_target_to_session,
    create_session,
    delete_rig,
    delete_session,
    delete_site,
    get_catalog_progress_summary,
    get_chat_thread,
    get_session,
    list_chat_threads,
    list_rigs,
    list_sessions,
    list_sites,
    save_rig,
    save_site,
    update_session,
    upsert_catalog_progress,
)

router = APIRouter(prefix="/api/logbook", tags=["logbook"])


class RigPayload(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    focal_length_mm: float = Field(default=250.0, gt=0)
    aperture_mm: float | None = Field(default=None, gt=0)
    sensor_format: str = Field(default="APS-C", max_length=48)
    sensor_width_mm: float = Field(default=23.5, gt=0)
    sensor_height_mm: float = Field(default=15.6, gt=0)
    mount_type: str = Field(default="tracker", max_length=32)
    preset_id: str | None = Field(default=None, max_length=64)
    is_default: bool = False
    notes: str = ""
    rig_id: str | None = None


class SitePayload(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    bortle_class: int | None = Field(default=None, ge=1, le=9)
    elevation_m: float | None = None
    is_default: bool = False
    notes: str = ""
    site_id: str | None = None


class TargetPayload(BaseModel):
    object_id: str = Field(..., min_length=1, max_length=32)
    object_name: str = ""
    ra_deg: float | None = None
    dec_deg: float | None = None
    exposure_minutes: float = Field(default=0.0, ge=0.0)
    sub_exposure_s: float | None = None
    frames_count: int = Field(default=0, ge=0)
    filter_name: str | None = None
    status: Literal["planned", "observed", "imaged"] = "observed"
    rating: int | None = Field(default=None, ge=1, le=5)
    notes: str = ""


class SessionCreatePayload(BaseModel):
    title: str = Field(..., min_length=1, max_length=160)
    obs_date: str = Field(default_factory=lambda: date.today().isoformat(), max_length=16)
    status: Literal["planned", "completed", "cancelled"] = "planned"
    site_id: str | None = None
    site_name: str = ""
    lat: float | None = None
    lon: float | None = None
    rig_id: str | None = None
    rig_name: str = ""
    cloud_cover_pct: float | None = None
    moon_illumination_pct: float | None = None
    notes: str = ""
    targets: list[TargetPayload] = Field(default_factory=list)


class SessionUpdatePayload(BaseModel):
    title: str | None = None
    obs_date: str | None = None
    status: Literal["planned", "completed", "cancelled"] | None = None
    site_name: str | None = None
    rig_name: str | None = None
    cloud_cover_pct: float | None = None
    moon_illumination_pct: float | None = None
    notes: str | None = None


class CatalogProgressPayload(BaseModel):
    object_id: str = Field(..., min_length=1, max_length=32)
    status: Literal["wishlist", "observed", "imaged"] = "observed"
    observed_date: str | None = None
    add_exposure_minutes: float = Field(default=0.0, ge=0.0)
    notes: str = ""


class VisionLogPayload(BaseModel):
    title: str = Field(default="Sky Vision Capture", min_length=1, max_length=160)
    obs_date: str = Field(default_factory=lambda: date.today().isoformat(), max_length=16)
    site_name: str = ""
    rig_name: str = ""
    exposure_minutes: float = Field(default=0.0, ge=0.0)
    targets: list[TargetPayload] = Field(..., min_length=1, max_length=50)


@router.get("/rigs")
def api_list_rigs(
    request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> dict[str, Any]:
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    return {"rigs": list_rigs(db, user_id=user.id)}


@router.post("/rigs")
def api_save_rig(
    req: RigPayload,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    try:
        rig = save_rig(db, user_id=user.id, **req.model_dump())
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"rig": rig}


@router.delete("/rigs/{rig_id}")
def api_delete_rig(
    rig_id: str,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    if not delete_rig(db, user_id=user.id, rig_id=rig_id):
        raise HTTPException(status_code=404, detail="Rig not found.")
    return {"deleted": True, "rig_id": rig_id}


@router.get("/sites")
def api_list_sites(
    request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> dict[str, Any]:
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    return {"sites": list_sites(db, user_id=user.id)}


@router.post("/sites")
def api_save_site(
    req: SitePayload,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    try:
        site = save_site(db, user_id=user.id, **req.model_dump())
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"site": site}


@router.delete("/sites/{site_id}")
def api_delete_site(
    site_id: str,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    if not delete_site(db, user_id=user.id, site_id=site_id):
        raise HTTPException(status_code=404, detail="Site not found.")
    return {"deleted": True, "site_id": site_id}


@router.get("/sessions")
def api_list_sessions(
    request: Request,
    status: str | None = None,
    limit: int = 50,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    return {"sessions": list_sessions(db, user_id=user.id, status=status, limit=limit)}


@router.post("/sessions")
def api_create_session(
    req: SessionCreatePayload,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    try:
        sess = create_session(
            db,
            user_id=user.id,
            **{**req.model_dump(), "targets": [t.model_dump() for t in req.targets]},
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"session": sess}


@router.get("/sessions/{session_id}")
def api_get_session(
    session_id: str,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    sess = get_session(db, user_id=user.id, session_id=session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session": sess}


@router.patch("/sessions/{session_id}")
def api_update_session(
    session_id: str,
    req: SessionUpdatePayload,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    try:
        sess = update_session(db, user_id=user.id, session_id=session_id, **req.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"session": sess}


@router.delete("/sessions/{session_id}")
def api_delete_session(
    session_id: str,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    if not delete_session(db, user_id=user.id, session_id=session_id):
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"deleted": True, "session_id": session_id}


@router.post("/sessions/{session_id}/targets")
def api_add_target(
    session_id: str,
    req: TargetPayload,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    try:
        target = add_target_to_session(
            db, user_id=user.id, session_id=session_id, **req.model_dump()
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"target": target}


@router.get("/sessions/{session_id}/export/{fmt}")
def api_export_logbook_session(
    session_id: str,
    fmt: Literal["ics", "nina_csv", "stellarium"],
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    sess = get_session(db, user_id=user.id, session_id=session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    base_name = f"{sess['obs_date']}_{sess['title']}"
    if fmt == "ics":
        body = export_logbook_session_ics(sess)
        fname = safe_download_filename(base_name, "ics")
        media = "text/calendar; charset=utf-8"
    elif fmt == "nina_csv":
        body = export_logbook_session_nina_csv(sess)
        fname = safe_download_filename(base_name, "csv")
        media = "text/csv; charset=utf-8"
    else:
        body = export_logbook_session_stellarium_json(sess)
        fname = safe_download_filename(base_name, "json")
        media = "application/json; charset=utf-8"
    return Response(
        content=body,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.get("/progress")
def api_get_catalog_progress(
    request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> dict[str, Any]:
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    return get_catalog_progress_summary(db, user_id=user.id)


@router.post("/progress")
def api_upsert_catalog_progress(
    req: CatalogProgressPayload,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    item = upsert_catalog_progress(
        db,
        user_id=user.id,
        object_id=req.object_id,
        status=req.status,
        observed_date=req.observed_date or date.today().isoformat(),
        add_exposure_minutes=req.add_exposure_minutes,
        notes=req.notes,
    )
    return {"item": item, "summary": get_catalog_progress_summary(db, user_id=user.id)}


@router.post("/vision-log")
def api_log_vision_targets(
    req: VisionLogPayload,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """One-click bridge logging Sky Vision identified targets into a completed session + catalog tracker."""
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    targets = [
        {
            **t.model_dump(),
            "status": "imaged",
            "exposure_minutes": t.exposure_minutes or req.exposure_minutes,
        }
        for t in req.targets
    ]
    sess = create_session(
        db,
        user_id=user.id,
        title=req.title,
        obs_date=req.obs_date,
        status="completed",
        site_name=req.site_name,
        rig_name=req.rig_name,
        targets=targets,
    )
    return {"session": sess, "progress": get_catalog_progress_summary(db, user_id=user.id)}


@router.get("/threads")
def api_list_threads(
    request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> dict[str, Any]:
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    return {"threads": list_chat_threads(db, user_id=user.id)}


@router.get("/threads/{thread_id}")
def api_get_thread(
    thread_id: str,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    thread = get_chat_thread(db, user_id=user.id, thread_id=thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found.")
    return {"thread": thread}
