"""User-isolated Observing Logbook, Rigs, Sites, Messier/Caldwell Progress, and HITL Approval service."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from astropilote.auth.crypto import generate_raw_token, hash_token
from astropilote.db.models import (
    CatalogProgress,
    ChatMessage,
    ChatThread,
    ObservingSession,
    PendingActionApproval,
    Rig,
    SessionTarget,
    Site,
)

_SESSION_STATUSES = frozenset({"planned", "completed", "cancelled"})
_TARGET_STATUSES = frozenset({"planned", "observed", "imaged"})
_CATALOG_STATUSES = frozenset({"wishlist", "observed", "imaged"})
_OBJECT_ID_RE = re.compile(r"^(M|C|NGC|IC)\s*0*(\d{1,4})$", re.IGNORECASE)

MESSIER_TOTAL = 110
CALDWELL_TOTAL = 109


def _as_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def canonicalize_object_id(raw_id: str) -> tuple[str, str]:
    """Normalize a catalog ID (e.g. `m31` -> `('M031', 'Messier')`, `c20` -> `('C020', 'Caldwell')`)."""
    cleaned = raw_id.strip().upper().replace(" ", "")
    m = _OBJECT_ID_RE.match(cleaned)
    if not m:
        return cleaned[:32], "Other"
    prefix = m.group(1).upper()
    num = int(m.group(2))
    if prefix == "M":
        return f"M{num:03d}", "Messier"
    if prefix == "C":
        return f"C{num:03d}", "Caldwell"
    if prefix == "NGC":
        return f"NGC{num:04d}", "NGC"
    return f"IC{num:04d}", "IC"


# ============================================================================
# Rigs
# ============================================================================
def list_rigs(db: Session, *, user_id: str) -> list[dict[str, Any]]:
    rows = db.scalars(
        select(Rig).where(Rig.user_id == user_id).order_by(Rig.is_default.desc(), Rig.created_at)
    ).all()
    return [_rig_to_dict(r) for r in rows]


def save_rig(
    db: Session,
    *,
    user_id: str,
    name: str,
    focal_length_mm: float = 250.0,
    aperture_mm: float | None = None,
    sensor_format: str = "APS-C",
    sensor_width_mm: float = 23.5,
    sensor_height_mm: float = 15.6,
    mount_type: str = "tracker",
    preset_id: str | None = None,
    is_default: bool = False,
    notes: str = "",
    rig_id: str | None = None,
) -> dict[str, Any]:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Rig name cannot be empty.")
    if focal_length_mm <= 0:
        raise ValueError("focal_length_mm must be positive.")

    if is_default:
        for existing in db.scalars(select(Rig).where(Rig.user_id == user_id, Rig.is_default)).all():
            existing.is_default = False

    if rig_id:
        rig = db.scalar(select(Rig).where(Rig.id == rig_id, Rig.user_id == user_id))
        if rig is None:
            raise KeyError(f"Rig `{rig_id}` not found.")
    else:
        rig = Rig(user_id=user_id, name=clean_name)
        db.add(rig)

    rig.name = clean_name[:120]
    rig.preset_id = preset_id
    rig.focal_length_mm = float(focal_length_mm)
    rig.aperture_mm = float(aperture_mm) if aperture_mm is not None else None
    rig.sensor_format = sensor_format.strip()[:48]
    rig.sensor_width_mm = float(sensor_width_mm)
    rig.sensor_height_mm = float(sensor_height_mm)
    rig.mount_type = mount_type.strip()[:32]
    rig.is_default = bool(is_default)
    rig.notes = notes.strip()
    db.flush()
    return _rig_to_dict(rig)


def delete_rig(db: Session, *, user_id: str, rig_id: str) -> bool:
    rig = db.scalar(select(Rig).where(Rig.id == rig_id, Rig.user_id == user_id))
    if rig is None:
        return False
    db.delete(rig)
    db.flush()
    return True


def _rig_to_dict(r: Rig) -> dict[str, Any]:
    return {
        "id": r.id,
        "name": r.name,
        "preset_id": r.preset_id,
        "focal_length_mm": r.focal_length_mm,
        "aperture_mm": r.aperture_mm,
        "sensor_format": r.sensor_format,
        "sensor_width_mm": r.sensor_width_mm,
        "sensor_height_mm": r.sensor_height_mm,
        "mount_type": r.mount_type,
        "is_default": r.is_default,
        "notes": r.notes,
    }


# ============================================================================
# Sites
# ============================================================================
def list_sites(db: Session, *, user_id: str) -> list[dict[str, Any]]:
    rows = db.scalars(
        select(Site)
        .where(Site.user_id == user_id)
        .order_by(Site.is_default.desc(), Site.created_at)
    ).all()
    return [_site_to_dict(s) for s in rows]


def save_site(
    db: Session,
    *,
    user_id: str,
    name: str,
    lat: float,
    lon: float,
    bortle_class: int | None = None,
    elevation_m: float | None = None,
    is_default: bool = False,
    notes: str = "",
    site_id: str | None = None,
) -> dict[str, Any]:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Site name cannot be empty.")
    if not (-90.0 <= float(lat) <= 90.0) or not (-180.0 <= float(lon) <= 180.0):
        raise ValueError("Latitude/longitude out of range.")
    if bortle_class is not None and not (1 <= int(bortle_class) <= 9):
        raise ValueError("bortle_class must be between 1 and 9.")

    if is_default:
        for existing in db.scalars(
            select(Site).where(Site.user_id == user_id, Site.is_default)
        ).all():
            existing.is_default = False

    if site_id:
        site = db.scalar(select(Site).where(Site.id == site_id, Site.user_id == user_id))
        if site is None:
            raise KeyError(f"Site `{site_id}` not found.")
    else:
        site = Site(user_id=user_id, name=clean_name, lat=float(lat), lon=float(lon))
        db.add(site)

    site.name = clean_name[:120]
    site.lat = float(lat)
    site.lon = float(lon)
    site.bortle_class = int(bortle_class) if bortle_class is not None else None
    site.elevation_m = float(elevation_m) if elevation_m is not None else None
    site.is_default = bool(is_default)
    site.notes = notes.strip()
    db.flush()
    return _site_to_dict(site)


def delete_site(db: Session, *, user_id: str, site_id: str) -> bool:
    site = db.scalar(select(Site).where(Site.id == site_id, Site.user_id == user_id))
    if site is None:
        return False
    db.delete(site)
    db.flush()
    return True


def _site_to_dict(s: Site) -> dict[str, Any]:
    return {
        "id": s.id,
        "name": s.name,
        "lat": s.lat,
        "lon": s.lon,
        "bortle_class": s.bortle_class,
        "elevation_m": s.elevation_m,
        "is_default": s.is_default,
        "notes": s.notes,
    }


# ============================================================================
# Observing Sessions & Targets
# ============================================================================
def list_sessions(
    db: Session,
    *,
    user_id: str,
    status: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    stmt = (
        select(ObservingSession)
        .options(selectinload(ObservingSession.targets))
        .where(ObservingSession.user_id == user_id)
    )
    if status:
        stmt = stmt.where(ObservingSession.status == status)
    stmt = stmt.order_by(
        ObservingSession.obs_date.desc(), ObservingSession.created_at.desc()
    ).limit(max(1, min(limit, 200)))
    rows = db.scalars(stmt).all()
    return [_session_to_dict(s) for s in rows]


def get_session(db: Session, *, user_id: str, session_id: str) -> dict[str, Any] | None:
    row = db.scalar(
        select(ObservingSession)
        .options(selectinload(ObservingSession.targets))
        .where(ObservingSession.id == session_id, ObservingSession.user_id == user_id)
    )
    return _session_to_dict(row) if row is not None else None


def create_session(
    db: Session,
    *,
    user_id: str,
    title: str,
    obs_date: str,
    status: str = "planned",
    site_id: str | None = None,
    site_name: str = "",
    lat: float | None = None,
    lon: float | None = None,
    rig_id: str | None = None,
    rig_name: str = "",
    cloud_cover_pct: float | None = None,
    moon_illumination_pct: float | None = None,
    notes: str = "",
    targets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    clean_title = title.strip()
    if not clean_title:
        raise ValueError("Session title cannot be empty.")
    if status not in _SESSION_STATUSES:
        raise ValueError(f"Invalid status `{status}`.")

    if site_id:
        site = db.scalar(select(Site).where(Site.id == site_id, Site.user_id == user_id))
        if site is None:
            raise KeyError(f"Site `{site_id}` not found.")
        site_name = site_name or site.name
        lat = site.lat if lat is None else lat
        lon = site.lon if lon is None else lon

    if rig_id:
        rig = db.scalar(select(Rig).where(Rig.id == rig_id, Rig.user_id == user_id))
        if rig is None:
            raise KeyError(f"Rig `{rig_id}` not found.")
        rig_name = rig_name or rig.name

    sess = ObservingSession(
        user_id=user_id,
        title=clean_title[:160],
        obs_date=obs_date.strip()[:16],
        status=status,
        site_id=site_id,
        site_name=site_name.strip()[:120],
        lat=lat,
        lon=lon,
        rig_id=rig_id,
        rig_name=rig_name.strip()[:120],
        cloud_cover_pct=cloud_cover_pct,
        moon_illumination_pct=moon_illumination_pct,
        notes=notes.strip(),
    )
    db.add(sess)
    db.flush()

    for t in targets or []:
        add_target_to_session(
            db,
            user_id=user_id,
            session_id=sess.id,
            object_id=str(t["object_id"]),
            object_name=str(t.get("object_name") or ""),
            ra_deg=t.get("ra_deg"),
            dec_deg=t.get("dec_deg"),
            exposure_minutes=float(t.get("exposure_minutes") or 0.0),
            sub_exposure_s=t.get("sub_exposure_s"),
            frames_count=int(t.get("frames_count") or 0),
            filter_name=t.get("filter_name"),
            status=str(t.get("status") or ("observed" if status == "completed" else "planned")),
            rating=t.get("rating"),
            notes=str(t.get("notes") or ""),
        )

    db.refresh(sess)
    return _session_to_dict(sess)


def update_session(
    db: Session,
    *,
    user_id: str,
    session_id: str,
    title: str | None = None,
    obs_date: str | None = None,
    status: str | None = None,
    site_name: str | None = None,
    rig_name: str | None = None,
    cloud_cover_pct: float | None = None,
    moon_illumination_pct: float | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    sess = db.scalar(
        select(ObservingSession)
        .options(selectinload(ObservingSession.targets))
        .where(ObservingSession.id == session_id, ObservingSession.user_id == user_id)
    )
    if sess is None:
        raise KeyError(f"Session `{session_id}` not found.")
    if title is not None and title.strip():
        sess.title = title.strip()[:160]
    if obs_date is not None and obs_date.strip():
        sess.obs_date = obs_date.strip()[:16]
    if status is not None:
        if status not in _SESSION_STATUSES:
            raise ValueError(f"Invalid status `{status}`.")
        sess.status = status
    if site_name is not None:
        sess.site_name = site_name.strip()[:120]
    if rig_name is not None:
        sess.rig_name = rig_name.strip()[:120]
    if cloud_cover_pct is not None:
        sess.cloud_cover_pct = float(cloud_cover_pct)
    if moon_illumination_pct is not None:
        sess.moon_illumination_pct = float(moon_illumination_pct)
    if notes is not None:
        sess.notes = notes.strip()
    db.flush()
    return _session_to_dict(sess)


def add_target_to_session(
    db: Session,
    *,
    user_id: str,
    session_id: str,
    object_id: str,
    object_name: str = "",
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    exposure_minutes: float = 0.0,
    sub_exposure_s: float | None = None,
    frames_count: int = 0,
    filter_name: str | None = None,
    status: str = "planned",
    rating: int | None = None,
    notes: str = "",
) -> dict[str, Any]:
    sess = db.scalar(
        select(ObservingSession).where(
            ObservingSession.id == session_id, ObservingSession.user_id == user_id
        )
    )
    if sess is None:
        raise KeyError(f"Session `{session_id}` not found.")
    if status not in _TARGET_STATUSES:
        raise ValueError(f"Invalid target status `{status}`.")

    canon_id, catalog_name = canonicalize_object_id(object_id)
    target = SessionTarget(
        session_id=sess.id,
        object_id=canon_id,
        object_name=object_name.strip()[:120],
        ra_deg=float(ra_deg) if ra_deg is not None else None,
        dec_deg=float(dec_deg) if dec_deg is not None else None,
        exposure_minutes=max(0.0, float(exposure_minutes)),
        sub_exposure_s=float(sub_exposure_s) if sub_exposure_s is not None else None,
        frames_count=max(0, int(frames_count)),
        filter_name=filter_name.strip()[:48] if filter_name else None,
        status=status,
        rating=int(rating) if rating is not None else None,
        notes=notes.strip(),
    )
    db.add(target)
    db.flush()

    if status in ("observed", "imaged"):
        upsert_catalog_progress(
            db,
            user_id=user_id,
            object_id=canon_id,
            catalog=catalog_name,
            status=status,
            observed_date=sess.obs_date,
            add_exposure_minutes=target.exposure_minutes,
            session_id=sess.id,
            notes=notes,
        )
    return _target_to_dict(target)


def delete_session(db: Session, *, user_id: str, session_id: str) -> bool:
    sess = db.scalar(
        select(ObservingSession).where(
            ObservingSession.id == session_id, ObservingSession.user_id == user_id
        )
    )
    if sess is None:
        return False
    db.delete(sess)
    db.flush()
    return True


def _target_to_dict(t: SessionTarget) -> dict[str, Any]:
    return {
        "id": t.id,
        "object_id": t.object_id,
        "object_name": t.object_name,
        "ra_deg": t.ra_deg,
        "dec_deg": t.dec_deg,
        "exposure_minutes": t.exposure_minutes,
        "sub_exposure_s": t.sub_exposure_s,
        "frames_count": t.frames_count,
        "filter_name": t.filter_name,
        "status": t.status,
        "rating": t.rating,
        "notes": t.notes,
    }


def _session_to_dict(s: ObservingSession) -> dict[str, Any]:
    return {
        "id": s.id,
        "title": s.title,
        "obs_date": s.obs_date,
        "status": s.status,
        "site_id": s.site_id,
        "site_name": s.site_name,
        "lat": s.lat,
        "lon": s.lon,
        "rig_id": s.rig_id,
        "rig_name": s.rig_name,
        "cloud_cover_pct": s.cloud_cover_pct,
        "moon_illumination_pct": s.moon_illumination_pct,
        "notes": s.notes,
        "targets": [_target_to_dict(t) for t in s.targets],
    }


# ============================================================================
# Catalog Progress (Messier 110 / Caldwell 109 Tracker)
# ============================================================================
def upsert_catalog_progress(
    db: Session,
    *,
    user_id: str,
    object_id: str,
    status: str = "observed",
    catalog: str | None = None,
    observed_date: str | None = None,
    add_exposure_minutes: float = 0.0,
    session_id: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    if status not in _CATALOG_STATUSES:
        raise ValueError(f"Invalid catalog status `{status}`.")
    canon_id, inferred_cat = canonicalize_object_id(object_id)
    resolved_cat = catalog or inferred_cat

    row = db.scalar(
        select(CatalogProgress).where(
            CatalogProgress.user_id == user_id, CatalogProgress.object_id == canon_id
        )
    )
    rank = {"wishlist": 0, "observed": 1, "imaged": 2}
    if row is None:
        row = CatalogProgress(
            user_id=user_id,
            object_id=canon_id,
            catalog=resolved_cat,
            status=status,
            first_observed_at=observed_date if status != "wishlist" else None,
            total_exposure_minutes=max(0.0, float(add_exposure_minutes)),
            best_session_id=session_id,
            notes=notes.strip(),
        )
        db.add(row)
    else:
        if rank[status] >= rank.get(row.status, 0):
            row.status = status
        if observed_date and not row.first_observed_at and status != "wishlist":
            row.first_observed_at = observed_date
        row.total_exposure_minutes = round(
            float(row.total_exposure_minutes) + max(0.0, float(add_exposure_minutes)), 2
        )
        if session_id:
            row.best_session_id = session_id
        if notes.strip():
            row.notes = notes.strip()
    db.flush()
    return {
        "object_id": row.object_id,
        "catalog": row.catalog,
        "status": row.status,
        "first_observed_at": row.first_observed_at,
        "total_exposure_minutes": row.total_exposure_minutes,
        "best_session_id": row.best_session_id,
        "notes": row.notes,
    }


def get_catalog_progress_summary(db: Session, *, user_id: str) -> dict[str, Any]:
    rows = db.scalars(
        select(CatalogProgress)
        .where(CatalogProgress.user_id == user_id)
        .order_by(CatalogProgress.object_id)
    ).all()
    items = [
        {
            "object_id": r.object_id,
            "catalog": r.catalog,
            "status": r.status,
            "first_observed_at": r.first_observed_at,
            "total_exposure_minutes": r.total_exposure_minutes,
            "best_session_id": r.best_session_id,
            "notes": r.notes,
        }
        for r in rows
    ]
    completed_messier = sum(
        1 for r in rows if r.catalog == "Messier" and r.status in ("observed", "imaged")
    )
    completed_caldwell = sum(
        1 for r in rows if r.catalog == "Caldwell" and r.status in ("observed", "imaged")
    )
    imaged_total = sum(1 for r in rows if r.status == "imaged")
    return {
        "messier": {
            "completed": completed_messier,
            "total": MESSIER_TOTAL,
            "pct": round(completed_messier / MESSIER_TOTAL * 100.0, 1),
        },
        "caldwell": {
            "completed": completed_caldwell,
            "total": CALDWELL_TOTAL,
            "pct": round(completed_caldwell / CALDWELL_TOTAL * 100.0, 1),
        },
        "imaged_total": imaged_total,
        "total_exposure_minutes": round(sum(r.total_exposure_minutes for r in rows), 1),
        "items": items,
    }


# ============================================================================
# Chat Threads & Messages (Server Persistence)
# ============================================================================
def list_chat_threads(db: Session, *, user_id: str, limit: int = 30) -> list[dict[str, Any]]:
    rows = db.scalars(
        select(ChatThread)
        .where(ChatThread.user_id == user_id)
        .order_by(ChatThread.updated_at.desc())
        .limit(max(1, min(limit, 100)))
    ).all()
    return [
        {
            "id": t.id,
            "title": t.title,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "updated_at": t.updated_at.isoformat() if t.updated_at else None,
        }
        for t in rows
    ]


def get_chat_thread(db: Session, *, user_id: str, thread_id: str) -> dict[str, Any] | None:
    thread = db.scalar(
        select(ChatThread)
        .options(selectinload(ChatThread.messages))
        .where(ChatThread.id == thread_id, ChatThread.user_id == user_id)
    )
    if thread is None:
        return None
    return {
        "id": thread.id,
        "title": thread.title,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "metadata": json.loads(m.metadata_json or "{}"),
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in thread.messages
        ],
    }


def append_chat_message(
    db: Session,
    *,
    user_id: str,
    role: str,
    content: str,
    thread_id: str | None = None,
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    thread: ChatThread | None = None
    if thread_id:
        thread = db.scalar(
            select(ChatThread).where(ChatThread.id == thread_id, ChatThread.user_id == user_id)
        )
    if thread is None:
        first_line = next(iter(content.strip().splitlines()), "")
        derived_title = (title or first_line or "New chat")[:120]
        # Never reuse an unowned/client-supplied `thread_id` as primary key on insert
        thread = ChatThread(user_id=user_id, title=derived_title)
        db.add(thread)
        db.flush()

    msg = ChatMessage(
        thread_id=thread.id,
        role=role,
        content=content,
        metadata_json=json.dumps(metadata or {}, ensure_ascii=False),
    )
    db.add(msg)
    thread.updated_at = datetime.now(UTC)
    db.flush()
    return {"thread_id": thread.id, "message_id": msg.id}


# ============================================================================
# Single-Use SHA-256 HITL Approvals (Destructive Logbook Actions)
# ============================================================================
def issue_action_approval(
    db: Session,
    *,
    user_id: str,
    action_type: str,
    target_id: str,
    ttl_minutes: int = 10,
) -> str:
    """Issue a single-use SHA-256 capability token authorizing a destructive action."""
    raw = generate_raw_token()
    db.add(
        PendingActionApproval(
            token_hash=hash_token(raw),
            user_id=user_id,
            action_type=action_type,
            target_id=target_id,
            expires_at=datetime.now(UTC) + timedelta(minutes=ttl_minutes),
            consumed=False,
        )
    )
    db.flush()
    return raw


def consume_action_approval(
    db: Session,
    *,
    raw_token: str,
    user_id: str,
    action_type: str,
    target_id: str,
) -> bool:
    """Atomically validate and consume a HITL capability token bound to `(user_id, action_type, target_id)`."""
    row = db.get(PendingActionApproval, hash_token(raw_token.strip()))
    if row is None or row.consumed:
        return False
    if row.user_id != user_id or row.action_type != action_type or row.target_id != target_id:
        return False
    if _as_utc(row.expires_at) < datetime.now(UTC):
        return False
    row.consumed = True
    db.flush()
    return True
