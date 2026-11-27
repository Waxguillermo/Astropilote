"""Unified in-process tool registry covering Sky/RAG, Optics, Observing Logbook, and Exports.

Replaces the dual `agent/` vs. `agentic/` tool split with a single typed dispatcher scoped by
`user_id` and `database_url`, reusable directly by the unified `LangGraph` agent and optionally
exposed to external MCP clients via `astropilote-mcp`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from astropilote.config import settings
from astropilote.db import db_session, init_db
from astropilote.export import (
    export_logbook_session_ics,
    export_logbook_session_nina_csv,
    export_logbook_session_stellarium_json,
)
from astropilote.logbook import (
    add_target_to_session,
    consume_action_approval,
    create_session,
    delete_session,
    get_catalog_progress_summary,
    get_session,
    list_rigs,
    list_sessions,
    list_sites,
    save_rig,
    save_site,
    update_session,
)
from astropilote.tools.catalog import get_object as get_object_by_id
from astropilote.tools.catalog import search_catalog
from astropilote.tools.sky import night as get_night_info
from astropilote.tools.sky import visibility as get_targets_visibility
from astropilote.tools.world import geocode as geocode_place
from astropilote.tools.world import weather as get_weather_forecast

READ_ONLY_TOOLS: frozenset[str] = frozenset(
    {
        "search_catalog",
        "get_object",
        "geocode",
        "night",
        "weather",
        "visibility",
        "list_rigs",
        "list_sites",
        "list_sessions",
        "get_session",
        "get_catalog_progress",
        "export_session",
    }
)
SAFE_WRITE_TOOLS: frozenset[str] = frozenset(
    {"save_rig", "save_site", "create_session", "update_session", "log_observed_target"}
)
DANGEROUS_WRITE_TOOLS: frozenset[str] = frozenset({"delete_session"})
ALL_UNIFIED_TOOLS: frozenset[str] = READ_ONLY_TOOLS | SAFE_WRITE_TOOLS | DANGEROUS_WRITE_TOOLS


class ToolExecutionError(RuntimeError):
    """Typed tool error carrying a machine-readable `code` and safe `message`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class UnifiedToolRegistry:
    """In-process typed tool dispatcher bound to one authenticated `user_id` and DB."""

    user_id: str
    database_url: str | None = None

    def __post_init__(self) -> None:
        init_db(self.database_url)

    def call(self, tool_name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Dispatch `tool_name` with validated arguments; returns JSON-serializable dict."""
        name = tool_name.strip()
        params = dict(args or {})
        if name not in ALL_UNIFIED_TOOLS:
            raise ToolExecutionError("UNKNOWN_TOOL", f"Tool `{name}` is not registered.")

        # --- Sky & Catalog read-only tools ---
        if name == "search_catalog":
            query = str(params.get("query") or "").strip()
            if not query:
                raise ToolExecutionError("MISSING_ARGUMENTS", "query is required.")
            chunks = search_catalog(
                query=query,
                catalog=params.get("catalog"),
                object_type=params.get("object_type"),
                constellation=params.get("constellation"),
                max_magnitude=params.get("max_magnitude"),
                narrowband_only=bool(params.get("narrowband_only", False)),
                exclude_guides=bool(params.get("exclude_guides", False)),
                top_k=int(params.get("top_k") or 5),
            )
            return {
                "hits": [
                    {
                        "doc_id": ch.doc_id,
                        "title": ch.title,
                        "section": ch.section,
                        "score": ch.score,
                        "snippet": ch.text[:360],
                    }
                    for ch in chunks
                ]
            }

        if name == "get_object":
            obj_id = str(params.get("object_id") or "").strip()
            if not obj_id:
                raise ToolExecutionError("MISSING_ARGUMENTS", "object_id is required.")
            res = get_object_by_id(
                obj_id,
                focal_length_mm=params.get("focal_length_mm"),
                sensor_format=str(params.get("sensor_format") or "APS-C"),
                aperture_mm=params.get("aperture_mm"),
            )
            if res is None:
                raise ToolExecutionError("NOT_FOUND", f"Object `{obj_id}` was not found.")
            return res

        if name == "geocode":
            place = str(params.get("place_name") or "").strip()
            if not place:
                raise ToolExecutionError("MISSING_ARGUMENTS", "place_name is required.")
            return geocode_place(place)

        if name == "night":
            lat, lon = params.get("lat"), params.get("lon")
            if lat is None or lon is None:
                raise ToolExecutionError("MISSING_ARGUMENTS", "lat and lon are required.")
            obs_date = str(params.get("obs_date") or date.today().isoformat())
            return get_night_info(
                float(lat), float(lon), obs_date, days=int(params.get("days") or 1)
            )

        if name == "weather":
            lat, lon = params.get("lat"), params.get("lon")
            if lat is None or lon is None:
                raise ToolExecutionError("MISSING_ARGUMENTS", "lat and lon are required.")
            obs_date = str(params.get("obs_date") or date.today().isoformat())
            return get_weather_forecast(
                float(lat),
                float(lon),
                obs_date,
                days=int(params.get("days") or 1),
                place_name=params.get("place_name"),
            )

        if name == "visibility":
            obj_ids = params.get("object_ids")
            if isinstance(obj_ids, str):
                obj_ids = [obj_ids]
            lat, lon = params.get("lat"), params.get("lon")
            if not obj_ids or lat is None or lon is None:
                raise ToolExecutionError(
                    "MISSING_ARGUMENTS", "object_ids, lat, and lon are required."
                )
            obs_date = str(params.get("obs_date") or date.today().isoformat())
            return get_targets_visibility(
                list(obj_ids),
                float(lat),
                float(lon),
                obs_date,
                focal_length_mm=params.get("focal_length_mm"),
            )

        # --- Observing Logbook tools (strictly scoped to `self.user_id`) ---
        with db_session(self.database_url) as db:
            if name == "list_rigs":
                return {"rigs": list_rigs(db, user_id=self.user_id)}
            if name == "save_rig":
                rig = save_rig(
                    db,
                    user_id=self.user_id,
                    name=str(params.get("name") or ""),
                    focal_length_mm=float(params.get("focal_length_mm") or 250.0),
                    aperture_mm=params.get("aperture_mm"),
                    sensor_format=str(params.get("sensor_format") or "APS-C"),
                    sensor_width_mm=float(params.get("sensor_width_mm") or 23.5),
                    sensor_height_mm=float(params.get("sensor_height_mm") or 15.6),
                    mount_type=str(params.get("mount_type") or "tracker"),
                    is_default=bool(params.get("is_default", False)),
                    notes=str(params.get("notes") or ""),
                    rig_id=params.get("rig_id"),
                )
                return {"rig": rig}
            if name == "list_sites":
                return {"sites": list_sites(db, user_id=self.user_id)}
            if name == "save_site":
                site = save_site(
                    db,
                    user_id=self.user_id,
                    name=str(params.get("name") or ""),
                    lat=float(params["lat"]),
                    lon=float(params["lon"]),
                    bortle_class=params.get("bortle_class"),
                    elevation_m=params.get("elevation_m"),
                    is_default=bool(params.get("is_default", False)),
                    notes=str(params.get("notes") or ""),
                    site_id=params.get("site_id"),
                )
                return {"site": site}
            if name == "list_sessions":
                return {
                    "sessions": list_sessions(
                        db,
                        user_id=self.user_id,
                        status=params.get("status"),
                        limit=int(params.get("limit") or 25),
                    )
                }
            if name == "get_session":
                sid = str(params.get("session_id") or "").strip()
                sess = get_session(db, user_id=self.user_id, session_id=sid)
                if sess is None:
                    raise ToolExecutionError("NOT_FOUND", f"Session `{sid}` not found.")
                return {"session": sess}
            if name == "create_session":
                sess = create_session(
                    db,
                    user_id=self.user_id,
                    title=str(params.get("title") or "Observing Session"),
                    obs_date=str(params.get("obs_date") or date.today().isoformat()),
                    status=str(params.get("status") or "planned"),
                    site_id=params.get("site_id"),
                    site_name=str(params.get("site_name") or ""),
                    lat=params.get("lat"),
                    lon=params.get("lon"),
                    rig_id=params.get("rig_id"),
                    rig_name=str(params.get("rig_name") or ""),
                    cloud_cover_pct=params.get("cloud_cover_pct"),
                    moon_illumination_pct=params.get("moon_illumination_pct"),
                    notes=str(params.get("notes") or ""),
                    targets=params.get("targets"),
                )
                return {"session": sess}
            if name == "update_session":
                sid = str(params.get("session_id") or "").strip()
                sess = update_session(
                    db,
                    user_id=self.user_id,
                    session_id=sid,
                    title=params.get("title"),
                    obs_date=params.get("obs_date"),
                    status=params.get("status"),
                    site_name=params.get("site_name"),
                    rig_name=params.get("rig_name"),
                    cloud_cover_pct=params.get("cloud_cover_pct"),
                    moon_illumination_pct=params.get("moon_illumination_pct"),
                    notes=params.get("notes"),
                )
                return {"session": sess}
            if name == "log_observed_target":
                sid = str(params.get("session_id") or "").strip()
                target = add_target_to_session(
                    db,
                    user_id=self.user_id,
                    session_id=sid,
                    object_id=str(params.get("object_id") or ""),
                    object_name=str(params.get("object_name") or ""),
                    ra_deg=params.get("ra_deg"),
                    dec_deg=params.get("dec_deg"),
                    exposure_minutes=float(params.get("exposure_minutes") or 0.0),
                    sub_exposure_s=params.get("sub_exposure_s"),
                    frames_count=int(params.get("frames_count") or 0),
                    filter_name=params.get("filter_name"),
                    status=str(params.get("status") or "observed"),
                    rating=params.get("rating"),
                    notes=str(params.get("notes") or ""),
                )
                return {"target": target}
            if name == "get_catalog_progress":
                return get_catalog_progress_summary(db, user_id=self.user_id)
            if name == "export_session":
                sid = str(params.get("session_id") or "").strip()
                fmt = str(params.get("format") or "ics").strip().lower()
                sess = get_session(db, user_id=self.user_id, session_id=sid)
                if sess is None:
                    raise ToolExecutionError("NOT_FOUND", f"Session `{sid}` not found.")
                if fmt == "ics":
                    return {"format": "ics", "content": export_logbook_session_ics(sess)}
                if fmt == "nina_csv":
                    return {"format": "nina_csv", "content": export_logbook_session_nina_csv(sess)}
                if fmt == "stellarium":
                    return {
                        "format": "stellarium",
                        "content": export_logbook_session_stellarium_json(sess),
                    }
                raise ToolExecutionError(
                    "INVALID_FORMAT", "format must be 'ics', 'nina_csv', or 'stellarium'."
                )
            if name == "delete_session":
                sid = str(params.get("session_id") or "").strip()
                token = str(params.get("approval_token") or "").strip()
                if not sid:
                    raise ToolExecutionError("MISSING_ARGUMENTS", "session_id is required.")
                if not token or not consume_action_approval(
                    db,
                    raw_token=token,
                    user_id=self.user_id,
                    action_type="delete_session",
                    target_id=sid,
                ):
                    raise ToolExecutionError(
                        "APPROVAL_REQUIRED",
                        f"Deleting session `{sid}` requires explicit human approval.",
                    )
                deleted = delete_session(db, user_id=self.user_id, session_id=sid)
                if not deleted:
                    raise ToolExecutionError("NOT_FOUND", f"Session `{sid}` not found.")
                return {"deleted": True, "session_id": sid}

        raise ToolExecutionError("UNKNOWN_TOOL", f"Unhandled tool `{name}`.")


def default_user_registry(user_id: str) -> UnifiedToolRegistry:
    """Build a `UnifiedToolRegistry` pointing at the configured application database."""
    return UnifiedToolRegistry(user_id=user_id, database_url=settings.resolved_database_url())
