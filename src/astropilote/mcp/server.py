"""Optional standalone `stdio` MCP server (`astropilote-mcp`) exposing the unified tool registry.

Allows external MCP hosts (Claude Desktop, Cursor, Antigravity) to query Astropilote's sky
ephemeris, ONNX/BM25 catalog RAG, and personal Observing Logbook directly over stdio.

Usage:
    uv run astropilote-mcp
"""

from __future__ import annotations

import json
import os
from typing import Any

from mcp.server.mcpserver import MCPServer

from astropilote.db import LOCAL_USER_ID, db_session, ensure_local_user, init_db
from astropilote.tools.registry import ToolExecutionError, UnifiedToolRegistry


def build_unified_mcp_server(
    *,
    user_id: str | None = None,
    database_url: str | None = None,
) -> MCPServer:
    """Build the MCP 2.x server backed by `UnifiedToolRegistry`."""
    resolved_user = (user_id or os.environ.get("ASTROPILOTE_MCP_USER_ID") or LOCAL_USER_ID).strip()
    init_db(database_url)
    with db_session(database_url) as db:
        if resolved_user == LOCAL_USER_ID:
            ensure_local_user(db)
    registry = UnifiedToolRegistry(user_id=resolved_user, database_url=database_url)

    mcp = MCPServer("astropilote")

    def _run(tool_name: str, args: dict[str, Any]) -> str:
        try:
            return json.dumps(registry.call(tool_name, args), ensure_ascii=False)
        except ToolExecutionError as exc:
            return json.dumps({"error": {"code": exc.code, "message": exc.message}})

    @mcp.tool()
    def search_catalog(query: str, top_k: int = 5) -> str:
        """Search the 567-document deep-sky catalog and observing guides."""
        return _run("search_catalog", {"query": query, "top_k": top_k})

    @mcp.tool()
    def get_object(object_id: str, focal_length_mm: float | None = None) -> str:
        """Fetch exact OpenNGC astrometry and optional optical framing for a deep-sky object."""
        return _run("get_object", {"object_id": object_id, "focal_length_mm": focal_length_mm})

    @mcp.tool()
    def geocode(place_name: str) -> str:
        """Resolve a city or observing site name to latitude and longitude."""
        return _run("geocode", {"place_name": place_name})

    @mcp.tool()
    def night(lat: float, lon: float, obs_date: str, days: int = 1) -> str:
        """Compute astronomical twilight, darkness hours, and Moon illumination."""
        return _run("night", {"lat": lat, "lon": lon, "obs_date": obs_date, "days": days})

    @mcp.tool()
    def weather(lat: float, lon: float, obs_date: str, days: int = 1, place_name: str = "") -> str:
        """Fetch live Open-Meteo nighttime cloud cover, wind, and humidity forecast."""
        return _run(
            "weather",
            {
                "lat": lat,
                "lon": lon,
                "obs_date": obs_date,
                "days": days,
                "place_name": place_name or None,
            },
        )

    @mcp.tool()
    def visibility(object_ids: list[str], lat: float, lon: float, obs_date: str) -> str:
        """Compute hourly altitude curves (>= 20 deg observability check) for deep-sky targets."""
        return _run(
            "visibility",
            {"object_ids": object_ids, "lat": lat, "lon": lon, "obs_date": obs_date},
        )

    @mcp.tool()
    def list_rigs() -> str:
        """List saved telescope/camera rigs in the user's Observing Logbook."""
        return _run("list_rigs", {})

    @mcp.tool()
    def list_sites() -> str:
        """List saved observing sites in the user's Observing Logbook."""
        return _run("list_sites", {})

    @mcp.tool()
    def list_sessions(status: str | None = None, limit: int = 20) -> str:
        """List planned or completed observing sessions."""
        return _run("list_sessions", {"status": status, "limit": limit})

    @mcp.tool()
    def create_session(
        title: str,
        obs_date: str,
        site_name: str = "",
        rig_name: str = "",
        notes: str = "",
    ) -> str:
        """Create a new observing session in the user's Observing Logbook."""
        return _run(
            "create_session",
            {
                "title": title,
                "obs_date": obs_date,
                "site_name": site_name,
                "rig_name": rig_name,
                "notes": notes,
            },
        )

    @mcp.tool()
    def log_observed_target(
        session_id: str,
        object_id: str,
        exposure_minutes: float = 0.0,
        status: str = "observed",
        notes: str = "",
    ) -> str:
        """Log an observed or imaged deep-sky target and update Messier/Caldwell progress."""
        return _run(
            "log_observed_target",
            {
                "session_id": session_id,
                "object_id": object_id,
                "exposure_minutes": exposure_minutes,
                "status": status,
                "notes": notes,
            },
        )

    @mcp.tool()
    def get_catalog_progress() -> str:
        """Return Messier (110) and Caldwell (109) completion stats for the current user."""
        return _run("get_catalog_progress", {})

    return mcp


def main() -> None:
    """Run the optional Astropilote stdio MCP server."""
    build_unified_mcp_server().run()


if __name__ == "__main__":
    main()
