"""Read-only MCP server wrapping AstroPilote's six existing astronomy tools.

Run from project root: uv run python -m astropilote.mcp.sky_server
Uses MCP SDK 2.x, stdio transport. Do not print to stdout from an MCP server.
"""

from __future__ import annotations

from typing import Annotated

from mcp.server import MCPServer
from mcp.types import CallToolResult
from pydantic import Field

from astropilote.mcp.common import invoke
from astropilote.tools import catalog as catalog_tools
from astropilote.tools import sky as sky_tools
from astropilote.tools import world as world_tools

Lat = Annotated[float, Field(ge=-90, le=90, description="Latitude in degrees")]
Lon = Annotated[float, Field(ge=-180, le=180, description="Longitude in degrees")]
DateISO = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="Date YYYY-MM-DD")]

mcp = MCPServer("astropilote-sky")


@mcp.tool()
def search_catalog(
    query: Annotated[str, Field(min_length=1)],
    catalog: str | None = None,
    object_type: str | None = None,
    constellation: str | None = None,
    max_magnitude: float | None = None,
    narrowband_only: bool = False,
    top_k: Annotated[int, Field(ge=1, le=20)] = 5,
    exclude_guides: bool = False,
) -> CallToolResult:
    """Search existing astronomy corpus (hybrid BM25/vector + reranking); READ ONLY."""
    return invoke(
        lambda: catalog_tools.search_catalog(
            query=query,
            catalog=catalog,
            object_type=object_type,
            constellation=constellation,
            max_magnitude=max_magnitude,
            narrowband_only=narrowband_only,
            top_k=top_k,
            exclude_guides=exclude_guides,
        )
    )


@mcp.tool()
def get_object(
    object_id: Annotated[str, Field(min_length=1)],
    focal_length_mm: Annotated[float, Field(gt=0)] | None = None,
    sensor_format: str = "APS-C",
    aperture_mm: Annotated[float, Field(gt=0)] | None = None,
    mount_type: str = "equatorial_goto",
    has_autoguiding: bool = False,
    filters: list[str] | None = None,
    pixel_size_um: Annotated[float, Field(gt=0)] | None = None,
) -> CallToolResult:
    """Read catalog object metadata and optical framing advice; READ ONLY."""
    return invoke(
        lambda: catalog_tools.get_object(
            object_id=object_id,
            focal_length_mm=focal_length_mm,
            sensor_format=sensor_format,
            aperture_mm=aperture_mm,
            mount_type=mount_type,
            has_autoguiding=has_autoguiding,
            filters=filters,
            pixel_size_um=pixel_size_um,
        )
    )


@mcp.tool()
def geocode(place_name: Annotated[str, Field(min_length=1)]) -> CallToolResult:
    """Resolve a location to geographic coordinates; READ ONLY."""
    return invoke(lambda: world_tools.geocode(place_name=place_name))


@mcp.tool()
def night(
    lat: Lat,
    lon: Lon,
    obs_date: DateISO,
    days: Annotated[int, Field(ge=1, le=14)] = 1,
) -> CallToolResult:
    """Calculate twilight, dark time and Moon over 1–14 nights; READ ONLY."""
    return invoke(lambda: sky_tools.night(lat=lat, lon=lon, obs_date=obs_date, days=days))


@mcp.tool()
def weather(
    lat: Lat,
    lon: Lon,
    obs_date: DateISO,
    days: Annotated[int, Field(ge=1, le=14)] = 1,
    place_name: str | None = None,
) -> CallToolResult:
    """Get atmospheric conditions (live or offline fixtures); READ ONLY."""
    return invoke(
        lambda: world_tools.weather(
            lat=lat, lon=lon, obs_date=obs_date, days=days, place_name=place_name
        )
    )


@mcp.tool()
def visibility(
    object_ids: Annotated[list[str], Field(min_length=1, max_length=12)],
    lat: Lat,
    lon: Lon,
    obs_date: DateISO,
    focal_length_mm: Annotated[float, Field(gt=0)] | None = None,
) -> CallToolResult:
    """Compute object visibility and observing windows; READ ONLY."""
    return invoke(
        lambda: sky_tools.visibility(
            object_ids=object_ids,
            lat=lat,
            lon=lon,
            obs_date=obs_date,
            focal_length_mm=focal_length_mm,
        )
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
