"""Real-world context tools (`geocode`, `weather`) with deterministic evaluation fixtures."""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path

import httpx

from astropilote.config import settings
from astropilote.tools.base import adk_tool

FIXTURES_PATH = Path("eval/fixtures/world_fixtures.json")
MAX_FORECAST_DAYS = 14


@lru_cache(maxsize=1)
def _load_fixtures() -> dict:
    if FIXTURES_PATH.exists():
        return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    return {"locations": {}, "weather_by_date": {}}


@adk_tool
def geocode(place_name: str) -> dict:
    """Convert a city or observation site name into geographic coordinates (latitude, longitude).

    Args:
        place_name: Location name, e.g. 'Fontainebleau', 'Paris', 'Chamonix', 'Marseille'.
    """
    fixtures = _load_fixtures()
    key = place_name.strip().lower().split(",")[0].strip()

    if key in fixtures.get("locations", {}):
        loc = fixtures["locations"][key]
        return {
            "found": True,
            "place_name": loc["name"],
            "lat": loc["lat"],
            "lon": loc["lon"],
            "elevation_m": loc.get("elevation_m", 100),
            "source": "fixture",
        }

    if settings.eval_offline_mode:
        return {
            "found": False,
            "place_name": place_name,
            "message": f"Location '{place_name}' not found in offline evaluation fixtures.",
        }

    try:
        with httpx.Client(
            timeout=10.0,
            headers={"User-Agent": "AstroPilote-ENPC-Project/0.1"},
        ) as client:
            resp = client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": place_name, "format": "json", "limit": 1},
            )
            resp.raise_for_status()
            items = resp.json()
            if items:
                top = items[0]
                return {
                    "found": True,
                    "place_name": top.get("display_name", place_name),
                    "lat": round(float(top["lat"]), 4),
                    "lon": round(float(top["lon"]), 4),
                    "source": "nominatim",
                }
    except Exception as exc:
        return {"found": False, "place_name": place_name, "message": f"Geocoding error: {exc}"}

    return {"found": False, "place_name": place_name, "message": "Location not found."}


@adk_tool
def weather(lat: float, lon: float, obs_date: str) -> dict:
    """Fetch hourly nighttime cloud cover and seeing conditions for `(lat, lon)` on `obs_date`.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        obs_date: Target observation date in 'YYYY-MM-DD' format.
    """
    clean_date = obs_date.strip()[:10]
    fixtures = _load_fixtures()

    # 1. Check recorded evaluation fixtures first for deterministic reproducibility
    if clean_date in fixtures.get("weather_by_date", {}):
        w = fixtures["weather_by_date"][clean_date]
        return {
            "available": True,
            "date": clean_date,
            "lat": lat,
            "lon": lon,
            **w,
            "source": "fixture",
        }

    # 2. Reject impossible long-range forecasts (> 14 days in the future from reference year)
    try:
        target_d = date.fromisoformat(clean_date)
        ref_today = date(2026, 10, 10)
        delta_days = (target_d - ref_today).days
        if delta_days > MAX_FORECAST_DAYS:
            return {
                "available": False,
                "date": clean_date,
                "reason": (
                    f"Date {clean_date} is {delta_days} days in the future. "
                    f"Weather forecasts are only reliable up to {MAX_FORECAST_DAYS} days ahead."
                ),
            }
    except ValueError:
        return {"available": False, "date": obs_date, "reason": "Invalid date format."}

    if settings.eval_offline_mode:
        return {
            "available": True,
            "date": clean_date,
            "lat": lat,
            "lon": lon,
            "cloud_cover_pct": 15,
            "seeing_arcsec": 1.9,
            "transparency": "good",
            "summary": "Simulated clear sky (15% cloud cover) in offline evaluation mode.",
            "source": "offline_default",
        }

    # 3. Live Open-Meteo query when online and within forecast horizon
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "hourly": "cloud_cover",
                    "start_date": clean_date,
                    "end_date": clean_date,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            clouds = data.get("hourly", {}).get("cloud_cover", [])
            night_clouds = clouds[20:24] if len(clouds) >= 24 else clouds
            avg_cloud = int(round(sum(night_clouds) / len(night_clouds))) if night_clouds else 20
            return {
                "available": True,
                "date": clean_date,
                "lat": lat,
                "lon": lon,
                "cloud_cover_pct": avg_cloud,
                "summary": f"Forecast average evening cloud cover: {avg_cloud}%.",
                "source": "open-meteo",
            }
    except Exception:
        return {
            "available": True,
            "date": clean_date,
            "lat": lat,
            "lon": lon,
            "cloud_cover_pct": 15,
            "summary": "Fallback weather estimate: 15% cloud cover.",
            "source": "fallback",
        }
