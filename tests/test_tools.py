"""Unit tests for the 6 AstroPilote agent tools (`search_catalog`, `get_object`, `night`, `visibility`, `geocode`, `weather`)."""

from astropilote.tools.catalog import get_object, search_catalog
from astropilote.tools.sky import night, visibility
from astropilote.tools.world import geocode, weather


def test_geocode_and_weather_fixtures() -> None:
    """Verify deterministic geocoding and weather lookup from eval/fixtures/world_fixtures.json."""
    geo = geocode.invoke({"place_name": "Fontainebleau"})
    assert geo["found"] is True
    assert abs(geo["lat"] - 48.4047) < 0.01
    assert abs(geo["lon"] - 2.7016) < 0.01

    w_ok = weather.invoke({"lat": 48.4047, "lon": 2.7016, "obs_date": "2026-10-17"})
    assert w_ok["available"] is True
    assert w_ok["cloud_cover_pct"] == 10

    # Impossible future forecast (> 14 days ahead)
    w_imp = weather.invoke({"lat": 48.4047, "lon": 2.7016, "obs_date": "2028-03-03"})
    assert w_imp["available"] is False
    assert "14 days" in w_imp["reason"]


def test_night_and_visibility_calculations() -> None:
    """Verify astronomical darkness and altitude calculations on real sky coordinates."""
    n_oct = night.invoke({"lat": 48.4047, "lon": 2.7016, "obs_date": "2026-10-17"})
    assert n_oct["has_astronomical_darkness"] is True
    assert 0 <= n_oct["moon_illumination_pct"] <= 100

    # Summer solstice in Paris -> sun never dips below -18 deg
    n_solstice = night.invoke({"lat": 48.8566, "lon": 2.3522, "obs_date": "2026-06-21"})
    assert n_solstice["has_astronomical_darkness"] is False

    # Visibility of M31 and Omega Centauri (C080) from Fontainebleau
    vis = visibility.invoke(
        {
            "object_ids": ["M31", "C080"],
            "lat": 48.4047,
            "lon": 2.7016,
            "obs_date": "2026-10-17",
        }
    )
    assert vis["objects"]["M031"]["observable"] is True
    assert vis["objects"]["M031"]["max_altitude_deg"] > 60.0
    # Omega Centauri (Dec -47.5°) never rises at 48.4° N
    assert vis["objects"]["C080"]["observable"] is False
    assert vis["objects"]["C080"]["max_altitude_deg"] < 0.0


def test_catalog_get_object_and_search() -> None:
    """Verify get_object FOV framing calculation and search_catalog hybrid retrieval."""
    m31 = get_object.invoke({"object_id": "M31", "focal_length_mm": 480.0})
    assert m31["found"] is True
    assert m31["doc_id"] == "M031"
    assert m31["framing_advice"] is not None

    hits = search_catalog.invoke(
        {
            "query": "Dumbbell Nebula planetary nebula in Vulpecula",
            "narrowband_only": True,
            "top_k": 3,
        }
    )
    assert len(hits) >= 1
    assert any(h["doc_id"] == "M027" for h in hits)
