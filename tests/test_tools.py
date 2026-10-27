"""Unit tests for the 6 AstroPilote agent tools (`search_catalog`, `get_object`, `night`, `visibility`, `geocode`, `weather`)."""

from astropilote.config import settings
from astropilote.tools.catalog import get_object, search_catalog
from astropilote.tools.sky import night, visibility
from astropilote.tools.world import geocode, rank_observing_nights, weather


def test_live_and_offline_geocode_and_weather() -> None:
    """Verify live Nominatim/Open-Meteo priority by default and fixture fallback only when EVAL_OFFLINE_MODE=true."""
    geo_live = geocode.invoke({"place_name": "Fontainebleau"})
    assert geo_live["found"] is True
    assert geo_live["source"] == "nominatim"
    assert abs(geo_live["lat"] - 48.4047) < 0.05
    assert abs(geo_live["lon"] - 2.7016) < 0.05

    today_iso = settings.get_reference_date().isoformat()
    w_live = weather.invoke({"lat": geo_live["lat"], "lon": geo_live["lon"], "obs_date": today_iso})
    assert w_live["available"] is True
    assert w_live["source"] == "open-meteo"
    assert 0 <= w_live["cloud_cover_pct"] <= 100

    # Offline evaluation mode reads deterministic fixtures
    prev_offline = settings.eval_offline_mode
    try:
        settings.eval_offline_mode = True
        geo_fix = geocode.invoke({"place_name": "Fontainebleau"})
        assert geo_fix["source"] == "fixture"
        w_fix = weather.invoke({"lat": 48.4047, "lon": 2.7016, "obs_date": "2026-10-17"})
        assert w_fix["available"] is True
        assert w_fix["cloud_cover_pct"] == 10
        assert w_fix["source"] == "fixture"
    finally:
        settings.eval_offline_mode = prev_offline

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

    # Visibility of M31 and Omega Centauri (C080) from 48.4°N
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
    assert m31["shooting_guide"] is not None
    assert m31["shooting_guide"]["orientation"]["constellation_code"] == "AND"
    assert "Andromeda" in m31["shooting_guide"]["orientation"]["constellation_name"]
    assert "HDR" in m31["shooting_guide"]["appearance"]["visual_and_photo_look"]

    # Equipment-aware shooting guide: 50mm lens on fixed tripod -> 500-rule sub-exposure
    m42_tripod = get_object.invoke(
        {
            "object_id": "M42",
            "focal_length_mm": 50.0,
            "aperture_mm": 28.0,
            "sensor_format": "APS-C",
            "mount_type": "tripod",
            "has_autoguiding": False,
            "filters": ["none"],
        }
    )
    sg_tripod = m42_tripod["shooting_guide"]
    assert sg_tripod["camera_settings"]["mount_type"] == "tripod"
    assert "500" in sg_tripod["camera_settings"]["recommended_sub_exposure"]
    assert "Orion" in sg_tripod["orientation"]["constellation_name"]

    # Tracked mount + dual-band filter on emission nebula (M42) vs broadband galaxy (M31)
    m42_nb = get_object.invoke(
        {
            "object_id": "M42",
            "focal_length_mm": 480.0,
            "aperture_mm": 80.0,
            "sensor_format": "APS-C",
            "mount_type": "equatorial_goto",
            "has_autoguiding": True,
            "filters": ["dual_band"],
        }
    )
    assert "Dual-Band" in m42_nb["shooting_guide"]["camera_settings"]["filter_recommendation"]

    m31_nb_warn = get_object.invoke(
        {
            "object_id": "M31",
            "focal_length_mm": 480.0,
            "aperture_mm": 80.0,
            "sensor_format": "APS-C",
            "mount_type": "equatorial_goto",
            "has_autoguiding": True,
            "filters": ["dual_band"],
        }
    )
    assert (
        "Avoid" in m31_nb_warn["shooting_guide"]["camera_settings"]["filter_recommendation"]
        or "continuum" in m31_nb_warn["shooting_guide"]["camera_settings"]["filter_recommendation"]
    )

    hits = search_catalog.invoke(
        {
            "query": "Dumbbell Nebula planetary nebula in Vulpecula",
            "narrowband_only": True,
            "top_k": 3,
        }
    )
    assert len(hits) >= 1
    assert any(h["doc_id"] == "M027" for h in hits)


def test_regional_geocode_multi_night_dst_and_ranking() -> None:
    """Verify live regional geocoding, CEST->CET DST transition, multi-night ephemeris, and night ranking."""
    for variant in ("Ile de France", "Île-de-France"):
        geo_idf = geocode.invoke({"place_name": variant})
        assert geo_idf["found"] is True
        assert 48.1 <= geo_idf["lat"] <= 49.2
        assert 1.5 <= geo_idf["lon"] <= 3.5

    geo_coord = geocode.invoke({"place_name": "48.6443°N, 2.7538°E"})
    assert geo_coord["found"] is True
    assert abs(geo_coord["lat"] - 48.6443) < 0.01

    # Multi-night ephemeris across the European DST switch (Oct 23 CEST UTC+2 -> Oct 25 CET UTC+1)
    multi_n = night.invoke({"lat": 48.6443, "lon": 2.7538, "obs_date": "2026-10-23", "days": 7})
    assert multi_n["days"] == 7
    assert len(multi_n["daily_nights"]) == 7
    assert multi_n["daily_nights"][0]["tz_offset_hours"] == 2
    assert multi_n["daily_nights"][2]["tz_offset_hours"] == 1  # Sunday 2026-10-25 switches to CET
    assert multi_n["daily_nights"][0]["moon_phase_name"] != ""

    multi_w = weather.invoke(
        {
            "lat": 48.6443,
            "lon": 2.7538,
            "obs_date": settings.get_reference_date().isoformat(),
            "days": 7,
        }
    )
    assert multi_w["days"] == 7
    assert len(multi_w["daily_forecasts"]) == 7

    # Ranking must place a clear night (17% clouds) well ahead of an overcast night (100% clouds)
    synthetic_forecasts = [
        {
            "available": True,
            "date": "2026-10-24",
            "cloud_cover_pct": 100,
            "clear_hours": 0,
            "max_precip_prob_pct": 100,
        },
        {
            "available": True,
            "date": "2026-10-28",
            "cloud_cover_pct": 17,
            "clear_hours": 8,
            "max_precip_prob_pct": 4,
        },
    ]
    subset_nights = [
        n for n in multi_n["daily_nights"] if n["date"] in ("2026-10-24", "2026-10-28")
    ]
    ranked = rank_observing_nights(subset_nights, synthetic_forecasts)
    assert ranked[0]["date"] == "2026-10-28"
    assert ranked[0]["observability_score"] > ranked[1]["observability_score"]
