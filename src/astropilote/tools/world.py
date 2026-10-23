"""Real-world context tools (`geocode`, `weather`) and deterministic multi-night ranking."""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import httpx

from astropilote.config import settings

FIXTURES_PATH = Path(__file__).resolve().parents[3] / "eval" / "fixtures" / "world_fixtures.json"

_COORD_PAIR_RE = re.compile(
    r"^\s*([+-]?\d{1,2}(?:\.\d+)?)\s*°?\s*([NSns])?\s*[,;/\s]+\s*([+-]?\d{1,3}(?:\.\d+)?)\s*°?\s*([EWew])?\s*$"
)


def _normalize_place_key(raw: str) -> str:
    """Normalize accents, punctuation, and whitespace for resilient location lookup."""
    decomposed = unicodedata.normalize("NFKD", raw)
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = re.sub(r"[-'’_]+", " ", ascii_only.lower())
    cleaned = re.sub(r"\b(?:region|area|departement|department)\b", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


@lru_cache(maxsize=1)
def _load_fixtures() -> dict:
    if FIXTURES_PATH.exists():
        return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    return {"locations": {}, "weather_by_date": {}}


@lru_cache(maxsize=128)
def _nominatim_lookup(query: str) -> dict | None:
    """Query OpenStreetMap Nominatim with caching to avoid duplicate lookups."""
    with httpx.Client(
        timeout=10.0,
        headers={"User-Agent": "AstroPilote-Albert-School-x-Mines-Paris-Project/1.1"},
    ) as client:
        resp = client.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1, "accept-language": "en,fr"},
        )
        resp.raise_for_status()
        items = resp.json()
        if items:
            return items[0]
    return None


def _lookup_fixture_location(raw: str) -> dict | None:
    """Look up `raw` in offline evaluation fixtures (`EVAL_OFFLINE_MODE=true`)."""
    fixtures = _load_fixtures()
    first_segment = raw.split(",")[0].strip()
    norm_key = _normalize_place_key(first_segment)
    locations_map = fixtures.get("locations", {})
    matched = locations_map.get(first_segment.lower()) or locations_map.get(norm_key)
    if matched is None:
        for fk, finfo in locations_map.items():
            if _normalize_place_key(fk) == norm_key:
                matched = finfo
                break
    if matched is None:
        return None
    display = str(matched["name"])
    return {
        "found": True,
        "place_name": display,
        "short_name": display.split(",")[0].strip(),
        "lat": matched["lat"],
        "lon": matched["lon"],
        "elevation_m": matched.get("elevation_m", 100),
        "source": "fixture",
    }


def geocode(place_name: str) -> dict:
    """Convert a city, region, landmark, or coordinate string into geographic `(lat, lon)`."""
    raw = place_name.strip()
    if not raw:
        return {"found": False, "place_name": place_name, "message": "Empty location name."}

    coord_m = _COORD_PAIR_RE.match(raw)
    if coord_m:
        lat_v = float(coord_m.group(1))
        lon_v = float(coord_m.group(3))
        if (coord_m.group(2) or "").upper() == "S":
            lat_v = -abs(lat_v)
        if (coord_m.group(4) or "").upper() == "W":
            lon_v = -abs(lon_v)
        if -90.0 <= lat_v <= 90.0 and -180.0 <= lon_v <= 180.0:
            return {
                "found": True,
                "place_name": f"{lat_v:.4f}, {lon_v:.4f}",
                "short_name": f"{lat_v:.2f}°, {lon_v:.2f}°",
                "lat": round(lat_v, 4),
                "lon": round(lon_v, 4),
                "source": "coordinates",
            }

    if settings.eval_offline_mode:
        offline_hit = _lookup_fixture_location(raw)
        if offline_hit is not None:
            return offline_hit
        return {
            "found": False,
            "place_name": place_name,
            "message": f"Location '{place_name}' not found in offline evaluation fixtures.",
        }

    try:
        top = _nominatim_lookup(raw)
        if top:
            display = str(top.get("display_name", place_name))
            short_label = str(top.get("name") or display.split(",")[0]).strip()
            return {
                "found": True,
                "place_name": display,
                "short_name": short_label,
                "lat": round(float(top["lat"]), 4),
                "lon": round(float(top["lon"]), 4),
                "source": "nominatim",
            }
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        return {"found": False, "place_name": place_name, "message": f"Geocoding error: {exc}"}

    return {"found": False, "place_name": place_name, "message": "Location not found."}


def _summarize_single_night_hourly(
    d: date,
    times: list[str],
    clouds: list[float | None],
    precips: list[float | None],
    winds: list[float | None],
    humidities: list[float | None],
) -> dict:
    """Aggregate Open-Meteo local hourly forecasts across the astronomical night (`20:00` -> `05:00`)."""
    d_iso = d.isoformat()
    next_iso = (d + timedelta(days=1)).isoformat()

    evening_idx: list[int] = []
    late_idx: list[int] = []
    night_slots: list[tuple[str, float, float, float, float]] = []

    for idx, ts in enumerate(times):
        if len(ts) < 13:
            continue
        day_part = ts[:10]
        hour_val = int(ts[11:13])
        is_evening = day_part == d_iso and hour_val >= 20
        is_late = day_part == next_iso and hour_val <= 5
        if not (is_evening or is_late):
            continue

        c_val = float(clouds[idx]) if idx < len(clouds) and clouds[idx] is not None else 50.0
        p_val = float(precips[idx]) if idx < len(precips) and precips[idx] is not None else 0.0
        w_val = float(winds[idx]) if idx < len(winds) and winds[idx] is not None else 10.0
        h_val = (
            float(humidities[idx])
            if idx < len(humidities) and humidities[idx] is not None
            else 70.0
        )

        if is_evening:
            evening_idx.append(len(night_slots))
        else:
            late_idx.append(len(night_slots))
        night_slots.append((ts[11:16], c_val, p_val, w_val, h_val))

    if not night_slots:
        return {
            "available": False,
            "date": d_iso,
            "weekday": d.strftime("%A"),
            "reason": f"No nighttime hourly forecast slots returned for {d_iso}.",
        }

    avg_cloud = int(round(sum(s[1] for s in night_slots) / len(night_slots)))
    eve_cloud = (
        int(round(sum(night_slots[i][1] for i in evening_idx) / len(evening_idx)))
        if evening_idx
        else avg_cloud
    )
    late_cloud = (
        int(round(sum(night_slots[i][1] for i in late_idx) / len(late_idx)))
        if late_idx
        else avg_cloud
    )
    max_precip = int(round(max(s[2] for s in night_slots)))
    avg_wind = round(sum(s[3] for s in night_slots) / len(night_slots), 1)
    avg_hum = int(round(sum(s[4] for s in night_slots) / len(night_slots)))

    clear_slot_hours = [s[0] for s in night_slots if s[1] <= 35.0 and s[2] <= 25.0]
    clear_hours = len(clear_slot_hours)
    if clear_slot_hours:
        end_hr = (int(clear_slot_hours[-1][:2]) + 1) % 24
        best_clear_window: str | None = f"{clear_slot_hours[0]}–{end_hr:02d}:00"
    else:
        best_clear_window = None

    if avg_cloud <= 20 and avg_hum <= 80:
        transparency = "excellent"
    elif avg_cloud <= 40:
        transparency = "good"
    elif avg_cloud <= 65:
        transparency = "moderate"
    else:
        transparency = "poor"

    win_note = f", clearest {best_clear_window}" if best_clear_window else ""
    summary = (
        f"{d.strftime('%a %Y-%m-%d')}: {avg_cloud}% night clouds "
        f"(evening {eve_cloud}%, late-night {late_cloud}%, {clear_hours}/{len(night_slots)}h clear{win_note}), "
        f"max precip {max_precip}%, wind {avg_wind} km/h, humidity {avg_hum}%."
    )
    return {
        "available": True,
        "date": d_iso,
        "weekday": d.strftime("%A"),
        "cloud_cover_pct": avg_cloud,
        "evening_cloud_cover_pct": eve_cloud,
        "late_night_cloud_cover_pct": late_cloud,
        "clear_hours": clear_hours,
        "total_night_hours": len(night_slots),
        "best_clear_window": best_clear_window,
        "max_precip_prob_pct": max_precip,
        "wind_kmh": avg_wind,
        "humidity_pct": avg_hum,
        "transparency": transparency,
        "summary": summary,
        "source": "open-meteo",
    }


def weather(lat: float, lon: float, obs_date: str, days: int = 1) -> dict:
    """Fetch hourly nighttime cloud cover and observing conditions for `(lat, lon)` over `days` nights."""
    clean_date = obs_date.strip()[:10]
    span = max(1, min(int(days or 1), settings.max_forecast_days))

    try:
        start_d = date.fromisoformat(clean_date)
        ref_today = settings.get_reference_date()
        delta_days = (start_d - ref_today).days
        if delta_days > settings.max_forecast_days:
            return {
                "available": False,
                "date": clean_date,
                "reason": (
                    f"Date {clean_date} is {delta_days} days in the future. "
                    f"Weather forecasts are only reliable up to {settings.max_forecast_days} days ahead."
                ),
            }
    except ValueError:
        return {"available": False, "date": obs_date, "reason": "Invalid date format."}

    max_allowed_end = ref_today + timedelta(days=settings.max_forecast_days)
    end_d = min(start_d + timedelta(days=span - 1), max_allowed_end)
    effective_span = max(1, (end_d - start_d).days + 1)

    if settings.eval_offline_mode:
        fixtures = _load_fixtures()
        daily_offline = []
        for offset in range(effective_span):
            cur_d = start_d + timedelta(days=offset)
            cur_iso = cur_d.isoformat()
            fix_w = fixtures.get("weather_by_date", {}).get(cur_iso)
            if fix_w:
                daily_offline.append(
                    {
                        "available": True,
                        "date": cur_iso,
                        "weekday": cur_d.strftime("%A"),
                        "lat": lat,
                        "lon": lon,
                        **fix_w,
                        "source": "fixture",
                    }
                )
            else:
                daily_offline.append(
                    {
                        "available": True,
                        "date": cur_iso,
                        "weekday": cur_d.strftime("%A"),
                        "lat": lat,
                        "lon": lon,
                        "cloud_cover_pct": 15,
                        "clear_hours": 8,
                        "total_night_hours": 10,
                        "seeing_arcsec": 1.9,
                        "transparency": "good",
                        "summary": "Simulated clear sky (15% cloud cover) in offline evaluation mode.",
                        "source": "offline_default",
                    }
                )
        if effective_span == 1:
            return daily_offline[0]
        return {
            **daily_offline[0],
            "days": effective_span,
            "end_date": end_d.isoformat(),
            "daily_forecasts": daily_offline,
        }

    fetch_end_d = min(end_d + timedelta(days=1), ref_today + timedelta(days=15))
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "hourly": "cloud_cover,precipitation_probability,wind_speed_10m,relative_humidity_2m",
                    "timezone": "auto",
                    "start_date": start_d.isoformat(),
                    "end_date": fetch_end_d.isoformat(),
                },
            )
            resp.raise_for_status()
            hourly = resp.json().get("hourly", {})
            times = hourly.get("time", [])
            clouds = hourly.get("cloud_cover", [])
            precips = hourly.get("precipitation_probability", [])
            winds = hourly.get("wind_speed_10m", [])
            humidities = hourly.get("relative_humidity_2m", [])

            daily_results = []
            for offset in range(effective_span):
                cur_d = start_d + timedelta(days=offset)
                entry = _summarize_single_night_hourly(
                    cur_d, times, clouds, precips, winds, humidities
                )
                entry["lat"] = lat
                entry["lon"] = lon
                daily_results.append(entry)

            if effective_span == 1:
                return daily_results[0]
            return {
                **daily_results[0],
                "days": effective_span,
                "end_date": end_d.isoformat(),
                "daily_forecasts": daily_results,
            }
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        return {
            "available": False,
            "date": clean_date,
            "lat": lat,
            "lon": lon,
            "reason": f"Live meteorological forecast service unavailable: {exc}",
            "source": "error",
        }


def rank_observing_nights(
    daily_nights: list[dict],
    daily_forecasts: list[dict],
) -> list[dict]:
    """Deterministically score and rank multi-day nights combining cloud cover, precipitation, and Moon darkness."""
    weather_by_d = {str(w.get("date")): w for w in daily_forecasts if isinstance(w, dict)}
    ranked: list[dict] = []

    for n_info in daily_nights:
        d_str = str(n_info.get("date", ""))
        w_info = weather_by_d.get(d_str, {})
        w_avail = bool(w_info.get("available"))
        clouds = int(w_info.get("cloud_cover_pct", 50)) if w_avail else None
        clear_h = int(w_info.get("clear_hours", 0)) if w_avail else None
        max_precip = int(w_info.get("max_precip_prob_pct", 0)) if w_avail else 0

        moon_pct = int(n_info.get("moon_illumination_pct", 50))
        moonless_h = float(n_info.get("moonless_dark_hours", 0.0))

        if w_avail and clouds is not None:
            cloud_base = max(0.0, 100.0 - float(clouds))
            clear_bonus = min(15.0, float(clear_h or 0) * 1.8)
            rain_pen = max(0.0, (float(max_precip) - 25.0) * 0.45)
            clarity_score = max(0.0, min(100.0, 0.85 * cloud_base + clear_bonus - rain_pen))
            lunar_bonus = min(12.0, moonless_h * 2.0) + (100.0 - float(moon_pct)) * 0.08
            total_score = round(
                min(100.0, clarity_score * 0.82 + lunar_bonus * (clarity_score / 100.0)),
                1,
            )
            if clouds <= 25 and max_precip <= 20:
                rating = (
                    "Excellent (Clear & Dark)"
                    if moon_pct <= 45
                    else "Good (Clear Sky, Bright Moon)"
                )
            elif clouds <= 48 and (clear_h or 0) >= 3:
                rating = "Fair (Partly Clear Windows)"
            else:
                rating = "Poor (Mostly Cloudy / Rain Risk)"
        else:
            total_score = round(min(100.0, moonless_h * 8.0 + (100.0 - float(moon_pct)) * 0.5), 1)
            rating = "Dark Moon Window" if moon_pct <= 45 else "Bright Moon"

        recommended_focus = (
            "Galaxies, faint nebulae & Milky Way (dark sky)"
            if moon_pct <= 40 or moonless_h >= 3.5
            else "Bright star clusters, double stars, planetary & Hα/OIII narrowband nebulae (bright Moon)"
        )

        ranked.append(
            {
                "date": d_str,
                "weekday": n_info.get("weekday", ""),
                "observability_score": total_score,
                "rating": rating,
                "weather_available": w_avail,
                "cloud_cover_pct": clouds,
                "evening_cloud_cover_pct": w_info.get("evening_cloud_cover_pct"),
                "late_night_cloud_cover_pct": w_info.get("late_night_cloud_cover_pct"),
                "clear_hours": clear_h,
                "best_clear_window": w_info.get("best_clear_window"),
                "max_precip_prob_pct": w_info.get("max_precip_prob_pct"),
                "wind_kmh": w_info.get("wind_kmh"),
                "astronomical_dark_start": n_info.get("astronomical_dark_start"),
                "astronomical_dark_end": n_info.get("astronomical_dark_end"),
                "tz_label": n_info.get("tz_label", "UTC"),
                "moon_illumination_pct": moon_pct,
                "moon_phase_name": n_info.get("moon_phase_name", ""),
                "moon_rise_local": n_info.get("moon_rise_local"),
                "moon_set_local": n_info.get("moon_set_local"),
                "moonless_dark_hours": moonless_h,
                "recommended_focus": recommended_focus,
            }
        )

    ranked.sort(key=lambda item: float(item["observability_score"]), reverse=True)
    for idx, entry in enumerate(ranked, start=1):
        entry["rank"] = idx
    return ranked
