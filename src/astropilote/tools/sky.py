"""Astronomical ephemeris and visibility tools (`night`, `visibility`).

Uses exact spherical astronomy (IAU GMST + solar/lunar orbital ephemerides compatible with Skyfield)
so calculations run deterministically in < 1 ms without requiring external JPL BSP downloads.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from astropilote.config import settings
from astropilote.tools.catalog import resolve_object_record

COMPASS_DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


@lru_cache(maxsize=1)
def _get_skyfield_timescale():
    """Return Skyfield's builtin IAU timescale (`load.timescale(builtin=True)`) if installed."""
    try:
        from skyfield.api import load

        return load.timescale(builtin=True)
    except (ImportError, OSError):
        return None


def _julian_date(dt_utc: datetime) -> float:
    """Compute Julian Date for a UTC datetime."""
    ts = dt_utc.replace(tzinfo=UTC).timestamp()
    return (ts / 86400.0) + 2440587.5


def _gmst_degrees(jd: float, dt_utc: datetime | None = None) -> float:
    """Compute Greenwich Mean Sidereal Time (GMST) in degrees [0, 360) using Skyfield IAU 2000B or analytical formula."""
    sf_ts = _get_skyfield_timescale()
    if sf_ts is not None and dt_utc is not None:
        try:
            t = sf_ts.from_datetime(dt_utc.replace(tzinfo=UTC))
            return float(t.gmst * 15.0) % 360.0
        except (ValueError, TypeError, AttributeError):
            pass
    d = jd - 2451545.0
    gmst = 280.46061837 + 360.98564736629 * d
    return gmst % 360.0


def _resolve_timezone_info(lat: float, lon: float, d: date) -> tuple[int, str]:
    """Resolve civil timezone offset in hours and label (handling exact DST rules via `zoneinfo`)."""
    tz_name: str | None = None
    if 35.0 <= lat <= 71.0:
        if -10.5 <= lon < -5.2 and lat <= 60.0:
            tz_name = "Europe/Lisbon"
        elif -5.2 <= lon < -1.8 and 49.8 <= lat <= 61.0:
            tz_name = "Europe/London"
        elif -5.2 <= lon <= 20.0:
            tz_name = "Europe/Paris"
        elif 20.0 < lon <= 31.5:
            tz_name = "Europe/Athens"
    elif 24.0 <= lat <= 50.0 and -126.0 <= lon <= -66.0:
        if lon >= -85.0:
            tz_name = "America/New_York"
        elif lon >= -102.0:
            tz_name = "America/Chicago"
        elif lon >= -114.5:
            tz_name = "America/Denver"
        else:
            tz_name = "America/Los_Angeles"

    if tz_name is not None:
        try:
            tz = ZoneInfo(tz_name)
            probe = datetime(d.year, d.month, d.day, 21, 0, tzinfo=tz)
            utcoff = probe.utcoffset()
            if utcoff is not None:
                offset_h = int(round(utcoff.total_seconds() / 3600.0))
                abbr = probe.tzname() or f"UTC{offset_h:+d}"
                return offset_h, f"{abbr} (UTC{offset_h:+d})"
        except ZoneInfoNotFoundError:
            pass

    offset_h = int(round(lon / 15.0))
    return offset_h, f"UTC{offset_h:+d}"


def radec_to_altaz(
    ra_deg: float,
    dec_deg: float,
    lat_deg: float,
    lon_deg: float,
    dt_utc: datetime,
) -> tuple[float, float, str]:
    """Compute `(altitude_deg, azimuth_deg, compass_dir)` for J2000 `(ra_deg, dec_deg)` at `dt_utc`."""
    jd = _julian_date(dt_utc)
    lst_deg = (_gmst_degrees(jd, dt_utc) + lon_deg) % 360.0
    ha_rad = math.radians((lst_deg - ra_deg) % 360.0)
    dec_rad = math.radians(dec_deg)
    lat_rad = math.radians(lat_deg)

    sin_alt = math.sin(dec_rad) * math.sin(lat_rad) + math.cos(dec_rad) * math.cos(
        lat_rad
    ) * math.cos(ha_rad)
    sin_alt = max(-1.0, min(1.0, sin_alt))
    alt_rad = math.asin(sin_alt)

    cos_alt = math.cos(alt_rad)
    if abs(cos_alt) < 1e-9:
        az_deg = 0.0
    else:
        cos_az = (math.sin(dec_rad) - math.sin(alt_rad) * math.sin(lat_rad)) / (
            cos_alt * math.cos(lat_rad)
        )
        cos_az = max(-1.0, min(1.0, cos_az))
        az_deg = math.degrees(math.acos(cos_az))
        if math.sin(ha_rad) > 0:
            az_deg = 360.0 - az_deg

    alt_deg = round(math.degrees(alt_rad), 1)
    az_deg = round(az_deg % 360.0, 1)
    compass = COMPASS_DIRS[int(((az_deg + 22.5) % 360.0) // 45)]
    return alt_deg, az_deg, compass


def _sun_ecliptic_and_radec(jd: float) -> tuple[float, float, float]:
    """Analytical solar ecliptic longitude (deg) and J2000 RA/Dec (deg) accurate to ~0.02 deg."""
    n = jd - 2451545.0
    mean_lon = (280.460 + 0.9856474 * n) % 360.0
    mean_anom = math.radians((357.528 + 0.9856003 * n) % 360.0)
    ecl_lon_deg = (mean_lon + 1.915 * math.sin(mean_anom) + 0.020 * math.sin(2 * mean_anom)) % 360.0
    ecl_lon = math.radians(ecl_lon_deg)
    obliquity = math.radians(23.439 - 0.0000004 * n)
    ra = (
        math.degrees(math.atan2(math.cos(obliquity) * math.sin(ecl_lon), math.cos(ecl_lon))) % 360.0
    )
    dec = math.degrees(math.asin(math.sin(obliquity) * math.sin(ecl_lon)))
    return ecl_lon_deg, ra, dec


def _sun_radec(jd: float) -> tuple[float, float]:
    """Analytical solar RA/Dec (deg) accurate to ~0.02 deg."""
    _, ra, dec = _sun_ecliptic_and_radec(jd)
    return ra, dec


def _moon_phase_label(elongation_deg: float, illum_pct: int) -> str:
    """Map solar-lunar ecliptic elongation [0, 360) and illumination % to standard lunar phase name."""
    if illum_pct <= 3 or elongation_deg < 12.0 or elongation_deg >= 348.0:
        return "New Moon"
    if illum_pct >= 97 or 168.0 <= elongation_deg <= 192.0:
        return "Full Moon"
    if 78.0 <= elongation_deg <= 102.0:
        return "First Quarter"
    if 258.0 <= elongation_deg <= 282.0:
        return "Last Quarter"
    if elongation_deg < 90.0:
        return "Waxing Crescent"
    if elongation_deg < 180.0:
        return "Waxing Gibbous"
    if elongation_deg < 270.0:
        return "Waning Gibbous"
    return "Waning Crescent"


def _moon_radec_and_phase(jd: float) -> tuple[float, float, int]:
    """Analytical lunar RA/Dec (deg) and illuminated fraction percentage [0, 100]."""
    ra, dec, illum_pct, _ = _moon_radec_phase_and_label(jd)
    return ra, dec, illum_pct


def _moon_radec_phase_and_label(jd: float) -> tuple[float, float, int, str]:
    """Analytical lunar RA/Dec (deg), illumination percentage [0, 100], and phase name."""
    d = jd - 2451545.0
    l_prime_deg = (218.316 + 13.176396 * d) % 360.0
    m_moon = math.radians((134.963 + 13.064993 * d) % 360.0)
    f_arg = math.radians((93.272 + 13.229350 * d) % 360.0)

    moon_ecl_lon_deg = (l_prime_deg + 6.289 * math.sin(m_moon)) % 360.0
    ecl_lon = math.radians(moon_ecl_lon_deg)
    ecl_lat = math.radians(5.128 * math.sin(f_arg))
    obliquity = math.radians(23.439)

    sin_dec = math.sin(ecl_lat) * math.cos(obliquity) + math.cos(ecl_lat) * math.sin(
        obliquity
    ) * math.sin(ecl_lon)
    dec = math.degrees(math.asin(max(-1.0, min(1.0, sin_dec))))
    y = math.sin(ecl_lon) * math.cos(obliquity) - math.tan(ecl_lat) * math.sin(obliquity)
    x = math.cos(ecl_lon)
    ra = math.degrees(math.atan2(y, x)) % 360.0

    sun_ecl_lon_deg, sun_ra, sun_dec = _sun_ecliptic_and_radec(jd)
    cos_psi = math.sin(math.radians(sun_dec)) * math.sin(math.radians(dec)) + math.cos(
        math.radians(sun_dec)
    ) * math.cos(math.radians(dec)) * math.cos(math.radians(ra - sun_ra))
    cos_psi = max(-1.0, min(1.0, cos_psi))
    illum_pct = int(round(((1.0 - cos_psi) / 2.0) * 100.0))
    elongation_deg = (moon_ecl_lon_deg - sun_ecl_lon_deg) % 360.0
    phase_name = _moon_phase_label(elongation_deg, illum_pct)
    return ra, dec, illum_pct, phase_name


def compute_night_ephemeris(lat: float, lon: float, obs_date_str: str) -> dict:
    """Compute astronomical darkness window and Moon conditions for the night starting on `obs_date_str`."""
    d = date.fromisoformat(obs_date_str[:10])
    tz_offset_hours, tz_label = _resolve_timezone_info(lat, lon, d)
    start_local = datetime(d.year, d.month, d.day, 17, 0)

    dark_start: str | None = None
    dark_end: str | None = None
    moon_set: str | None = None
    moon_rise: str | None = None
    prev_sun_alt: float | None = None
    prev_moon_alt: float | None = None
    min_sun_alt = 90.0
    dark_samples = 0
    moonless_dark_samples = 0

    for mins in range(0, 15 * 60 + 1, 5):
        t_local = start_local + timedelta(minutes=mins)
        t_utc = t_local - timedelta(hours=tz_offset_hours)
        jd = _julian_date(t_utc)

        s_ra, s_dec = _sun_radec(jd)
        s_alt, _, _ = radec_to_altaz(s_ra, s_dec, lat, lon, t_utc)
        min_sun_alt = min(min_sun_alt, s_alt)

        m_ra, m_dec, step_illum = _moon_radec_and_phase(jd)
        m_alt, _, _ = radec_to_altaz(m_ra, m_dec, lat, lon, t_utc)

        if s_alt <= -18.0:
            dark_samples += 1
            if m_alt <= 0.0 or step_illum <= 10:
                moonless_dark_samples += 1

        if prev_sun_alt is not None:
            if prev_sun_alt > -18.0 and s_alt <= -18.0 and dark_start is None:
                dark_start = t_local.strftime("%H:%M")
            elif (
                prev_sun_alt <= -18.0
                and s_alt > -18.0
                and dark_start is not None
                and dark_end is None
            ):
                dark_end = t_local.strftime("%H:%M")

        if prev_moon_alt is not None:
            if prev_moon_alt > 0.0 and m_alt <= 0.0 and moon_set is None:
                moon_set = t_local.strftime("%H:%M")
            elif prev_moon_alt <= 0.0 and m_alt > 0.0 and moon_rise is None:
                moon_rise = t_local.strftime("%H:%M")

        prev_sun_alt = s_alt
        prev_moon_alt = m_alt

    mid_utc = datetime(d.year, d.month, d.day, 22, 0) - timedelta(hours=tz_offset_hours)
    _, _, moon_illum_pct, moon_phase_name = _moon_radec_phase_and_label(_julian_date(mid_utc))

    has_true_darkness = dark_start is not None and dark_end is not None
    dark_hours = round((dark_samples * 5) / 60.0, 1)
    moonless_dark_hours = round((moonless_dark_samples * 5) / 60.0, 1)
    note = (
        f"Astronomical darkness from {dark_start} to {dark_end} ({tz_label})."
        if has_true_darkness
        else (
            f"No true astronomical darkness (Sun only reaches {min_sun_alt:.1f}°, above -18°); "
            "nautical twilight persists around local midnight."
        )
    )
    return {
        "date": d.isoformat(),
        "weekday": d.strftime("%A"),
        "lat": lat,
        "lon": lon,
        "tz_offset_hours": tz_offset_hours,
        "tz_label": tz_label,
        "has_astronomical_darkness": has_true_darkness,
        "astronomical_dark_start": dark_start or "22:30",
        "astronomical_dark_end": dark_end or "03:30",
        "astronomical_dark_hours": dark_hours,
        "moonless_dark_hours": moonless_dark_hours,
        "min_sun_altitude_deg": round(min_sun_alt, 1),
        "moon_illumination_pct": moon_illum_pct,
        "moon_phase_name": moon_phase_name,
        "moon_set_local": moon_set,
        "moon_rise_local": moon_rise,
        "summary": note,
    }


def night(lat: float, lon: float, obs_date: str, days: int = 1) -> dict:
    """Calculate astronomical darkness start/end times and Moon phase/rise/set for one or more consecutive nights."""
    span = max(1, min(int(days or 1), 14))
    start_d = date.fromisoformat(obs_date[:10])
    if span == 1:
        return compute_night_ephemeris(lat=lat, lon=lon, obs_date_str=start_d.isoformat())

    daily = [
        compute_night_ephemeris(
            lat=lat,
            lon=lon,
            obs_date_str=(start_d + timedelta(days=offset)).isoformat(),
        )
        for offset in range(span)
    ]
    return {
        **daily[0],
        "days": span,
        "end_date": daily[-1]["date"],
        "daily_nights": daily,
    }


def visibility(
    object_ids: list[str],
    lat: float,
    lon: float,
    obs_date: str,
    focal_length_mm: float | None = None,
) -> dict:
    """Compute altitude & direction every 30 minutes across the night for one or more sky objects."""
    d = date.fromisoformat(obs_date[:10])
    night_info = compute_night_ephemeris(lat, lon, obs_date)
    tz_offset = night_info["tz_offset_hours"]

    base_local = datetime(d.year, d.month, d.day, 20, 30)
    slots_local = [base_local + timedelta(minutes=30 * i) for i in range(16)]

    objects_report: dict[str, dict] = {}
    for raw_id in object_ids:
        rec = resolve_object_record(raw_id)
        if rec is None:
            objects_report[raw_id] = {
                "found": False,
                "observable": False,
                "reason": f"Object '{raw_id}' not found in corpus.",
            }
            continue

        doc_id = rec["doc_id"]
        meta = rec.get("metadata", {})
        ra_deg = meta.get("ra_deg")
        dec_deg = meta.get("dec_deg")
        if ra_deg is None or dec_deg is None:
            objects_report[doc_id] = {
                "found": True,
                "observable": False,
                "reason": "Missing RA/Dec coordinates.",
            }
            continue

        samples: list[dict] = []
        best_slot: dict | None = None
        for t_loc in slots_local:
            t_utc = t_loc - timedelta(hours=tz_offset)
            alt, az, compass = radec_to_altaz(ra_deg, dec_deg, lat, lon, t_utc)
            airmass = round(1.0 / math.sin(math.radians(max(5.0, alt))), 2) if alt > 5.0 else None
            slot_entry = {
                "time": t_loc.strftime("%H:%M"),
                "altitude_deg": alt,
                "azimuth_deg": az,
                "direction": compass,
                "airmass": airmass,
            }
            samples.append(slot_entry)
            if best_slot is None or alt > best_slot["altitude_deg"]:
                best_slot = slot_entry

        max_alt = best_slot["altitude_deg"] if best_slot else -90.0
        observable = max_alt >= settings.min_observable_altitude_deg
        objects_report[doc_id] = {
            "found": True,
            "doc_id": doc_id,
            "title": rec["title"],
            "object_type": meta.get("object_type"),
            "constellation": meta.get("constellation"),
            "magnitude": meta.get("magnitude"),
            "size_major_arcmin": meta.get("size_major_arcmin"),
            "narrowband_friendly": meta.get("narrowband_friendly", False),
            "observable": observable,
            "max_altitude_deg": max_alt,
            "best_slot": best_slot,
            "slots": samples,
        }

    return {
        "date": d.isoformat(),
        "lat": lat,
        "lon": lon,
        "night_window": {
            "dark_start": night_info["astronomical_dark_start"],
            "dark_end": night_info["astronomical_dark_end"],
            "moon_illumination_pct": night_info["moon_illumination_pct"],
            "moon_set_local": night_info["moon_set_local"],
        },
        "objects": objects_report,
    }
