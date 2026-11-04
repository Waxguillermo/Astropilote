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
from astropilote.optics import angular_separation_deg
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


@lru_cache(maxsize=1)
def _get_timezone_finder():
    """Return a singleton `TimezoneFinder` instance for exact worldwide IANA timezone lookup."""
    from timezonefinder import TimezoneFinder

    return TimezoneFinder(in_memory=True)


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


def _format_utc_offset(offset_h: float) -> str:
    """Format fractional hour offset (e.g. `+1.0` -> `+1`, `+5.5` -> `+5:30`, `-3.0` -> `-3`)."""
    sign = "+" if offset_h >= 0 else "-"
    total_minutes = int(round(abs(offset_h) * 60.0))
    hh, mm = divmod(total_minutes, 60)
    if mm == 0:
        return f"{sign}{hh}"
    return f"{sign}{hh}:{mm:02d}"


def _resolve_timezone_details(
    lat: float, lon: float, d: date
) -> tuple[float, str, str | None, ZoneInfo | None]:
    """Resolve exact fractional timezone offset, display label, IANA name, and `ZoneInfo`."""
    tz_name: str | None = None
    try:
        tf = _get_timezone_finder()
        tz_name = tf.timezone_at(lat=lat, lng=lon) or tf.certain_timezone_at(lat=lat, lng=lon)
    except Exception:
        tz_name = None

    if tz_name is not None:
        try:
            tz = ZoneInfo(tz_name)
            probe = datetime(d.year, d.month, d.day, 21, 0, tzinfo=tz)
            utcoff = probe.utcoffset()
            if utcoff is not None:
                offset_h = round(utcoff.total_seconds() / 3600.0, 4)
                off_str = _format_utc_offset(offset_h)
                abbr = probe.tzname() or ""
                if abbr and not abbr.startswith(("+", "-")):
                    label = f"{abbr} ({tz_name}, UTC{off_str})"
                else:
                    label = f"{tz_name} (UTC{off_str})"
                return offset_h, label, tz_name, tz
        except ZoneInfoNotFoundError:
            pass

    offset_h = float(round(lon / 15.0))
    off_str = _format_utc_offset(offset_h)
    return offset_h, f"UTC{off_str}", None, None


def _resolve_timezone_info(lat: float, lon: float, d: date) -> tuple[float, str]:
    """Resolve civil timezone offset in hours (`float`) and human-readable label via `TimezoneFinder` + `ZoneInfo`."""
    offset_h, label, _, _ = _resolve_timezone_details(lat, lon, d)
    return offset_h, label


def _local_to_utc_naive(
    t_local_naive: datetime, tz: ZoneInfo | None, fallback_offset_h: float
) -> datetime:
    """Convert a naive local wall-clock datetime to naive UTC, respecting exact DST transitions if `tz` is known."""
    if tz is not None:
        aware = t_local_naive.replace(tzinfo=tz)
        return aware.astimezone(UTC).replace(tzinfo=None)
    return t_local_naive - timedelta(hours=fallback_offset_h)


def _altaz_from_lst(
    ra_deg: float,
    dec_deg: float,
    lst_deg: float,
    sin_lat: float,
    cos_lat: float,
) -> tuple[float, float, str]:
    """Fast inner spherical conversion from precomputed `lst_deg`, `sin_lat`, `cos_lat`."""
    ha_rad = math.radians((lst_deg - ra_deg) % 360.0)
    dec_rad = math.radians(dec_deg)

    sin_alt = math.sin(dec_rad) * sin_lat + math.cos(dec_rad) * cos_lat * math.cos(ha_rad)
    sin_alt = max(-1.0, min(1.0, sin_alt))
    alt_rad = math.asin(sin_alt)

    cos_alt = math.cos(alt_rad)
    if abs(cos_alt) < 1e-9 or abs(cos_lat) < 1e-9:
        az_deg = 0.0
    else:
        cos_az = (math.sin(dec_rad) - math.sin(alt_rad) * sin_lat) / (cos_alt * cos_lat)
        cos_az = max(-1.0, min(1.0, cos_az))
        az_deg = math.degrees(math.acos(cos_az))
        if math.sin(ha_rad) > 0:
            az_deg = 360.0 - az_deg

    alt_deg = round(math.degrees(alt_rad), 1)
    az_deg = round(az_deg % 360.0, 1)
    compass = COMPASS_DIRS[int(((az_deg + 22.5) % 360.0) // 45)]
    return alt_deg, az_deg, compass


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
    lat_rad = math.radians(lat_deg)
    return _altaz_from_lst(ra_deg, dec_deg, lst_deg, math.sin(lat_rad), math.cos(lat_rad))


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


@lru_cache(maxsize=256)
def _compute_night_ephemeris_cached(lat_r: float, lon_r: float, obs_date_iso: str) -> dict:
    """Cached core ephemeris calculation keyed on 3-decimal `(lat, lon)` and ISO date."""
    d = date.fromisoformat(obs_date_iso)
    tz_offset_hours, tz_label, tz_name, tz_obj = _resolve_timezone_details(lat_r, lon_r, d)
    lat_rad = math.radians(lat_r)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)

    # Sweep 15:00 local on day `d` through 09:00 local on `d + 1` (18 hours in 5-min steps)
    start_local = datetime(d.year, d.month, d.day, 15, 0)
    sunset_local: str | None = None
    sunrise_local: str | None = None
    nautical_dark_start: str | None = None
    nautical_dark_end: str | None = None
    last_nautical_local: str | None = None
    dark_start: str | None = None
    dark_end: str | None = None
    last_dark_local: str | None = None
    moon_set: str | None = None
    moon_rise: str | None = None

    prev_sun_alt: float | None = None
    prev_moon_alt: float | None = None
    min_sun_alt = 90.0
    dark_samples = 0
    moonless_dark_samples = 0

    for mins in range(0, 18 * 60 + 1, 5):
        t_local = start_local + timedelta(minutes=mins)
        t_str = t_local.strftime("%H:%M")
        t_utc = _local_to_utc_naive(t_local, tz_obj, tz_offset_hours)
        jd = _julian_date(t_utc)
        lst_deg = (_gmst_degrees(jd, t_utc) + lon_r) % 360.0

        s_ra, s_dec = _sun_radec(jd)
        s_alt, _, _ = _altaz_from_lst(s_ra, s_dec, lst_deg, sin_lat, cos_lat)
        min_sun_alt = min(min_sun_alt, s_alt)

        m_ra, m_dec, step_illum = _moon_radec_and_phase(jd)
        m_alt, _, _ = _altaz_from_lst(m_ra, m_dec, lst_deg, sin_lat, cos_lat)

        if s_alt <= -12.0:
            if nautical_dark_start is None:
                nautical_dark_start = t_str
            last_nautical_local = t_str

        if s_alt <= -18.0:
            dark_samples += 1
            if dark_start is None:
                dark_start = t_str
            last_dark_local = t_str
            if m_alt <= 0.0 or step_illum <= 10:
                moonless_dark_samples += 1

        if prev_sun_alt is not None:
            if prev_sun_alt > -0.833 and s_alt <= -0.833 and sunset_local is None:
                sunset_local = t_str
            elif (
                prev_sun_alt <= -0.833
                and s_alt > -0.833
                and (sunset_local is not None or mins > 0)
                and sunrise_local is None
            ):
                sunrise_local = t_str

            if (
                prev_sun_alt <= -12.0
                and s_alt > -12.0
                and nautical_dark_start is not None
                and nautical_dark_end is None
            ):
                nautical_dark_end = t_str

            if (
                prev_sun_alt <= -18.0
                and s_alt > -18.0
                and dark_start is not None
                and dark_end is None
            ):
                dark_end = t_str

        if prev_moon_alt is not None:
            if prev_moon_alt > 0.0 and m_alt <= 0.0 and moon_set is None:
                moon_set = t_str
            elif prev_moon_alt <= 0.0 and m_alt > 0.0 and moon_rise is None:
                moon_rise = t_str

        prev_sun_alt = s_alt
        prev_moon_alt = m_alt

    # Handle polar winter nights where the Sun stays below -12°/-18° past the end of the sweep
    if nautical_dark_start is not None and nautical_dark_end is None:
        nautical_dark_end = last_nautical_local
    if dark_start is not None and dark_end is None:
        dark_end = last_dark_local

    mid_local = datetime(d.year, d.month, d.day, 23, 30)
    mid_utc = _local_to_utc_naive(mid_local, tz_obj, tz_offset_hours)
    moon_ra_deg, moon_dec_deg, moon_illum_pct, moon_phase_name = _moon_radec_phase_and_label(
        _julian_date(mid_utc)
    )

    has_true_darkness = dark_start is not None and dark_end is not None and min_sun_alt <= -18.0
    dark_hours = round((dark_samples * 5) / 60.0, 1)
    moonless_dark_hours = round((moonless_dark_samples * 5) / 60.0, 1)

    if has_true_darkness:
        twilight_regime = "astronomical_darkness"
        astro_start: str | None = dark_start
        astro_end: str | None = dark_end
        obs_win_start: str | None = nautical_dark_start or dark_start
        obs_win_end: str | None = nautical_dark_end or dark_end
        twilight_note = f"Astronomical darkness from {dark_start} to {dark_end} ({tz_label})."
    elif min_sun_alt > -0.833:
        twilight_regime = "midnight_sun"
        astro_start = None
        astro_end = None
        obs_win_start = None
        obs_win_end = None
        twilight_note = (
            "Midnight sun — the Sun remains above the horizon all night; "
            "deep-sky observation is impossible."
        )
    elif min_sun_alt > -6.0:
        twilight_regime = "civil_twilight"
        astro_start = None
        astro_end = None
        obs_win_start = None
        obs_win_end = None
        twilight_note = (
            f"Civil twilight persists all night (Sun only reaches {min_sun_alt:.1f}°, above -6°); "
            "sky stays too bright for deep-sky observation."
        )
    elif min_sun_alt > -12.0:
        twilight_regime = "nautical_twilight"
        astro_start = None
        astro_end = None
        obs_win_start = None
        obs_win_end = None
        twilight_note = (
            f"Nautical twilight persists all night (Sun only reaches {min_sun_alt:.1f}°, above -12°); "
            "no dark sky window for deep-sky observation."
        )
    else:
        # -18.0 < min_sun_alt <= -12.0 (e.g. Paris in June): astronomical twilight persists
        twilight_regime = "astronomical_twilight"
        astro_start = nautical_dark_start
        astro_end = nautical_dark_end
        obs_win_start = nautical_dark_start
        obs_win_end = nautical_dark_end
        win_txt = (
            f"{nautical_dark_start} to {nautical_dark_end} ({tz_label})"
            if nautical_dark_start and nautical_dark_end
            else tz_label
        )
        twilight_note = (
            f"Persistent astronomical twilight around solar midnight (Sun reaches {min_sun_alt:.1f}°, "
            f"between -12° and -18°, usable twilight window {win_txt}); no true astronomical darkness, "
            "though planetary, lunar, bright cluster, and narrowband imaging remain possible."
        )

    return {
        "date": d.isoformat(),
        "weekday": d.strftime("%A"),
        "lat": lat_r,
        "lon": lon_r,
        "timezone": tz_name or tz_label,
        "tz_offset_hours": tz_offset_hours,
        "tz_label": tz_label,
        "sunset_local": sunset_local,
        "sunrise_local": sunrise_local,
        "has_astronomical_darkness": has_true_darkness,
        "twilight_regime": twilight_regime,
        "twilight_note": twilight_note,
        "astronomical_dark_start": astro_start,
        "astronomical_dark_end": astro_end,
        "observing_window_start": obs_win_start,
        "observing_window_end": obs_win_end,
        "astronomical_dark_hours": dark_hours,
        "moonless_dark_hours": moonless_dark_hours,
        "min_sun_altitude_deg": round(min_sun_alt, 1),
        "moon_illumination_pct": moon_illum_pct,
        "moon_phase_name": moon_phase_name,
        "moon_ra_deg": round(moon_ra_deg, 2),
        "moon_dec_deg": round(moon_dec_deg, 2),
        "moon_set_local": moon_set,
        "moon_rise_local": moon_rise,
        "summary": twilight_note,
    }


def compute_night_ephemeris(lat: float, lon: float, obs_date_str: str) -> dict:
    """Compute astronomical darkness window and Moon conditions for the night starting on `obs_date_str`."""
    cached = _compute_night_ephemeris_cached(
        round(float(lat), 3),
        round(float(lon), 3),
        obs_date_str.strip()[:10],
    )
    return {**cached, "lat": lat, "lon": lon}


compute_night_ephemeris.cache_info = _compute_night_ephemeris_cached.cache_info  # type: ignore[attr-defined]
compute_night_ephemeris.cache_clear = _compute_night_ephemeris_cached.cache_clear  # type: ignore[attr-defined]


def night(lat: float, lon: float, obs_date: str, days: int = 1) -> dict:
    """Calculate astronomical darkness start/end times and Moon phase/rise/set for one or more consecutive nights."""
    span = max(1, min(int(days or 1), 14))
    start_d = date.fromisoformat(obs_date.strip()[:10])
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
    """Compute altitude, direction, airmass & lunar separation every 30 min across the real dark night window."""
    d = date.fromisoformat(obs_date.strip()[:10])
    night_info = compute_night_ephemeris(lat, lon, obs_date)
    tz_offset, _, _, tz_obj = _resolve_timezone_details(lat, lon, d)

    lat_rad = math.radians(lat)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)

    # Sample 30-min slots across the entire potential night (16:30 to 07:30 local), filtered on Sun <= -12.0°
    base_local = datetime(d.year, d.month, d.day, 16, 30)
    precomputed_slots: list[tuple[str, float]] = []
    for i in range(31):
        t_loc = base_local + timedelta(minutes=30 * i)
        t_utc = _local_to_utc_naive(t_loc, tz_obj, tz_offset)
        jd = _julian_date(t_utc)
        lst_deg = (_gmst_degrees(jd, t_utc) + lon) % 360.0
        s_ra, s_dec = _sun_radec(jd)
        s_alt, _, _ = _altaz_from_lst(s_ra, s_dec, lst_deg, sin_lat, cos_lat)
        if s_alt <= -12.0:
            precomputed_slots.append((t_loc.strftime("%H:%M"), lst_deg))

    moon_ra_deg = float(night_info["moon_ra_deg"])
    moon_dec_deg = float(night_info["moon_dec_deg"])

    objects_report: dict[str, dict] = {}
    unique_raw_ids = list(dict.fromkeys(object_ids))

    for raw_id in unique_raw_ids:
        rec = resolve_object_record(raw_id)
        if rec is None:
            objects_report[raw_id] = {
                "found": False,
                "observable": False,
                "reason": f"Object '{raw_id}' not found in corpus.",
            }
            continue

        doc_id = rec["doc_id"]
        if doc_id in objects_report:
            continue

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

        moon_sep = round(
            angular_separation_deg(float(ra_deg), float(dec_deg), moon_ra_deg, moon_dec_deg), 1
        )

        if not precomputed_slots:
            objects_report[doc_id] = {
                "found": True,
                "doc_id": doc_id,
                "title": rec["title"],
                "object_type": meta.get("object_type"),
                "constellation": meta.get("constellation"),
                "magnitude": meta.get("magnitude"),
                "size_major_arcmin": meta.get("size_major_arcmin"),
                "narrowband_friendly": meta.get("narrowband_friendly", False),
                "observable": False,
                "max_altitude_deg": -90.0,
                "moon_separation_deg": moon_sep,
                "best_slot": None,
                "slots": [],
                "reason": night_info["twilight_note"],
            }
            continue

        samples: list[dict] = []
        best_slot: dict | None = None
        ra_f = float(ra_deg)
        dec_f = float(dec_deg)
        for time_str, lst_deg in precomputed_slots:
            alt, az, compass = _altaz_from_lst(ra_f, dec_f, lst_deg, sin_lat, cos_lat)
            airmass = round(1.0 / math.sin(math.radians(max(5.0, alt))), 2) if alt > 5.0 else None
            slot_entry = {
                "time": time_str,
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
            "moon_separation_deg": moon_sep,
            "best_slot": best_slot,
            "slots": samples,
        }

    return {
        "date": d.isoformat(),
        "lat": lat,
        "lon": lon,
        "moon_ra_deg": moon_ra_deg,
        "moon_dec_deg": moon_dec_deg,
        "night_window": {
            "dark_start": night_info["astronomical_dark_start"],
            "dark_end": night_info["astronomical_dark_end"],
            "observing_window_start": night_info["observing_window_start"],
            "observing_window_end": night_info["observing_window_end"],
            "has_astronomical_darkness": night_info["has_astronomical_darkness"],
            "twilight_regime": night_info["twilight_regime"],
            "twilight_note": night_info["twilight_note"],
            "moon_illumination_pct": night_info["moon_illumination_pct"],
            "moon_phase_name": night_info["moon_phase_name"],
            "moon_ra_deg": moon_ra_deg,
            "moon_dec_deg": moon_dec_deg,
            "moon_set_local": night_info["moon_set_local"],
            "moon_rise_local": night_info["moon_rise_local"],
        },
        "objects": objects_report,
    }
