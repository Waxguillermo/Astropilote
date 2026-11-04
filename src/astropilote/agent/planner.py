"""Deterministic ephemeris target selector, White-Night / Midnight-Sun handler, and schedule builder."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Literal

from astropilote.agent.schemas import Intent, PlanItem
from astropilote.agent.validator import _parse_hhmm, _time_to_night_minutes
from astropilote.config import settings
from astropilote.optics import angular_separation_deg, compute_fov_arcmin
from astropilote.tools.catalog import load_documents_lookup
from astropilote.tools.sky import (
    _gmst_degrees,
    _julian_date,
    _moon_radec_and_phase,
    _resolve_timezone_info,
    _sun_radec,
    radec_to_altaz,
)

DarknessRegime = Literal["astronomical", "nautical_twilight", "bright_twilight_or_daylight"]


def _matches_intent_filters(doc: dict[str, Any], intent: Intent | None) -> bool:
    """Return True if a localizable corpus document satisfies user-requested catalog/metadata filters."""
    if intent is None:
        return True
    meta = doc.get("metadata") or {}
    if intent.catalog_filter and doc.get("catalog") != intent.catalog_filter:
        return False
    if (
        intent.constellation
        and str(meta.get("constellation") or "").upper() != intent.constellation.upper()
    ):
        return False
    if intent.object_type:
        otype = str(meta.get("object_type") or "")
        if intent.object_type == "galaxy":
            if not otype.startswith("galaxy"):
                return False
        elif intent.object_type == "nebula":
            if not (otype == "nebula" or otype.endswith("_nebula")):
                return False
        elif otype != intent.object_type:
            return False
    if intent.narrowband_only and not bool(meta.get("narrowband_friendly")):
        return False
    if intent.max_magnitude is not None:
        mag = meta.get("magnitude")
        if mag is None or float(mag) > float(intent.max_magnitude):
            return False
    return True


def _dark_window_utc_samples(
    lat: float,
    lon: float,
    obs_date_str: str,
    dark_start: str,
    dark_end: str,
    n_samples: int = 5,
) -> list[datetime]:
    """Generate `n_samples` UTC timestamps evenly spanning the local observing window."""
    obs_d = date.fromisoformat(obs_date_str[:10])
    tz_offset_h, _ = _resolve_timezone_info(lat, lon, obs_d)
    t_s = _parse_hhmm(dark_start) or time(21, 0)
    t_e = _parse_hhmm(dark_end) or time(5, 0)

    start_local = datetime(obs_d.year, obs_d.month, obs_d.day, t_s.hour, t_s.minute, tzinfo=UTC)
    if t_s.hour < 12:
        start_local += timedelta(days=1)
    end_local = datetime(obs_d.year, obs_d.month, obs_d.day, t_e.hour, t_e.minute, tzinfo=UTC)
    if t_e.hour < 12 or end_local <= start_local:
        end_local += timedelta(days=1)

    start_utc = start_local - timedelta(hours=tz_offset_h)
    end_utc = end_local - timedelta(hours=tz_offset_h)
    total_sec = max(1800.0, (end_utc - start_utc).total_seconds())
    if n_samples <= 1:
        return [start_utc + timedelta(seconds=total_sec / 2.0)]
    return [
        start_utc + timedelta(seconds=total_sec * i / (n_samples - 1)) for i in range(n_samples)
    ]


def compute_nautical_twilight_window(
    lat: float,
    lon: float,
    obs_date_str: str,
) -> tuple[str, str]:
    """Compute the darkest nautical-twilight window (`Sun <= -12°`) around solar midnight for white nights."""
    d = date.fromisoformat(obs_date_str[:10])
    tz_offset_h, _ = _resolve_timezone_info(lat, lon, d)
    start_local = datetime(d.year, d.month, d.day, 21, 0)
    naut_slots: list[str] = []
    min_alt = 90.0
    solar_midnight_hhmm = "01:00"

    for mins in range(0, 9 * 60 + 1, 15):
        t_loc = start_local + timedelta(minutes=mins)
        t_utc = t_loc - timedelta(hours=tz_offset_h)
        jd = _julian_date(t_utc)
        s_ra, s_dec = _sun_radec(jd)
        s_alt, _, _ = radec_to_altaz(s_ra, s_dec, lat, lon, t_utc)
        if s_alt < min_alt:
            min_alt = s_alt
            solar_midnight_hhmm = t_loc.strftime("%H:%M")
        if s_alt <= -12.0:
            naut_slots.append(t_loc.strftime("%H:%M"))

    if len(naut_slots) >= 2:
        # Clamp inside 22:30..03:30 if needed so visibility samples and validator stay aligned
        s_m = max(
            _time_to_night_minutes(_parse_hhmm(naut_slots[0]) or time(22, 30)),
            _time_to_night_minutes(time(22, 30)),
        )
        e_m = min(
            _time_to_night_minutes(_parse_hhmm(naut_slots[-1]) or time(3, 30)),
            _time_to_night_minutes(time(3, 30)),
        )
        if e_m > s_m:
            s_total = (s_m + 720) % 1440
            e_total = (e_m + 720) % 1440
            return (
                f"{s_total // 60:02d}:{s_total % 60:02d}",
                f"{e_total // 60:02d}:{e_total % 60:02d}",
            )

    mid_t = _parse_hhmm(solar_midnight_hhmm) or time(1, 0)
    mid_dt = datetime(d.year, d.month, d.day, mid_t.hour, mid_t.minute)
    return (mid_dt - timedelta(hours=1)).strftime("%H:%M"), (mid_dt + timedelta(hours=1)).strftime(
        "%H:%M"
    )


def classify_night_darkness(
    active_night: dict[str, Any] | None,
) -> tuple[DarknessRegime, str | None, str | None]:
    """Classify the night sky regime into astronomical darkness, nautical white night, or Midnight Sun / bright twilight."""
    if not active_night:
        return "astronomical", "21:00", "05:30"

    has_astro = active_night.get("has_astronomical_darkness")
    min_sun_alt = float(active_night.get("min_sun_altitude_deg", -25.0))

    if has_astro is False:
        if min_sun_alt > -12.0:
            return "bright_twilight_or_daylight", None, None
        obs_s = active_night.get("observing_window_start")
        obs_e = active_night.get("observing_window_end")
        if isinstance(obs_s, str) and obs_s and isinstance(obs_e, str) and obs_e:
            return "nautical_twilight", obs_s, obs_e
        lat = float(active_night.get("lat") or 48.85)
        lon = float(active_night.get("lon") or 2.35)
        obs_date_str = str(active_night.get("date") or settings.get_reference_date().isoformat())
        tw_start, tw_end = compute_nautical_twilight_window(lat, lon, obs_date_str)
        return "nautical_twilight", tw_start, tw_end

    return (
        "astronomical",
        str(active_night.get("astronomical_dark_start") or "21:00"),
        str(active_night.get("astronomical_dark_end") or "05:30"),
    )


def _select_dynamic_sky_targets(
    lat: float,
    lon: float,
    obs_date_str: str,
    dark_start: str,
    dark_end: str,
    moon_pct: int,
    intent: Intent | None,
    limit: int = 6,
) -> list[str]:
    """Scan all 447 localizable corpus objects to select optimal targets for `(lat, lon, obs_date_str)`.

    Precalculates GMST/LST and lunar coordinates outside the object loop for < 1 ms execution,
    computes exact trigonometric FOV via `optics.compute_fov_arcmin`, and filters/penalizes targets
    within `< 25°` of a bright Moon (`moon_pct >= 60`).
    """
    by_id, _ = load_documents_lookup()
    utc_samples = _dark_window_utc_samples(
        lat, lon, obs_date_str, dark_start, dark_end, n_samples=5
    )

    # Precalculate GMST -> Local Sidereal Time and observer sin/cos latitude once outside the 447-object loop
    lat_rad = math.radians(lat)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    lst_samples_deg: list[float] = [
        (_gmst_degrees(_julian_date(dt_u), dt_u) + lon) % 360.0 for dt_u in utc_samples
    ]

    mid_dt = utc_samples[len(utc_samples) // 2]
    moon_ra_deg, moon_dec_deg, _ = _moon_radec_and_phase(_julian_date(mid_dt))

    min_alt = max(settings.min_observable_altitude_deg, 25.0)
    is_visual_only = bool(intent and intent.mode == "visual")
    focal_mm = intent.focal_length_mm if intent else None
    sensor_fmt = intent.sensor_format if intent else "APS-C"
    fov_w_arcmin = (
        compute_fov_arcmin(focal_mm, sensor_fmt)[0] if (focal_mm and focal_mm > 0) else None
    )
    is_untracked_tripod = bool(intent and intent.mount_type == "tripod")
    user_filters = {str(f).lower() for f in (intent.filters if intent else [])}
    has_nb_filter = any(
        k in user_filters
        for k in ("dual_band", "dual-band", "narrowband_sho", "uhc_cls", "ha_oiii")
    )
    has_constellation_lock = bool(intent and intent.constellation)

    scored: list[tuple[float, str, str]] = []
    for doc_id, doc in by_id.items():
        if doc_id.startswith(("CON_", "GUIDE_")):
            continue
        meta = doc.get("metadata") or {}
        ra_deg, dec_deg = meta.get("ra_deg"), meta.get("dec_deg")
        if ra_deg is None or dec_deg is None:
            continue
        if not _matches_intent_filters(doc, intent):
            continue

        mag = meta.get("magnitude")
        if mag is None and (is_visual_only or is_untracked_tripod):
            continue
        if (
            mag is not None
            and float(mag) > 10.2
            and (intent is None or intent.max_magnitude is None)
        ):
            continue

        ra_f = float(ra_deg)
        dec_f = float(dec_deg)

        # Penalize or exclude targets glared by a bright Moon (< 25 deg angular separation when Moon >= 60%)
        moon_glare_penalty = 0.0
        if moon_pct >= 60:
            moon_sep_deg = angular_separation_deg(ra_f, dec_f, moon_ra_deg, moon_dec_deg)
            if moon_sep_deg < 25.0:
                if not has_constellation_lock:
                    continue
                moon_glare_penalty = -45.0

        dec_rad = math.radians(dec_f)
        sin_dec = math.sin(dec_rad)
        cos_dec = math.cos(dec_rad)
        peak_alt = -90.0
        for lst_deg in lst_samples_deg:
            ha_rad = math.radians((lst_deg - ra_f) % 360.0)
            sin_alt = max(-1.0, min(1.0, sin_dec * sin_lat + cos_dec * cos_lat * math.cos(ha_rad)))
            alt_deg = math.degrees(math.asin(sin_alt))
            if alt_deg > peak_alt:
                peak_alt = alt_deg

        if peak_alt < min_alt:
            continue

        otype = str(meta.get("object_type") or "deep_sky")
        nb_ok = bool(meta.get("narrowband_friendly"))
        cat = str(doc.get("catalog") or "")

        bright_score = (12.5 - float(mag)) * 5.2 if mag is not None else 22.0
        alt_score = min(peak_alt, 82.0) * 0.45

        moon_adj = 0.0
        if moon_pct >= 60 and (intent is None or not intent.object_type):
            if otype in ("open_cluster", "globular_cluster", "cluster_with_nebulosity") or nb_ok:
                moon_adj = 15.0 + (6.0 if (nb_ok and has_nb_filter) else 0.0)
            elif otype.startswith("galaxy") or otype in ("reflection_nebula", "dark_nebula"):
                moon_adj = -14.0

        fov_adj = 0.0
        size_arcmin = meta.get("size_major_arcmin")
        if fov_w_arcmin and size_arcmin:
            fill = float(size_arcmin) / fov_w_arcmin
            if 0.12 <= fill <= 1.05:
                fov_adj = 8.0
            elif fill > 1.8:
                fov_adj = -6.0

        mount_adj = 0.0
        if is_untracked_tripod and mag is not None:
            mount_adj = 10.0 if float(mag) <= 6.2 else (-10.0 if float(mag) > 8.0 else 0.0)

        cat_bonus = 6.0 if cat in ("Messier", "Caldwell", "Astrophoto") else 0.0
        total = (
            bright_score
            + alt_score
            + moon_adj
            + moon_glare_penalty
            + fov_adj
            + mount_adj
            + cat_bonus
        )
        scored.append((total, doc_id, otype))

    scored.sort(key=lambda item: item[0], reverse=True)

    # Diversify object types unless the user asked for a specific type
    selected: list[str] = []
    type_counts: dict[str, int] = {}
    has_type_filter = bool(intent and intent.object_type)
    for _, doc_id, otype in scored:
        if not has_type_filter and type_counts.get(otype, 0) >= 2:
            continue
        selected.append(doc_id)
        type_counts[otype] = type_counts.get(otype, 0) + 1
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for _, doc_id, _ in scored:
            if doc_id not in selected:
                selected.append(doc_id)
            if len(selected) >= limit:
                break

    return selected


def _select_seasonal_catalog_targets(
    obs_date: date,
    intent: Intent | None,
    limit: int = 4,
) -> list[str]:
    """Select bright seasonal corpus targets by midnight anti-solar Right Ascension when no location is given."""
    by_id, _ = load_documents_lookup()
    doy = obs_date.timetuple().tm_yday
    midnight_ra_deg = ((doy - 80) * 0.9856 + 180.0) % 360.0

    scored: list[tuple[float, str, str]] = []
    for doc_id, doc in by_id.items():
        if doc_id.startswith(("CON_", "GUIDE_")):
            continue
        if not _matches_intent_filters(doc, intent):
            continue
        meta = doc.get("metadata") or {}
        ra_deg, dec_deg, mag = meta.get("ra_deg"), meta.get("dec_deg"), meta.get("magnitude")
        if ra_deg is None or dec_deg is None or mag is None:
            continue
        if float(mag) > 9.0 and (intent is None or intent.max_magnitude is None):
            continue

        ra_diff = abs(((float(ra_deg) - midnight_ra_deg + 180.0) % 360.0) - 180.0)
        if ra_diff > 105.0 and (intent is None or not intent.constellation):
            continue

        otype = str(meta.get("object_type") or "deep_sky")
        cat = str(doc.get("catalog") or "")
        ra_score = (105.0 - min(ra_diff, 105.0)) * 0.25
        bright_score = (11.5 - float(mag)) * 5.5
        cat_bonus = 8.0 if cat == "Messier" else (4.0 if cat in ("Caldwell", "Astrophoto") else 0.0)
        scored.append((bright_score + ra_score + cat_bonus, doc_id, otype))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    type_counts: dict[str, int] = {}
    has_type_filter = bool(intent and intent.object_type)
    for _, doc_id, otype in scored:
        if not has_type_filter and type_counts.get(otype, 0) >= 2:
            continue
        selected.append(doc_id)
        type_counts[otype] = type_counts.get(otype, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def _night_minutes_to_time(night_mins: int) -> time:
    """Convert noon-anchored minutes `[0..1440)` back to a `datetime.time`."""
    total_mins = (night_mins + 720) % 1440
    return time(total_mins // 60, total_mins % 60)


def _build_plan_items(
    vis_data: dict[str, Any],
    dark_start: str,
    dark_end: str,
    retrieved_doc_ids: list[str],
    priority_ids: list[str],
    object_records: dict[str, dict[str, Any]] | None = None,
    intent: Intent | None = None,
    twilight_only: bool = False,
) -> list[PlanItem]:
    """Build deterministic `PlanItem` slots strictly inside `(dark_start, dark_end)` for both `start` and `end`."""
    objs_map = vis_data.get("objects") or {}
    if not objs_map:
        return []

    win_s_t = _parse_hhmm(dark_start) or time(21, 0)
    win_e_t = _parse_hhmm(dark_end) or time(5, 30)
    win_s_m = _time_to_night_minutes(win_s_t)
    win_e_m = _time_to_night_minutes(win_e_t)
    if win_e_m <= win_s_m:
        return []

    min_alt_floor = settings.min_observable_altitude_deg
    schedule_pool = list(dict.fromkeys(priority_ids + list(objs_map.keys())))
    plan_items: list[PlanItem] = []
    slot_idx = 0

    for oid in schedule_pool:
        if len(plan_items) >= 5 or oid not in retrieved_doc_ids:
            continue
        odata = objs_map.get(oid)
        if not odata or not odata.get("observable"):
            continue

        mag = odata.get("magnitude")
        nb_friendly = bool(odata.get("narrowband_friendly"))
        # During nautical white nights, restrict schedule to bright targets (mag <= 8.0) or narrowband targets
        if twilight_only:
            if mag is None and not nb_friendly:
                continue
            if mag is not None and float(mag) > 8.0 and not nb_friendly:
                continue

        valid_dark_slots: list[tuple[int, dict[str, Any]]] = []
        for s in odata.get("slots") or []:
            st_obj = _parse_hhmm(s["time"])
            if st_obj is None:
                continue
            sm_val = _time_to_night_minutes(st_obj)
            # Ensure at least a 15-minute slot fits before `win_e_m`
            if win_s_m <= sm_val <= (win_e_m - 15) and float(s["altitude_deg"]) >= min_alt_floor:
                valid_dark_slots.append((sm_val, s))

        # Fallback if window is very short (< 30 min) and a slot lands right inside [win_s_m, win_e_m]
        if not valid_dark_slots:
            for s in odata.get("slots") or []:
                st_obj = _parse_hhmm(s["time"])
                if st_obj is None:
                    continue
                sm_val = _time_to_night_minutes(st_obj)
                if win_s_m <= sm_val <= win_e_m and float(s["altitude_deg"]) >= min_alt_floor:
                    valid_dark_slots.append((sm_val, s))

        if not valid_dark_slots:
            continue

        chosen_sm, chosen = valid_dark_slots[min(slot_idx, len(valid_dark_slots) - 1)]
        end_sm = min(chosen_sm + 30, win_e_m)
        if end_sm <= chosen_sm:
            chosen_sm = max(win_s_m, win_e_m - 20)
            end_sm = win_e_m
        start_t = _night_minutes_to_time(chosen_sm)
        end_t = _night_minutes_to_time(end_sm)

        mag_val = float(mag) if mag is not None else 7.5
        diff: Literal["easy", "medium", "hard"] = (
            "easy" if mag_val <= 6.5 else ("medium" if mag_val <= 8.5 else "hard")
        )
        equip = (
            "Naked eye / Binoculars"
            if mag_val <= 6.2
            else ("Binoculars (10x50)" if mag_val <= 8.5 else "Small telescope")
        )
        size_arcmin = odata.get("size_major_arcmin")
        notes = f"{equip} · {size_arcmin}'" if size_arcmin else equip

        rec_obj = (object_records or {}).get(oid) or {}
        sg_cam = (rec_obj.get("shooting_guide") or {}).get("camera_settings") or {}
        if intent and (intent.focal_length_mm or intent.rig_summary) and sg_cam:
            focal_lbl = f"{int(sg_cam.get('focal_length_mm', 480))}mm"
            fill_pct = sg_cam.get("target_fill_pct")
            fill_lbl = f" ({fill_pct}% FOV)" if fill_pct is not None else ""
            sub_short = str(sg_cam.get("recommended_sub_exposure") or "").split("(")[0].strip()
            nb_tag = "Dual-Band Hα/OIII" if nb_friendly else "Broadband RGB"
            notes = f"{focal_lbl}{fill_lbl} · {sub_short} · {nb_tag}"

        plan_items.append(
            PlanItem(
                object_id=oid,
                name=odata.get("title", oid),
                start=start_t,
                end=end_t,
                altitude_deg=float(chosen["altitude_deg"]),
                direction=chosen.get("direction", "S"),
                difficulty=diff,
                imaging_notes=notes,
                sources=[oid],
            )
        )
        slot_idx += 1

    plan_items.sort(key=lambda p: _time_to_night_minutes(p.start))
    return plan_items
