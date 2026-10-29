"""AstroPilote workflow nodes (`guard`, `refuse`, `agent`, `tools`, `compose`, `validate`).

Natural-language understanding (intent, location, relative dates, target filters, optics, and
off-topic/injection detection) is handled by the structured LLM Router (`route_query_with_llm`).
All spherical astronomy, timezone/twilight math, weather aggregation, target culmination ranking,
and post-generation citation/altitude checks are 100% deterministic Python.
"""

from __future__ import annotations

import json
import math
import re
import time as time_mod
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from astropilote.agent.i18n import language_name, normalize_language, refusal_message
from astropilote.agent.schemas import (
    CATALOG_NAMES,
    AstroAnswer,
    Chunk,
    ChunkMetadata,
    EquipmentProfile,
    Intent,
    PlanItem,
    ToolCallRecord,
)
from astropilote.agent.state import AgentMessage, AstroState
from astropilote.agent.validator import _parse_hhmm, _time_to_night_minutes, verify_astro_answer
from astropilote.config import settings
from astropilote.ingest.catalog import IAU_CONSTELLATIONS
from astropilote.llm import generate_grounded_answer, route_query_with_llm
from astropilote.tools import TOOLS_BY_NAME
from astropilote.tools.catalog import (
    get_object,
    load_documents_lookup,
    normalize_object_id,
    resolve_object_record,
    resolve_sensor_specs,
)
from astropilote.tools.sky import _resolve_timezone_info, radec_to_altaz
from astropilote.tools.world import rank_observing_nights

# Standard astronomical catalog code syntax for corpus-membership verification
_EXPLICIT_CATALOG_CODE_PATTERNS = [
    r"\b(Messier\s*\d{1,4})\b",
    r"\b(Caldwell\s*\d{1,4})\b",
    r"\b(NGC\s*\d{1,4})\b",
    r"\b(IC\s*\d{1,4})\b",
    r"\b(Abell\s*\d{1,4})\b",
    r"\b(3C\s*\d{1,4})\b",
    r"\b(UGC\s*\d{1,5})\b",
    r"\b(PGC\s*\d{1,6})\b",
    r"\b(ESO\s*\d{1,3}-\d{1,3})\b",
    r"\b(K2-\d{1,4}[b-i])\b",
    r"\b(Kepler-\d{1,4}[b-i])\b",
    r"\b(TRAPPIST-\d[b-i])\b",
    r"\b(HD\s*\d{4,6}[b-i]?)\b",
    r"\b(\d{5,6}\s+[A-Z][a-z]{3,15})\b",
]

_CONSTELLATION_NAMES: dict[str, str] = {code: cname for code, cname, _, _, _ in IAU_CONSTELLATIONS}


# ---------------------------------------------------------------------------
# Pure Catalog & Spherical Astronomy Helpers
# ---------------------------------------------------------------------------


def _angular_separation_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Compute spherical angular separation in degrees between two J2000 coordinates."""
    r1, d1 = math.radians(ra1), math.radians(dec1)
    r2, d2 = math.radians(ra2), math.radians(dec2)
    cos_sep = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_sep))))


def _shares_complex_stem(doc_a: dict, doc_b: dict) -> bool:
    """Return True if two catalog entries share a multi-component nebula/cluster name stem (e.g. Veil Nebula)."""
    words_a = set(re.findall(r"[a-z]{4,}", (doc_a.get("title") or "").lower()))
    words_b = set(re.findall(r"[a-z]{4,}", (doc_b.get("title") or "").lower()))
    ignore = {"nebula", "galaxy", "cluster", "messier", "caldwell", "open", "globular", "planetary"}
    return bool((words_a & words_b) - ignore)


def _find_astrometric_companions(
    primary_id: str,
    max_sep_deg: float = 1.6,
    only_shared_complex: bool = False,
) -> list[str]:
    """Find physical companion or multi-component objects in the same constellation using J2000 coordinates."""
    by_id, _ = load_documents_lookup()
    primary = by_id.get(primary_id)
    if not primary:
        return []
    p_meta = primary.get("metadata", {})
    ra1, dec1 = p_meta.get("ra_deg"), p_meta.get("dec_deg")
    const1 = p_meta.get("constellation")
    if ra1 is None or dec1 is None or not const1:
        return []

    companions: list[tuple[float, str]] = []
    for doc_id, doc in by_id.items():
        if doc_id == primary_id or doc_id.startswith(("CON_", "GUIDE_")):
            continue
        if primary_id.startswith(("M", "C")) and not doc_id.startswith(("M", "C")):
            continue
        m = doc.get("metadata", {})
        if m.get("constellation") != const1:
            continue
        if only_shared_complex and not _shares_complex_stem(primary, doc):
            continue
        ra2, dec2 = m.get("ra_deg"), m.get("dec_deg")
        if ra2 is None or dec2 is None:
            continue
        sep = _angular_separation_deg(float(ra1), float(dec1), float(ra2), float(dec2))
        if sep <= max_sep_deg:
            companions.append((sep, doc_id))

    companions.sort(key=lambda x: x[0])
    return [cid for _, cid in companions[:2]]


def _resolve_corpus_ids(
    question: str,
    mentioned_targets: list[str] | None = None,
    wants_companions: bool = False,
) -> list[str]:
    """Resolve LLM-extracted targets and explicit catalog designations against the 567-doc corpus."""
    by_id, _ = load_documents_lookup()
    dso_ids: list[str] = []
    con_ids: list[str] = []
    guide_ids: list[str] = []

    def _register_doc(doc_id: str) -> None:
        if doc_id.startswith("GUIDE_"):
            guide_ids.append(doc_id)
        elif doc_id.startswith("CON_"):
            con_ids.append(doc_id)
        else:
            dso_ids.append(doc_id)

    for raw_target in mentioned_targets or []:
        rec = resolve_object_record(raw_target)
        if rec:
            _register_doc(str(rec["doc_id"]))

    for m in re.finditer(r"\b(?:Messier\s*|M)(\d{1,3})\b", question, flags=re.IGNORECASE):
        num = int(m.group(1))
        cand = f"M{num:03d}"
        if 1 <= num <= 110 and cand in by_id:
            dso_ids.append(cand)
    for m in re.finditer(r"\b(?:Caldwell\s*|C)(\d{1,3})\b", question, flags=re.IGNORECASE):
        num = int(m.group(1))
        cand = f"C{num:03d}"
        if 1 <= num <= 109 and cand in by_id:
            dso_ids.append(cand)
    for m in re.finditer(r"\bNGC\s*0*(\d{1,4})\b", question, flags=re.IGNORECASE):
        rec = resolve_object_record(f"NGC{int(m.group(1)):04d}")
        if rec:
            _register_doc(str(rec["doc_id"]))
    for m in re.finditer(r"\bIC\s*0*(\d{1,4})\b", question, flags=re.IGNORECASE):
        rec = resolve_object_record(f"IC{int(m.group(1)):04d}")
        if rec:
            _register_doc(str(rec["doc_id"]))

    q_low = question.lower()
    for doc_id, doc in by_id.items():
        for alias in doc.get("metadata", {}).get("aliases", []):
            a_low = alias.strip().lower()
            if len(a_low) >= 5 and not a_low.startswith(
                ("m0", "c0", "ngc", "ic0", "con_", "guide_")
            ):
                if re.search(rf"\b{re.escape(a_low)}s?\b", q_low):
                    _register_doc(doc_id)
                    break

    base_dsos = list(dict.fromkeys(dso_ids))
    for base_id in base_dsos:
        dso_ids.extend(
            _find_astrometric_companions(
                base_id,
                max_sep_deg=1.8 if len(base_dsos) <= 2 else 1.6,
                only_shared_complex=False if len(base_dsos) <= 2 else (not wants_companions),
            )
        )

    dso_ids = list(dict.fromkeys(dso_ids))
    host_con_ids: list[str] = []
    for did in dso_ids:
        const_code = (by_id.get(did, {}).get("metadata", {}).get("constellation") or "").upper()
        con_doc_id = f"CON_{const_code}"
        if const_code and con_doc_id in by_id:
            host_con_ids.append(con_doc_id)

    return list(dict.fromkeys(dso_ids + host_con_ids + con_ids + guide_ids))


def _find_unindexed_designations(question: str) -> list[str]:
    """Return any explicit astronomical catalog designations in `question` absent from the 567-doc corpus."""
    unindexed: list[str] = []
    for pat in _EXPLICIT_CATALOG_CODE_PATTERNS:
        for m in re.finditer(pat, question, flags=re.IGNORECASE):
            raw_code = m.group(1).strip()
            if resolve_object_record(raw_code) is None:
                unindexed.append(raw_code)
    return list(dict.fromkeys(unindexed))


def _check_out_of_corpus(
    question: str, *, strict: bool | None = None, language: str = "en"
) -> str | None:
    """Check bounded catalog limits (`Messier 1..110`, `Caldwell 1..109`) and optional strict corpus membership."""
    for m in re.finditer(r"\b(?:Messier|M)\s*(\d{1,4})\b", question, flags=re.IGNORECASE):
        num = int(m.group(1))
        if num < 1 or num > 110:
            return refusal_message("messier_out_of_range", language, num=num)

    for m in re.finditer(r"\b(?:Caldwell|C)\s*(\d{1,4})\b", question, flags=re.IGNORECASE):
        num = int(m.group(1))
        if num < 1 or num > 109:
            return refusal_message("caldwell_out_of_range", language, num=num)

    enforce_strict = settings.strict_corpus_guard if strict is None else strict
    if not enforce_strict:
        return None

    in_corpus_dsos = [
        oid for oid in _resolve_corpus_ids(question) if not oid.startswith(("CON_", "GUIDE_"))
    ]
    if in_corpus_dsos:
        return None

    unindexed = _find_unindexed_designations(question)
    if unindexed:
        return refusal_message("not_indexed", language, code=unindexed[0])
    return None


def _matches_intent_filters(doc: dict, intent: Intent | None) -> bool:
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
    """Generate `n_samples` UTC timestamps evenly spanning the local astronomical darkness window."""
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


def _extract_equipment_profile(state: AstroState) -> EquipmentProfile | None:
    """Coerce `state.get('equipment_profile')` into an active `EquipmentProfile` or return None."""
    raw = state.get("equipment_profile")
    if raw is None:
        return None
    if isinstance(raw, EquipmentProfile):
        return raw if raw.enabled else None
    if isinstance(raw, dict):
        try:
            prof = EquipmentProfile(**raw)
            return prof if prof.enabled else None
        except (TypeError, ValueError):
            return None
    return None


def _format_rig_summary(prof: EquipmentProfile) -> str:
    """Format a concise human-readable summary of `prof`."""
    f_ratio = round(prof.focal_length_mm / prof.aperture_mm, 1) if prof.aperture_mm else None
    f_str = f" (f/{f_ratio})" if f_ratio else ""
    optic_lbl = (
        prof.telescope_or_lens
        or prof.rig_name
        or f"{prof.aperture_mm:.0f}/{prof.focal_length_mm:.0f}mm"
    )
    cam_lbl = prof.camera_model or prof.sensor_format
    mount_map = {
        "equatorial_goto": "Equatorial GoTo mount",
        "star_tracker": "Portable Star Tracker",
        "altaz_goto": "Alt-Az GoTo mount",
        "tripod": "Fixed Tripod (untracked)",
    }
    mount_lbl = mount_map.get(prof.mount_type, prof.mount_type)
    if prof.has_autoguiding and prof.mount_type != "tripod":
        mount_lbl += " + Autoguiding"
    filt_lbl = f" · Filters: {', '.join(prof.filters)}" if prof.filters else ""
    bortle_lbl = f" · Sky Bortle {prof.bortle_class}" if prof.bortle_class else ""
    return (
        f"{optic_lbl} · {prof.focal_length_mm:.0f}mm{f_str} · "
        f"{cam_lbl} ({prof.sensor_format}) · {mount_lbl}{filt_lbl}{bortle_lbl}"
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

    Works across both Northern and Southern hemispheres, any date of the year, and adapts to
    lunar illumination, optical focal length, sensor format, mount tracking, and user metadata filters.
    """
    by_id, _ = load_documents_lookup()
    utc_samples = _dark_window_utc_samples(lat, lon, obs_date_str, dark_start, dark_end)
    min_alt = max(settings.min_observable_altitude_deg, 25.0)
    is_visual_only = bool(intent and intent.mode == "visual")
    focal_mm = intent.focal_length_mm if intent else None
    sensor_w, _, _, _ = resolve_sensor_specs(intent.sensor_format if intent else "APS-C")
    fov_w_arcmin = (57.2958 * 60.0 * sensor_w / focal_mm) if (focal_mm and focal_mm > 0) else None
    is_untracked_tripod = bool(intent and intent.mount_type == "tripod")
    user_filters = {str(f).lower() for f in (intent.filters if intent else [])}
    has_nb_filter = any(
        k in user_filters
        for k in ("dual_band", "dual-band", "narrowband_sho", "uhc_cls", "ha_oiii")
    )

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

        alts = [
            radec_to_altaz(float(ra_deg), float(dec_deg), lat, lon, dt_u)[0] for dt_u in utc_samples
        ]
        peak_alt = max(alts)
        if peak_alt < min_alt:
            continue

        otype = str(meta.get("object_type") or "deep_sky")
        nb_ok = bool(meta.get("narrowband_friendly"))
        cat = str(doc.get("catalog") or "")

        bright_score = (12.5 - float(mag)) * 5.2 if mag is not None else 22.0
        alt_score = min(peak_alt, 82.0) * 0.45

        moon_adj = 0.0
        if moon_pct >= 65 and (intent is None or not intent.object_type):
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
        total = bright_score + alt_score + moon_adj + fov_adj + mount_adj + cat_bonus
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


# ---------------------------------------------------------------------------
# Graph Nodes (`guard`, `refuse`, `agent`, `tools`, `compose`, `validate`)
# ---------------------------------------------------------------------------


def guard_node(state: AstroState) -> dict:
    """Node 1 (`guard` / Comprendre): Route the query via the structured LLM Router and verify date/horizon bounds."""
    q = state["question"]

    ref_today = settings.get_reference_date()
    if state.get("date_override"):
        try:
            override_d = date.fromisoformat(str(state["date_override"]).strip()[:10])
        except ValueError:
            override_d = None
    else:
        override_d = None

    anchor_date = override_d or ref_today
    routed, extra_in, extra_out = route_query_with_llm(
        question=q,
        ref_date_iso=anchor_date.isoformat(),
        ref_weekday=anchor_date.strftime("%A"),
        mode_override=state.get("llm_mode_override"),
    )

    req_type = str(routed.get("request_type") or "factual")
    lang = normalize_language(routed.get("language"))
    if req_type in ("prompt_injection", "off_topic"):
        intent = Intent(
            request_type=req_type,  # type: ignore[arg-type]
            language=lang,
            refusal_reason=str(routed.get("refusal_reason") or refusal_message(req_type, lang)),
        )
        return {
            "intent": intent,
            "tokens_in": state.get("tokens_in", 0) + extra_in,
            "tokens_out": state.get("tokens_out", 0) + extra_out,
        }

    # Parse extracted start_date
    parsed_date: date | None = override_d
    raw_start = routed.get("start_date")
    if isinstance(raw_start, str) and raw_start.strip():
        try:
            parsed_date = date.fromisoformat(raw_start.strip()[:10])
        except ValueError:
            parsed_date = override_d

    raw_days = int(routed.get("forecast_days") or 1)
    weather_only = bool(routed.get("weather_only"))
    mentioned_targets = [
        str(t).strip() for t in (routed.get("mentioned_targets") or []) if str(t).strip()
    ]
    resolved_dsos = [
        oid
        for oid in _resolve_corpus_ids(q, mentioned_targets)
        if not oid.startswith(("CON_", "GUIDE_"))
    ]

    # Deterministic meteorological horizon check (> MAX_FORECAST_DAYS) for weather-only queries
    if parsed_date is not None:
        delta_days = (parsed_date - ref_today).days
        if delta_days > settings.max_forecast_days and (weather_only and not resolved_dsos):
            intent = Intent(
                request_type="impossible_forecast",
                language=lang,
                target_date=parsed_date,
                location_name=routed.get("location_name") or None,
                weather_only=True,
                refusal_reason=refusal_message(
                    "impossible_forecast",
                    lang,
                    date=parsed_date.isoformat(),
                    delta_days=delta_days,
                    max_days=settings.max_forecast_days,
                ),
            )
            return {
                "intent": intent,
                "tokens_in": state.get("tokens_in", 0) + extra_in,
                "tokens_out": state.get("tokens_out", 0) + extra_out,
            }

    raw_loc = routed.get("location_name")
    clean_loc = str(raw_loc).strip() if isinstance(raw_loc, str) and raw_loc.strip() else None

    cat_filter = routed.get("catalog_filter")
    valid_cat = cat_filter if cat_filter in CATALOG_NAMES else None

    raw_const = routed.get("constellation")
    clean_const = (
        str(raw_const).strip().upper()
        if isinstance(raw_const, str) and str(raw_const).strip().upper() in _CONSTELLATION_NAMES
        else None
    )

    prof = _extract_equipment_profile(state)

    aperture = (
        float(routed["aperture_mm"])
        if routed.get("aperture_mm") is not None
        else (
            prof.aperture_mm
            if prof is not None
            else (
                state["aperture_override"] if state.get("aperture_override") is not None else None
            )
        )
    )
    focal = (
        float(routed["focal_length_mm"])
        if routed.get("focal_length_mm") is not None
        else (
            prof.focal_length_mm
            if prof is not None
            else (
                state["focal_length_override"]
                if state.get("focal_length_override") is not None
                else None
            )
        )
    )
    if routed.get("sensor_format") == "full-frame":
        sensor_fmt = "full-frame"
    elif prof is not None:
        sensor_fmt = prof.sensor_format
    else:
        sensor_fmt = "APS-C"

    mount_val = prof.mount_type if prof is not None else "equatorial_goto"
    autoguide_val = bool(prof.has_autoguiding) if prof is not None else False
    filters_val = list(prof.filters) if prof is not None else []
    rig_summary_val = _format_rig_summary(prof) if prof is not None else None

    mode_val = (
        routed.get("mode")
        if routed.get("mode") in ("visual", "astrophotography", "hybrid")
        else ("astrophotography" if prof is not None else "hybrid")
    )

    if req_type == "session_plan" and parsed_date is None:
        parsed_date = anchor_date

    # Concatenate companion hint into mentioned_targets via state intent
    resolved_with_companions = _resolve_corpus_ids(
        q,
        mentioned_targets,
        wants_companions=bool(routed.get("wants_companions")),
    )
    merged_targets = list(dict.fromkeys(mentioned_targets + resolved_with_companions))

    intent = Intent(
        request_type="session_plan" if req_type == "session_plan" else "factual",
        language=lang,
        mode=mode_val,  # type: ignore[arg-type]
        is_fictional_location=bool(routed.get("is_fictional_location")),
        location_name=clean_loc,
        target_date=parsed_date,
        forecast_days=max(1, min(raw_days, settings.max_forecast_days)),
        weather_only=weather_only,
        mentioned_targets=merged_targets,
        include_practical_guides=bool(routed.get("include_practical_guides")),
        search_query=str(routed["search_query"]).strip() if routed.get("search_query") else None,
        constellation=clean_const,
        catalog_filter=valid_cat,  # type: ignore[arg-type]
        object_type=str(routed["object_type"]) if routed.get("object_type") else None,
        max_magnitude=float(routed["max_magnitude"])
        if routed.get("max_magnitude") is not None
        else None,
        narrowband_only=bool(routed.get("narrowband_only")),
        aperture_mm=aperture,
        focal_length_mm=focal,
        sensor_format=sensor_fmt,  # type: ignore[arg-type]
        mount_type=mount_val,
        has_autoguiding=autoguide_val,
        filters=filters_val,
        rig_summary=rig_summary_val,
    )
    return {
        "intent": intent,
        "tokens_in": state.get("tokens_in", 0) + extra_in,
        "tokens_out": state.get("tokens_out", 0) + extra_out,
    }


def refuse_node(state: AstroState) -> dict:
    """Node (`refuse` / Refuser): Return a polite, source-free refusal without calling tools."""
    intent = state.get("intent")
    reason = (
        intent.refusal_reason
        if intent and intent.refusal_reason
        else refusal_message("generic_refusal", intent.language if intent else "en")
    )
    ans = AstroAnswer(
        answer=reason,
        plan=[],
        conditions=None,
        sources=[],
        abstained=True,
        verified=True,
    )
    return {"answer": ans}


def _extract_geocode_coords(messages: list[AgentMessage]) -> tuple[float, float, bool]:
    """Read `(lat, lon, geocode_found)` from prior `geocode` tool messages."""
    for msg in messages:
        if msg.role == "tool" and msg.name == "geocode":
            try:
                data = json.loads(msg.content)
                if data.get("found"):
                    return float(data["lat"]), float(data["lon"]), True
                return 0.0, 0.0, False
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                return 0.0, 0.0, False
    return 0.0, 0.0, False


def agent_node(state: AstroState) -> dict:
    """Node 2 (`agent` / Planifier): Deterministically orchestrate tool calls from the structured `Intent`."""
    steps = state.get("steps", 0)
    q = state["question"]
    intent = state.get("intent")
    exec_mode = state.get("execution_mode") or "agent"

    enforce_strict = (
        settings.strict_corpus_guard
        if state.get("strict_corpus_guard") is None
        else bool(state.get("strict_corpus_guard"))
    )

    # Short-circuit on non-existent catalog numbers (e.g. Messier 111) or fictional locations (e.g. Atlantis)
    if _check_out_of_corpus(q, strict=enforce_strict) is not None or (
        intent is not None and intent.is_fictional_location
    ):
        ai_msg = AgentMessage(role="assistant", content="", tool_calls=[])
        return {"messages": state.get("messages", []) + [ai_msg], "steps": steps + 1}

    tool_calls: list[dict[str, Any]] = []
    retrieval_query = (intent.search_query if intent and intent.search_query else q).strip()
    focal_mm = intent.focal_length_mm if intent else None
    aperture_mm = intent.aperture_mm if intent else None
    sensor_fmt = intent.sensor_format if intent else "APS-C"
    mount_type = intent.mount_type if intent else "equatorial_goto"
    has_autoguiding = bool(intent.has_autoguiding) if intent else False
    filters_list = list(intent.filters) if intent else []

    def _get_object_args(oid: str) -> dict[str, Any]:
        args: dict[str, Any] = {
            "object_id": oid,
            "focal_length_mm": focal_mm,
            "sensor_format": sensor_fmt,
        }
        if aperture_mm is not None:
            args["aperture_mm"] = aperture_mm
        if mount_type != "equatorial_goto" or has_autoguiding or filters_list:
            args["mount_type"] = mount_type
            args["has_autoguiding"] = has_autoguiding
            args["filters"] = filters_list
        return args

    is_located_session = bool(
        exec_mode != "rag"
        and intent is not None
        and intent.request_type == "session_plan"
        and intent.location_name
    )

    if steps == 0:
        resolved_ids = _resolve_corpus_ids(
            q,
            intent.mentioned_targets if intent else None,
        )

        if exec_mode == "rag":
            tool_calls.append(
                {
                    "name": "search_catalog",
                    "args": {"query": retrieval_query, "top_k": settings.final_top_k},
                    "id": "call_rag_search",
                }
            )
        else:
            if is_located_session and intent and intent.location_name:
                tool_calls.append(
                    {
                        "name": "geocode",
                        "args": {"place_name": intent.location_name},
                        "id": "call_geocode",
                    }
                )

            non_guide_ids = [oid for oid in resolved_ids if not oid.startswith("GUIDE_")]
            dso_only_ids = [oid for oid in non_guide_ids if not oid.startswith("CON_")]
            # When no location is provided on a general target/session query (or <4 targets), select seasonal catalog targets
            if (
                not is_located_session
                and len(dso_only_ids) < 4
                and intent
                and (
                    intent.request_type == "session_plan"
                    or intent.object_type
                    or intent.max_magnitude
                    or intent.constellation
                    or intent.catalog_filter
                    or intent.narrowband_only
                )
            ):
                obs_d = intent.target_date or settings.get_reference_date()
                seasonal_ids = _select_seasonal_catalog_targets(obs_d, intent, limit=5)
                non_guide_ids = list(dict.fromkeys(non_guide_ids + seasonal_ids))

            for oid in non_guide_ids[:6]:
                tool_calls.append(
                    {
                        "name": "get_object",
                        "args": _get_object_args(oid),
                        "id": f"call_get_{oid}",
                    }
                )

            search_args: dict[str, Any] = {
                "query": retrieval_query,
                "top_k": 6,
            }
            if intent:
                if intent.catalog_filter:
                    search_args["catalog"] = intent.catalog_filter
                if intent.object_type:
                    search_args["object_type"] = intent.object_type
                if intent.constellation:
                    search_args["constellation"] = intent.constellation
                if intent.max_magnitude is not None:
                    search_args["max_magnitude"] = intent.max_magnitude
                if intent.narrowband_only:
                    search_args["narrowband_only"] = True
                if non_guide_ids and not intent.include_practical_guides:
                    search_args["exclude_guides"] = True

            tool_calls.append(
                {
                    "name": "search_catalog",
                    "args": search_args,
                    "id": "call_search",
                }
            )

            if intent and (intent.include_practical_guides or is_located_session):
                tool_calls.append(
                    {
                        "name": "search_catalog",
                        "args": {"query": retrieval_query, "catalog": "Guide", "top_k": 3},
                        "id": "call_search_guides",
                    }
                )

    elif steps == 1 and is_located_session and intent is not None:
        lat, lon, geocode_ok = _extract_geocode_coords(state.get("messages", []))
        if geocode_ok:
            obs_date_str = (intent.target_date or settings.get_reference_date()).isoformat()
            span = max(1, min(intent.forecast_days, settings.max_forecast_days))
            tool_calls.extend(
                [
                    {
                        "name": "night",
                        "args": {"lat": lat, "lon": lon, "obs_date": obs_date_str, "days": span},
                        "id": "call_night",
                    },
                    {
                        "name": "weather",
                        "args": {"lat": lat, "lon": lon, "obs_date": obs_date_str, "days": span},
                        "id": "call_weather",
                    },
                ]
            )

    elif steps == 2 and is_located_session and intent is not None:
        lat, lon, geocode_ok = _extract_geocode_coords(state.get("messages", []))
        if geocode_ok:
            daily_nights: list[dict] = []
            daily_forecasts: list[dict] = []
            for msg in state.get("messages", []):
                if msg.role == "tool":
                    try:
                        payload = json.loads(msg.content)
                        if msg.name == "night":
                            daily_nights = list(payload.get("daily_nights") or [payload])
                        elif msg.name == "weather":
                            daily_forecasts = list(payload.get("daily_forecasts") or [payload])
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass

            best_date = (intent.target_date or settings.get_reference_date()).isoformat()
            dark_start = "21:00"
            dark_end = "05:00"
            moon_pct = 50

            if len(daily_nights) > 1:
                ranked = rank_observing_nights(daily_nights, daily_forecasts)
                if ranked:
                    best_date = str(ranked[0]["date"])
                    dark_start = str(ranked[0].get("astronomical_dark_start") or "21:00")
                    dark_end = str(ranked[0].get("astronomical_dark_end") or "05:00")
                    moon_pct = int(ranked[0].get("moon_illumination_pct") or 50)
            elif daily_nights:
                best_date = str(daily_nights[0].get("date") or best_date)
                dark_start = str(daily_nights[0].get("astronomical_dark_start") or "21:00")
                dark_end = str(daily_nights[0].get("astronomical_dark_end") or "05:00")
                moon_pct = int(daily_nights[0].get("moon_illumination_pct") or 50)

            explicit_dsos = [
                oid
                for oid in _resolve_corpus_ids(q, intent.mentioned_targets)
                if not oid.startswith(("CON_", "GUIDE_"))
            ]
            if len(explicit_dsos) >= 5:
                candidate_ids = explicit_dsos[:6]
            else:
                dyn_ids = _select_dynamic_sky_targets(
                    lat=lat,
                    lon=lon,
                    obs_date_str=best_date,
                    dark_start=dark_start,
                    dark_end=dark_end,
                    moon_pct=moon_pct,
                    intent=intent,
                    limit=6,
                )
                candidate_ids = list(dict.fromkeys(explicit_dsos + dyn_ids))[:6]

            existing_doc_ids = {c.doc_id for c in state.get("retrieved", [])}
            for oid in candidate_ids:
                if oid not in existing_doc_ids:
                    tool_calls.append(
                        {
                            "name": "get_object",
                            "args": _get_object_args(oid),
                            "id": f"call_get_dyn_{oid}",
                        }
                    )

            if candidate_ids:
                tool_calls.append(
                    {
                        "name": "visibility",
                        "args": {
                            "object_ids": candidate_ids,
                            "lat": lat,
                            "lon": lon,
                            "obs_date": best_date,
                            "focal_length_mm": focal_mm,
                        },
                        "id": "call_visibility",
                    }
                )

    ai_msg = AgentMessage(role="assistant", content="", tool_calls=tool_calls)
    return {
        "messages": state.get("messages", []) + [ai_msg],
        "steps": steps + 1,
    }


def tools_node(state: AstroState) -> dict:
    """Node 3 (`tools` / Outils): Execute requested deterministic tool calls and record telemetry."""
    messages = state.get("messages", [])
    if not messages:
        return {}
    last_msg = messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", []) or []

    new_tool_messages: list[AgentMessage] = []
    trace: list[ToolCallRecord] = list(state.get("tool_trace", []))
    retrieved: list[Chunk] = list(state.get("retrieved", []))
    existing_chunk_ids = {c.chunk_id for c in retrieved}
    existing_doc_ids = {c.doc_id for c in retrieved}
    calc_alts: dict[str, list[float]] = dict(state.get("calculated_altitudes", {}))
    night_win = state.get("night_window")

    for tc in tool_calls:
        t_name = tc["name"]
        t_args = tc.get("args", {})
        t_id = tc.get("id", t_name)
        tool_fn = TOOLS_BY_NAME.get(t_name)
        if tool_fn is None:
            continue

        t0 = time_mod.perf_counter()
        result = tool_fn(**t_args)
        dur = round(time_mod.perf_counter() - t0, 4)

        trace.append(ToolCallRecord(name=t_name, arguments=t_args, duration_s=dur))
        new_tool_messages.append(
            AgentMessage(
                role="tool",
                content=json.dumps(result, ensure_ascii=False),
                name=t_name,
                tool_call_id=t_id,
            )
        )

        if t_name == "search_catalog" and isinstance(result, list):
            for item in result:
                ch = Chunk(
                    chunk_id=item["chunk_id"],
                    doc_id=item["doc_id"],
                    title=item["title"],
                    section=item["section"],
                    text=item["text"],
                    score=item.get("score", 0.0),
                    metadata=ChunkMetadata(**item["metadata"]),
                )
                if ch.chunk_id not in existing_chunk_ids:
                    retrieved.append(ch)
                    existing_chunk_ids.add(ch.chunk_id)
                    existing_doc_ids.add(ch.doc_id)

        elif t_name == "get_object" and isinstance(result, dict) and result.get("found"):
            doc_id = result["doc_id"]
            if doc_id not in existing_doc_ids:
                meta = ChunkMetadata(
                    doc_id=doc_id,
                    catalog=result["catalog"],
                    object_type=result["object_type"],
                    constellation=result["constellation"],
                    ra_deg=result.get("ra_deg"),
                    dec_deg=result.get("dec_deg"),
                    magnitude=result.get("magnitude"),
                    surface_brightness=result.get("surface_brightness"),
                    size_major_arcmin=result.get("size_major_arcmin"),
                    size_minor_arcmin=result.get("size_minor_arcmin"),
                    narrowband_friendly=bool(result.get("narrowband_friendly", False)),
                    aliases=list(result.get("aliases", [])),
                )
                framing_str = f" {result['framing_advice']}" if result.get("framing_advice") else ""
                summary_text = (
                    f"[{doc_id} | {result['title']} | Type: {meta.object_type} | "
                    f"Constellation: {meta.constellation} | Mag: {meta.magnitude} | "
                    f"Size: {meta.size_major_arcmin}' | Narrowband: {meta.narrowband_friendly}]"
                    f"{framing_str}"
                )
                retrieved.append(
                    Chunk(
                        chunk_id=f"{doc_id}#catalog-0",
                        doc_id=doc_id,
                        title=result["title"],
                        section="Catalog Metadata",
                        text=summary_text,
                        score=1.0,
                        metadata=meta,
                    )
                )
                existing_doc_ids.add(doc_id)

        elif t_name == "night" and isinstance(result, dict):
            night_win = (result["astronomical_dark_start"], result["astronomical_dark_end"])

        elif t_name == "visibility" and isinstance(result, dict):
            nw = result.get("night_window") or {}
            if nw.get("dark_start") and nw.get("dark_end"):
                night_win = (str(nw["dark_start"]), str(nw["dark_end"]))
            objs = result.get("objects", {})
            for oid, odata in objs.items():
                if odata.get("slots"):
                    calc_alts[oid] = [float(s["altitude_deg"]) for s in odata["slots"]]

    return {
        "messages": messages + new_tool_messages,
        "tool_trace": trace,
        "retrieved": retrieved,
        "calculated_altitudes": calc_alts,
        "night_window": night_win,
    }


def _build_plan_items(
    vis_data: dict,
    dark_start: str,
    dark_end: str,
    retrieved_doc_ids: list[str],
    priority_ids: list[str],
    object_records: dict[str, dict[str, Any]] | None = None,
    intent: Intent | None = None,
) -> list[PlanItem]:
    """Build deterministic `PlanItem` slots strictly inside astronomical darkness and above `min_observable_altitude_deg`."""
    objs_map = vis_data.get("objects") or {}
    if not objs_map:
        return []

    win_s_t = _parse_hhmm(dark_start) or time(21, 0)
    win_e_t = _parse_hhmm(dark_end) or time(5, 30)
    win_s_m = _time_to_night_minutes(win_s_t)
    win_e_m = _time_to_night_minutes(win_e_t)
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
        valid_dark_slots = []
        for s in odata.get("slots") or []:
            st_obj = _parse_hhmm(s["time"])
            if st_obj is None:
                continue
            sm_val = _time_to_night_minutes(st_obj)
            if win_s_m <= sm_val <= win_e_m and float(s["altitude_deg"]) >= min_alt_floor:
                valid_dark_slots.append(s)

        if not valid_dark_slots:
            continue

        chosen = valid_dark_slots[min(slot_idx, len(valid_dark_slots) - 1)]
        sh, sm = [int(x) for x in chosen["time"].split(":")]
        start_t = time(sh, sm)
        eh = (sh + (1 if sm + 30 >= 60 else 0)) % 24
        em = (sm + 30) % 60
        end_t = time(eh, em)

        mag = odata.get("magnitude")
        mag_val = float(mag) if mag is not None else 7.5
        diff = "easy" if mag_val <= 6.5 else ("medium" if mag_val <= 8.5 else "hard")
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
            nb_tag = "Dual-Band Hα/OIII" if odata.get("narrowband_friendly") else "Broadband RGB"
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


def compose_node(state: AstroState) -> dict:
    """Node 4 (`compose` / Rédiger): Pass clean structured tool JSON + RAG passages to the LLM."""
    q = state["question"]
    retrieved = state.get("retrieved", [])
    intent = state.get("intent")
    lang = intent.language if intent else "en"

    enforce_strict = (
        settings.strict_corpus_guard
        if state.get("strict_corpus_guard") is None
        else bool(state.get("strict_corpus_guard"))
    )
    ooc_reason = _check_out_of_corpus(q, strict=enforce_strict, language=lang)
    if ooc_reason is not None or (enforce_strict and not retrieved):
        return {
            "answer": AstroAnswer(
                answer=ooc_reason or refusal_message("no_documents", lang),
                plan=[],
                conditions=None,
                sources=[],
                abstained=True,
                verified=True,
            )
        }

    if intent and intent.is_fictional_location:
        bad_place = intent.location_name or refusal_message("unknown_place", lang)
        return {
            "answer": AstroAnswer(
                answer=refusal_message("fictional_location", lang, place=bad_place),
                plan=[],
                conditions=None,
                sources=[],
                abstained=True,
                verified=True,
            )
        }

    night_data: dict[str, Any] = {}
    vis_data: dict[str, Any] = {}
    weather_data: dict[str, Any] = {}
    geocode_data: dict[str, Any] = {}
    object_records: dict[str, dict[str, Any]] = {}

    for msg in state.get("messages", []):
        if msg.role == "tool":
            try:
                payload = json.loads(msg.content)
                if msg.name == "night" and isinstance(payload, dict):
                    night_data = payload
                elif msg.name == "visibility" and isinstance(payload, dict):
                    vis_data = payload
                elif msg.name == "weather" and isinstance(payload, dict):
                    weather_data = payload
                elif msg.name == "geocode" and isinstance(payload, dict):
                    geocode_data = payload
                elif (
                    msg.name == "get_object" and isinstance(payload, dict) and payload.get("found")
                ):
                    object_records[str(payload["doc_id"])] = payload
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass

    if geocode_data and geocode_data.get("found") is False:
        bad_loc = (
            (intent.location_name if intent else None)
            or geocode_data.get("place_name")
            or refusal_message("unknown_place", lang)
        )
        return {
            "answer": AstroAnswer(
                answer=refusal_message("geocode_failed", lang, place=bad_loc),
                plan=[],
                conditions=None,
                sources=[],
                abstained=True,
                verified=True,
            )
        }

    is_session_plan = bool(intent and intent.request_type == "session_plan")
    has_real_location = bool(geocode_data.get("found") and intent and intent.location_name)

    daily_nights = list(night_data.get("daily_nights") or ([night_data] if night_data else []))
    daily_forecasts = list(
        weather_data.get("daily_forecasts") or ([weather_data] if weather_data else [])
    )
    is_multi_night = len(daily_nights) > 1
    ranked_nights = rank_observing_nights(daily_nights, daily_forecasts) if is_multi_night else []
    best_night = ranked_nights[0] if ranked_nights else None

    active_night = (
        next((n for n in daily_nights if n.get("date") == best_night["date"]), daily_nights[0])
        if (best_night and daily_nights)
        else night_data
    )
    active_weather = (
        next(
            (w for w in daily_forecasts if w.get("date") == best_night["date"]),
            daily_forecasts[0] if daily_forecasts else {},
        )
        if best_night
        else weather_data
    )

    retrieved_doc_ids = list(dict.fromkeys(c.doc_id for c in retrieved))
    resolved_mentions = _resolve_corpus_ids(q, intent.mentioned_targets if intent else None)
    priority_dsos = [
        d
        for d in resolved_mentions
        if d in retrieved_doc_ids and not d.startswith(("CON_", "GUIDE_"))
    ]

    plan_items: list[PlanItem] = []
    conditions_str: str | None = None

    if is_session_plan and has_real_location and active_night:
        site_label = str(
            intent.location_name
            if intent and intent.location_name
            else geocode_data.get("short_name", "")
        )
        dark_start = str(active_night.get("astronomical_dark_start") or "21:00")
        dark_end = str(active_night.get("astronomical_dark_end") or "05:30")
        tz_lbl = str(active_night.get("tz_label") or "local")
        moon_pct = int(active_night.get("moon_illumination_pct") or 0)
        moon_phase = str(active_night.get("moon_phase_name") or "")
        w_avail = bool(active_weather.get("available")) if active_weather else False
        cloud_note = (
            f"Cloud cover {active_weather.get('cloud_cover_pct')}%"
            if w_avail
            else f"Weather unavailable (> {settings.max_forecast_days} days forecast horizon)"
        )
        moon_note = f"{moon_pct}% {moon_phase}".strip()

        if is_multi_night and best_night:
            conditions_str = (
                f"{site_label} ({daily_nights[0]['date']} to {daily_nights[-1]['date']}): "
                f"Best night {best_night['weekday']} {best_night['date']} "
                f"({cloud_note}, dark {dark_start}–{dark_end} {tz_lbl}, Moon {moon_note})"
            )
        else:
            date_lbl = str(active_night.get("date") or "")
            conditions_str = (
                f"{site_label} on {date_lbl}: Astronomical darkness {dark_start} to {dark_end} ({tz_lbl}) | "
                f"Moon {moon_note} | {cloud_note}"
            )

        plan_items = _build_plan_items(
            vis_data=vis_data,
            dark_start=dark_start,
            dark_end=dark_end,
            retrieved_doc_ids=retrieved_doc_ids,
            priority_ids=priority_dsos + list(object_records.keys()),
            object_records=object_records,
            intent=intent,
        )

    # Assemble candidate corpus citation IDs (retrieved IDs only)
    scheduled_ids = [pi.object_id for pi in plan_items]
    con_ids = [d for d in resolved_mentions if d in retrieved_doc_ids and d.startswith("CON_")]
    guide_ids = [d for d in retrieved_doc_ids if d.startswith("GUIDE_")][:2]
    candidate_cite_ids = list(
        dict.fromkeys(
            priority_dsos
            + scheduled_ids
            + list(object_records.keys())
            + con_ids
            + guide_ids
            + retrieved_doc_ids
        )
    )[:8]

    # Build clean structured JSON payload of deterministic tool outputs for the LLM
    location_payload: dict[str, Any]
    if has_real_location:
        location_payload = {
            "provided": True,
            "requested_name": intent.location_name if intent else None,
            "resolved_place_name": geocode_data.get("place_name"),
            "lat": geocode_data.get("lat"),
            "lon": geocode_data.get("lon"),
        }
    else:
        location_payload = {
            "provided": False,
            "note": (
                "No geographic location was specified by the user. Do NOT assume or mention any default city "
                "(never mention Fontainebleau or Paris). If local sky/weather/timing matters for this question, "
                "present seasonal/catalog target recommendations and invite the user to share their city, region, "
                "or GPS coordinates for exact local cloud cover, darkness hours, and target altitudes."
            ),
        }

    unobservable_targets: list[dict[str, Any]] = []
    for oid, odata in (vis_data.get("objects") or {}).items():
        if not odata.get("observable"):
            unobservable_targets.append(
                {
                    "object_id": oid,
                    "title": odata.get("title", oid),
                    "observable": False,
                    "max_altitude_deg": odata.get("max_altitude_deg"),
                    "min_required_altitude_deg": settings.min_observable_altitude_deg,
                }
            )

    prof = _extract_equipment_profile(state)
    eff_focal = intent.focal_length_mm if intent else None
    eff_aperture = intent.aperture_mm if intent else None
    eff_sensor = intent.sensor_format if intent else "APS-C"
    eff_mount = intent.mount_type if intent else "equatorial_goto"
    eff_autoguide = bool(intent.has_autoguiding) if intent else False
    eff_filters = list(intent.filters) if intent else []

    # Enrich retrieved DSOs (e.g. from search_catalog in RAG/factual mode) so catalog_objects is never sparse
    for cid in candidate_cite_ids:
        if len(object_records) >= 6:
            break
        if cid not in object_records and not cid.startswith(("CON_", "GUIDE_")):
            enriched = get_object(
                object_id=cid,
                focal_length_mm=eff_focal,
                sensor_format=eff_sensor,
                aperture_mm=eff_aperture,
                mount_type=eff_mount,
                has_autoguiding=eff_autoguide,
                filters=eff_filters,
            )
            if enriched.get("found"):
                object_records[cid] = enriched

    has_rig = bool(
        prof is not None
        or eff_focal is not None
        or eff_aperture is not None
        or (intent and intent.mode == "astrophotography")
    )

    catalog_summary: list[dict[str, Any]] = []
    shared_cam_defaults: dict[str, Any] = {}
    for cid in candidate_cite_ids:
        if len(catalog_summary) >= 5:
            break
        if cid in object_records:
            rec = object_records[cid]
            const_code = str(rec.get("constellation") or "").upper()
            sg = rec.get("shooting_guide") or {}
            sg_ori = sg.get("orientation") or {}
            sg_app = sg.get("appearance") or {}
            sg_cam = sg.get("camera_settings") or {}
            if has_rig and not shared_cam_defaults and sg_cam:
                shared_cam_defaults = {
                    "fov_arcmin": sg_cam.get("fov_arcmin"),
                    "recommended_iso_or_gain": sg_cam.get("recommended_iso_or_gain"),
                }
            raw_look = str(sg_app.get("visual_and_photo_look") or "")
            short_look = raw_look.split(".")[0].strip()
            if "HDR" in raw_look:
                short_look += "; HDR bright core (blend short+long subs)"
            raw_sub = str(sg_cam.get("recommended_sub_exposure") or "")
            short_sub = raw_sub.split("(")[0].strip() if "500-rule" not in raw_sub else raw_sub
            nb_ok = bool(rec.get("narrowband_friendly"))
            compact_sg = {
                k: v
                for k, v in {
                    "coords": f"{sg_ori.get('ra_hms') or ''} {sg_ori.get('dec_dms') or ''}".strip()
                    or None,
                    "size_vs_moon": sg_ori.get("size_vs_full_moon"),
                    "how_to_find": str(sg_ori.get("how_to_find") or "").split(".")[0].strip()
                    or None,
                    "visual_and_photo_look": short_look or None,
                    "target_fill_pct": sg_cam.get("target_fill_pct") if has_rig else None,
                    "recommended_sub_exposure": short_sub or None,
                    "filter": "Dual-band Hα/OIII or RGB"
                    if nb_ok
                    else "Broadband UV/IR-cut (no narrowband)",
                }.items()
                if v is not None
            }
            obj_entry: dict[str, Any] = {
                "doc_id": cid,
                "title": rec.get("title"),
                "type": rec.get("object_type"),
                "constellation": f"{_CONSTELLATION_NAMES.get(const_code, const_code)} ({const_code})",
                "magnitude": rec.get("magnitude"),
                "size_arcmin": rec.get("size_major_arcmin"),
                "narrowband": nb_ok,
                "shooting_guide": compact_sg,
            }
            if f"CON_{const_code}" in retrieved_doc_ids:
                obj_entry["constellation_doc_id"] = f"CON_{const_code}"
            if has_rig and rec.get("framing_advice"):
                obj_entry["framing_advice"] = rec["framing_advice"]
            catalog_summary.append(obj_entry)

    optics_payload: dict[str, Any] = {
        k: v
        for k, v in {
            "mode": intent.mode if intent else "hybrid",
            "aperture_mm": eff_aperture,
            "focal_length_mm": eff_focal,
            "sensor_format": eff_sensor if has_rig else None,
            "mount_type": eff_mount if has_rig else None,
            "has_autoguiding": eff_autoguide if eff_autoguide else None,
            "filters": eff_filters if eff_filters else None,
            **shared_cam_defaults,
        }.items()
        if v is not None
    }
    if prof is not None:
        optics_payload["user_equipment_profile"] = {
            "rig_summary": _format_rig_summary(prof),
            "telescope_or_lens": prof.telescope_or_lens,
            "focal_length_mm": prof.focal_length_mm,
            "aperture_mm": prof.aperture_mm,
            "f_ratio": f"f/{round(prof.focal_length_mm / prof.aperture_mm, 1)}"
            if prof.aperture_mm
            else None,
            "camera_model": prof.camera_model,
            "sensor_format": prof.sensor_format,
            "mount_type": prof.mount_type,
            "has_autoguiding": prof.has_autoguiding,
            "filters": prof.filters,
            "bortle_class": prof.bortle_class,
        }

    deterministic_json: dict[str, Any] = {
        "reference_date": settings.get_reference_date().isoformat(),
        "max_weather_forecast_horizon_days": settings.max_forecast_days,
        "location": location_payload,
        "optics": optics_payload,
    }
    if unindexed := _find_unindexed_designations(q):
        deterministic_json["unindexed_designations_note"] = (
            f"Designation(s) {unindexed} are not standalone corpus records; "
            "answer astronomically and cite only retrieved IDs below."
        )
    if is_multi_night and ranked_nights:
        deterministic_json["multi_night_ranking"] = ranked_nights
    elif active_night or active_weather:
        if active_night:
            deterministic_json["night_ephemeris"] = {
                k: active_night[k]
                for k in (
                    "date",
                    "weekday",
                    "tz_label",
                    "astronomical_dark_start",
                    "astronomical_dark_end",
                    "moonless_dark_hours",
                    "moon_illumination_pct",
                    "moon_phase_name",
                    "moon_set_local",
                    "moon_rise_local",
                )
                if k in active_night and active_night[k] is not None
            }
        if active_weather:
            deterministic_json["night_weather"] = {
                k: active_weather[k]
                for k in (
                    "available",
                    "date",
                    "cloud_cover_pct",
                    "evening_cloud_cover_pct",
                    "late_night_cloud_cover_pct",
                    "clear_hours",
                    "best_clear_window",
                    "max_precip_prob_pct",
                    "wind_kmh",
                    "humidity_pct",
                    "transparency",
                )
                if k in active_weather and active_weather[k] is not None
            }
    if plan_items:
        deterministic_json["verified_observation_schedule"] = [
            {
                "object_id": pi.object_id,
                "name": pi.name,
                "slot": f"{pi.start.strftime('%H:%M')}–{pi.end.strftime('%H:%M')}",
                "altitude_deg": pi.altitude_deg,
                "direction": pi.direction,
                "difficulty": pi.difficulty,
            }
            for pi in plan_items
        ]
    if unobservable_targets:
        deterministic_json["unobservable_targets"] = unobservable_targets
    if catalog_summary:
        deterministic_json["catalog_objects"] = catalog_summary

    catalog_doc_ids = {item["doc_id"] for item in catalog_summary}
    snippet_chunks = [
        c
        for c in retrieved
        if c.doc_id in candidate_cite_ids
        and not (c.chunk_id.endswith("#catalog-0") and c.doc_id in catalog_doc_ids)
    ]
    if not snippet_chunks and not catalog_summary:
        snippet_chunks = [c for c in retrieved if c.doc_id in candidate_cite_ids]

    def _clean_chunk_body(txt: str) -> str:
        if txt.startswith("[") and "]\n" in txt:
            txt = txt.split("]\n", 1)[1]
        if txt.startswith("Section:"):
            parts = txt.split("\n", 1)
            txt = parts[1] if len(parts) > 1 else txt
        return txt.strip()[:220]

    context_snippets = "\n".join(
        f"[{c.doc_id}] ({c.section}): {_clean_chunk_body(c.text)}" for c in snippet_chunks[:4]
    )

    prev_ans = state.get("answer")
    retry_note = ""
    if state.get("retries", 0) > 0 and prev_ans and prev_ans.verification_errors:
        retry_note = (
            "\n\nCORRECTION REQUIRED FROM VALIDATOR: "
            + "; ".join(prev_ans.verification_errors)
            + ". ONLY cite IDs from `Available corpus citation IDs`."
        )

    lang_label = language_name(lang)
    sys_prompt = (
        "You are AstroPilote, an expert astronomy and astrophotography assistant.\n"
        f"LANGUAGE: Write your ENTIRE response in {lang_label} (prose, headings, lists), keeping bracketed IDs like `[M031]` unchanged.\n"
        "Strict grounding rules:\n"
        "1. Ground coordinates, darkness hours, Moon phase/illumination/rise/set, cloud cover %, clear windows, altitudes, and FOV strictly in `Deterministic Tool Outputs (JSON)`. Never claim live weather/Moon failed when present.\n"
        "2. If `night_weather.available` is false (> `max_weather_forecast_horizon_days` = 14 days), state that weather forecasts only extend 14 days ahead while still answering sky visibility.\n"
        "3. When `framing_advice` is present, cite exact sensor FOV arcminutes (e.g. `123.8 x 82.5` arcminutes) and whether the object fits.\n"
        "4. Politely correct any false premise (wrong constellation/type) using `catalog_objects` and cite `[CON_XXX]` if retrieved.\n"
        "5. If `location.provided` is false, NEVER assume any city (never mention Fontainebleau or Paris); suggest targets and invite the user to share their city/coordinates.\n"
        "6. Cite retrieved sources inline using ONLY exact corpus IDs from `Available corpus citation IDs` (e.g. `[M031]`, `[C020]`). NEVER put JSON key names like `[night_ephemeris]`, `[optics]`, or `[Deterministic Tool Outputs]` in brackets.\n"
        "7. USER EQUIPMENT PROFILE & PHOTOGRAPHY SETTINGS: When `optics.user_equipment_profile` or `focal_length_mm` is set, tailor framing (`fov_arcmin`, % fill), sub-exposure (500-rule on tripod vs tracked subs), ISO/Gain, and filter advice to that rig. When no telescope/camera is specified, give general naked-eye, binocular, and telescope advice without assuming a specific sensor.\n"
        "8. PRACTICAL OBJECT GUIDE: For each target you present from `catalog_objects`, cover orientation (constellation, direction/altitude, how to find), visual & photo look (magnitude, size vs 30' Moon, morphology/colors), and how to observe/shoot it.\n"
        "9. MULTI-OBJECT DEPTH & PACING: Skip generic greetings and never stop after 1 object when multiple targets fit:\n"
        "   - For session plans or target recommendations, start with a concise Night & Weather Conditions summary (3–4 bullets), then write a focused 80–110 word subsection for **EVERY ONE** of the 3–5 scheduled/recommended targets so all targets are completed thoroughly, and end with 2 lines of session tips.\n"
        "   - For single-object questions, give a thorough guide to that primary object AND a concise section highlighting 1–2 nearby companions/neighbors from `catalog_objects`."
    )

    if len(plan_items) >= 2:
        multi_target_hint = (
            f" Cover ALL {len(plan_items)} scheduled targets in order "
            f"({[pi.object_id for pi in plan_items]}) with ~90 words per target after a concise Night & Weather summary."
        )
    elif not unindexed and not priority_dsos and len(catalog_summary) >= 2:
        multi_target_hint = (
            f" Present 3 to {len(catalog_summary)} distinct targets from `catalog_objects` "
            f"({[c['doc_id'] for c in catalog_summary]}) with ~95 words of astronomical and practical detail per target."
        )
    elif not unindexed and priority_dsos and len(catalog_summary) >= 2:
        companion_ids = [c["doc_id"] for c in catalog_summary if c["doc_id"] not in priority_dsos][
            :2
        ]
        multi_target_hint = (
            f" Cover {priority_dsos} in rich detail and also present nearby companion/neighbor target(s) {companion_ids} (citing their `[DOC_ID]`)."
            if companion_ids
            else ""
        )
    else:
        multi_target_hint = ""
    user_prompt = (
        f"User Question: {q}\n\n"
        f"Deterministic Tool Outputs (JSON):\n{json.dumps(deterministic_json, ensure_ascii=False, separators=(',', ':'))}\n\n"
        f"Retrieved Corpus Passages:\n{context_snippets}\n\n"
        f"Available corpus citation IDs: {candidate_cite_ids}{retry_note}\n\n"
        f"Write a complete, well-structured response in {lang_label}, grounded in these exact figures and citing `[DOC_ID]` sources inline.{multi_target_hint}"
    )

    answer_text, extra_in, extra_out, _backend = generate_grounded_answer(
        sys_prompt,
        user_prompt,
        mode_override=state.get("llm_mode_override"),
    )
    # Strip any accidental bracketed JSON section names produced by smaller local models
    answer_text = re.sub(
        r"\s*\[(?:night_ephemeris|night_weather|optics|location|verified_observation_schedule|catalog_objects|multi_night_ranking|Deterministic[^\]]*)\]",
        "",
        answer_text,
        flags=re.IGNORECASE,
    )

    # Extract retrieved IDs actually cited inline (plus any scheduled plan items and host constellations)
    inline_cited: list[str] = []
    for m in re.finditer(r"\[((?:M|C|NGC\s*|IC\s*|CON_|GUIDE_)[A-Z0-9_]+)\]", answer_text):
        raw_cid = m.group(1).strip()
        if raw_cid in retrieved_doc_ids:
            inline_cited.append(raw_cid)
        else:
            norm_cid = normalize_object_id(raw_cid)
            if norm_cid in retrieved_doc_ids:
                inline_cited.append(norm_cid)
            else:
                rec = resolve_object_record(raw_cid)
                if rec and rec.get("doc_id") in retrieved_doc_ids:
                    inline_cited.append(str(rec["doc_id"]))

    plan_ids = [pi.object_id for pi in plan_items if pi.object_id in retrieved_doc_ids]
    merged_sources = list(dict.fromkeys(inline_cited + plan_ids))
    if not merged_sources and candidate_cite_ids:
        merged_sources = candidate_cite_ids[:3]

    ans = AstroAnswer(
        answer=answer_text,
        plan=plan_items,
        conditions=conditions_str,
        sources=merged_sources,
        abstained=False,
        verified=True,
    )
    return {
        "answer": ans,
        "tokens_in": state.get("tokens_in", 0) + extra_in,
        "tokens_out": state.get("tokens_out", 0) + extra_out,
    }


def validate_node(state: AstroState) -> dict:
    """Node 5 (`validate` / Vérifier): Run pure-Python verification on `state['answer']`."""
    ans = state.get("answer")
    if ans is None:
        return {}
    checked = verify_astro_answer(ans, state)
    retries = state.get("retries", 0)
    if not checked.verified:
        retries += 1
    return {"answer": checked, "retries": retries}
