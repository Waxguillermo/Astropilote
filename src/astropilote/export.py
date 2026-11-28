"""Field-ready astronomy session exports (RFC 5545 `.ics` calendars & N.I.N.A./Telescopius `.csv`)."""

from __future__ import annotations

import csv
import io
import math
import re
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from astropilote.tools.catalog import resolve_object_record

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_TIME_HM_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?")
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t")


def _finite_float(value: Any) -> float | None:
    """Coerce `value` to a finite float, returning `None` for missing, non-numeric, or non-finite inputs."""
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (ValueError, TypeError):
        return None
    return num if math.isfinite(num) else None


def safe_download_filename(prefix: str, suffix: str) -> str:
    """Return an ASCII filename safe for HTTP `Content-Disposition` headers."""
    cleaned_prefix = _SAFE_NAME_RE.sub("_", prefix.strip()).strip("._-") or "astropilote"
    cleaned_suffix = _SAFE_NAME_RE.sub("", suffix.lstrip(".")) or "txt"
    return f"{cleaned_prefix[:64]}.{cleaned_suffix[:12]}"


def deg_to_hms(ra_deg: float) -> str:
    """Format J2000 right ascension in degrees [0, 360) as `HHh MMm SS.Ss`."""
    total_seconds = ((float(ra_deg) % 360.0) / 15.0) * 3600.0
    hours = int(total_seconds // 3600)
    rem = total_seconds - hours * 3600
    minutes = int(rem // 60)
    seconds = round(rem - minutes * 60, 1)
    if seconds >= 60.0:
        seconds = 0.0
        minutes += 1
    if minutes >= 60:
        minutes = 0
        hours = (hours + 1) % 24
    return f"{hours:02d}h {minutes:02d}m {seconds:04.1f}s"


def deg_to_dms(dec_deg: float) -> str:
    """Format J2000 declination in degrees [-90, +90] as `±DD° MM' SS\"`."""
    val = max(-90.0, min(90.0, float(dec_deg)))
    sign = "-" if val < 0 else "+"
    total_arcsec = round(abs(val) * 3600.0)
    degrees = int(total_arcsec // 3600)
    rem = total_arcsec - degrees * 3600
    arcmin = int(rem // 60)
    arcsec = int(rem - arcmin * 60)
    return f"{sign}{degrees:02d}° {arcmin:02d}' {arcsec:02d}\""


def _sanitize_csv_text(value: Any) -> str:
    """Neutralize spreadsheet formula triggers on untrusted text cells."""
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    if text and text[0] in _FORMULA_PREFIXES:
        return "'" + text
    return text


def _ics_escape(value: Any) -> str:
    """Escape RFC 5545 TEXT values (`\\`, `;`, `,`, newlines)."""
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def _fold_ics_line(line: str, max_octets: int = 75) -> str:
    """Fold long RFC 5545 lines at `max_octets` UTF-8 bytes with CRLF + leading space."""
    if len(line.encode("utf-8")) <= max_octets:
        return line
    chunks: list[str] = []
    current = ""
    limit = max_octets
    for ch in line:
        candidate = current + ch
        if len(candidate.encode("utf-8")) > limit and current:
            chunks.append(current)
            current = " " + ch
            limit = max_octets
        else:
            current = candidate
    if current:
        chunks.append(current)
    return "\r\n".join(chunks)


def _ics_utc(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _parse_hhmm(raw: str | None) -> time | None:
    if not raw:
        return None
    match = _TIME_HM_RE.match(str(raw))
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    second = int(match.group(3) or 0)
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return None
    return time(hour, minute, second)


def _night_slot_bounds(
    obs_date: date,
    start_t: time,
    end_t: time,
    tz: ZoneInfo,
) -> tuple[datetime, datetime]:
    """Map an evening/night `HH:MM` slot onto `obs_date` (rolling past-midnight times to `+1d`)."""
    start_day = obs_date + timedelta(days=1) if start_t.hour < 12 else obs_date
    start_dt = datetime.combine(start_day, start_t, tzinfo=tz)
    end_day = start_day + timedelta(days=1) if end_t <= start_t else start_day
    end_dt = datetime.combine(end_day, end_t, tzinfo=tz)
    return start_dt, end_dt


def _resolve_tz(tz_name: Any) -> ZoneInfo:
    if isinstance(tz_name, str) and tz_name.strip():
        try:
            return ZoneInfo(tz_name.strip())
        except (ZoneInfoNotFoundError, ValueError):
            pass
    return ZoneInfo("UTC")


def export_plan_csv(
    items: Sequence[dict[str, Any]],
    *,
    lookup: Callable[[str], dict[str, Any] | None] | None = None,
) -> str:
    """Serialize an observation plan into a N.I.N.A. / Telescopius-compatible CSV target list."""
    resolver = lookup or resolve_object_record
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(
        [
            "Pane",
            "RA",
            "Dec",
            "Position Angle",
            "Object",
            "Catalog ID",
            "Constellation",
            "Magnitude",
            "Window",
            "Peak Alt (deg)",
            "Direction",
            "Notes",
        ]
    )

    for idx, item in enumerate(items, start=1):
        object_id = str(item.get("object_id") or "").strip()
        rec = resolver(object_id) if object_id else None
        meta = (rec or {}).get("metadata") or {}

        ra_val = _finite_float(
            item.get("ra_deg") if item.get("ra_deg") is not None else meta.get("ra_deg")
        )
        dec_val = _finite_float(
            item.get("dec_deg") if item.get("dec_deg") is not None else meta.get("dec_deg")
        )
        constellation = item.get("constellation") or meta.get("constellation") or ""
        magnitude = _finite_float(
            item.get("magnitude") if item.get("magnitude") is not None else meta.get("magnitude")
        )

        start_str = str(item.get("start") or "")[:5]
        end_str = str(item.get("end") or "")[:5]
        window = f"{start_str}-{end_str}" if start_str and end_str else start_str or end_str
        alt_val = _finite_float(item.get("altitude_deg"))
        alt_str = f"{alt_val:.1f}" if alt_val is not None else ""
        mag_str = f"{magnitude:.1f}" if magnitude is not None else ""

        writer.writerow(
            [
                f"Target {idx}",
                deg_to_hms(ra_val) if ra_val is not None else "",
                deg_to_dms(dec_val) if dec_val is not None else "",
                "0.0",
                _sanitize_csv_text(item.get("name") or object_id),
                _sanitize_csv_text(object_id),
                _sanitize_csv_text(constellation),
                mag_str,
                _sanitize_csv_text(window),
                alt_str,
                _sanitize_csv_text(item.get("direction") or ""),
                _sanitize_csv_text(item.get("imaging_notes") or ""),
            ]
        )
    return buf.getvalue()


def export_plan_ics(
    items: Sequence[dict[str, Any]],
    *,
    obs_date: date,
    location_name: str = "Observation Site",
    tz_name: str | None = None,
    night: dict[str, Any] | None = None,
    weather: dict[str, Any] | None = None,
) -> str:
    """Build an RFC 5545 `.ics` calendar containing one VEVENT per planned observing target."""
    tz = _resolve_tz(tz_name or (night or {}).get("timezone"))
    stamp = _ics_utc(datetime.now(UTC))
    night = night or {}
    weather = weather or {}

    cond_parts: list[str] = []
    moon_pct = _finite_float(night.get("moon_illumination_pct"))
    if moon_pct is not None:
        cond_parts.append(f"Moon: {round(moon_pct)}%")
    cloud_pct = (
        _finite_float(weather.get("cloud_cover_pct"))
        if weather.get("available") is not False
        else None
    )
    if cloud_pct is not None:
        cond_parts.append(f"Clouds: {round(cloud_pct)}%")
    conditions_line = " | ".join(cond_parts)

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//AstroPilote//Night Observation Planner//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        f"X-WR-CALNAME:{_ics_escape(f'AstroPilote — {location_name}')}",
    ]

    events_added = 0
    for idx, item in enumerate(items, start=1):
        start_t = _parse_hhmm(item.get("start"))
        end_t = _parse_hhmm(item.get("end"))
        if start_t is None or end_t is None:
            continue
        start_dt, end_dt = _night_slot_bounds(obs_date, start_t, end_t, tz)
        oid = str(item.get("object_id") or f"T{idx}").strip()
        name = str(item.get("name") or oid).strip()
        summary = f"Observe {name} ({oid})" if oid and oid not in name else f"Observe {name}"

        desc_lines = [f"Target: {name} ({oid})"]
        alt_val = _finite_float(item.get("altitude_deg"))
        if alt_val is not None:
            direction = str(item.get("direction") or "").strip()
            alt_text = f"Peak altitude: {alt_val:.0f}°"
            if direction:
                alt_text += f" ({direction})"
            desc_lines.append(alt_text)
        if conditions_line:
            desc_lines.append(conditions_line)
        if item.get("imaging_notes"):
            desc_lines.append(f"Notes: {str(item['imaging_notes']).strip()}")

        uid_token = _SAFE_NAME_RE.sub("", oid) or f"target{idx}"
        uid = f"astropilote-plan-{obs_date.isoformat()}-{idx}-{uid_token}@astropilote.local"
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:{uid}",
                f"DTSTAMP:{stamp}",
                f"DTSTART:{_ics_utc(start_dt)}",
                f"DTEND:{_ics_utc(end_dt)}",
                f"SUMMARY:{_ics_escape(summary)}",
                f"LOCATION:{_ics_escape(location_name)}",
                f"DESCRIPTION:{_ics_escape(chr(10).join(desc_lines))}",
                "END:VEVENT",
            ]
        )
        events_added += 1

    if events_added == 0:
        dark_start = _parse_hhmm(night.get("astronomical_dark_start")) or time(21, 0)
        dark_end = _parse_hhmm(night.get("astronomical_dark_end")) or time(4, 0)
        start_dt, end_dt = _night_slot_bounds(obs_date, dark_start, dark_end, tz)
        desc = conditions_line or "Planned astronomical observation session."
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:astropilote-night-{obs_date.isoformat()}@astropilote.local",
                f"DTSTAMP:{stamp}",
                f"DTSTART:{_ics_utc(start_dt)}",
                f"DTEND:{_ics_utc(end_dt)}",
                f"SUMMARY:{_ics_escape(f'AstroPilote Observing Night — {location_name}')}",
                f"LOCATION:{_ics_escape(location_name)}",
                f"DESCRIPTION:{_ics_escape(desc)}",
                "END:VEVENT",
            ]
        )

    lines.append("END:VCALENDAR")
    return "\r\n".join(_fold_ics_line(line) for line in lines) + "\r\n"


def export_booking_ics(
    *,
    booking_id: str,
    telescope_name: str,
    start_utc: str,
    end_utc: str,
    site: str = "Lyon Observatory",
    aperture_mm: int | float | None = None,
    price_cents: int | None = None,
) -> str:
    """Build an RFC 5545 `.ics` calendar invite for a confirmed remote observatory booking."""
    start_dt = datetime.fromisoformat(start_utc.strip().replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end_utc.strip().replace("Z", "+00:00"))
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=UTC)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=UTC)
    if end_dt <= start_dt:
        raise ValueError("Booking end_utc must be after start_utc.")

    stamp = _ics_utc(datetime.now(UTC))
    if aperture_mm is not None:
        aperture_val = _finite_float(aperture_mm)
        if aperture_val is None or aperture_val <= 0:
            raise ValueError("Booking aperture_mm must be a finite positive number.")
    else:
        aperture_val = None
    tel_label = f"{telescope_name} ({round(aperture_val)}mm)" if aperture_val else telescope_name
    desc_lines = [
        f"Reservation reference: {booking_id}",
        f"Instrument: {tel_label}",
        f"Site: {site}",
    ]
    if price_cents is not None:
        desc_lines.append(f"Session fee: €{price_cents / 100:.0f}")

    uid_token = _SAFE_NAME_RE.sub("", booking_id) or "session"
    uid = f"astropilote-booking-{uid_token}@astropilote.local"
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//AstroPilote//Observatory Concierge//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{stamp}",
        f"DTSTART:{_ics_utc(start_dt)}",
        f"DTEND:{_ics_utc(end_dt)}",
        f"SUMMARY:{_ics_escape(f'Telescope Session · {tel_label} ({booking_id})')}",
        f"LOCATION:{_ics_escape(site)}",
        f"DESCRIPTION:{_ics_escape(chr(10).join(desc_lines))}",
        "END:VEVENT",
        "END:VCALENDAR",
    ]
    return "\r\n".join(_fold_ics_line(line) for line in lines) + "\r\n"
