"""Unit and API tests for field-ready `.ics` calendar and N.I.N.A. `.csv` session exports."""

from __future__ import annotations

import csv
import io
from datetime import date

import pytest
from fastapi.testclient import TestClient

from astropilote.app.main import app
from astropilote.export import (
    deg_to_dms,
    deg_to_hms,
    export_booking_ics,
    export_plan_csv,
    export_plan_ics,
    safe_download_filename,
)

client = TestClient(app)


def test_coordinate_formatters_for_nina_and_telescopius() -> None:
    assert deg_to_hms(10.6847) == "00h 42m 44.3s"
    assert deg_to_dms(41.2687) == "+41° 16' 07\""
    assert deg_to_dms(-5.3911) == "-05° 23' 28\""
    assert deg_to_hms(359.9999) == "00h 00m 00.0s"


def test_safe_download_filename_strips_header_injection() -> None:
    name = safe_download_filename('plan"; filename="evil\r\n', ".ics")
    assert '"' not in name
    assert "\r" not in name
    assert "\n" not in name
    assert name.endswith(".ics")


def test_export_plan_csv_enriches_catalog_coordinates_and_sanitizes_formulas() -> None:
    rows = export_plan_csv(
        [
            {
                "object_id": "M031",
                "name": "Andromeda Galaxy",
                "start": "=1+1",
                "end": "",
                "altitude_deg": 78.4,
                "direction": "SE",
                "imaging_notes": "=CMD|'/C calc'!A0",
            },
            {
                "object_id": "M042",
                "name": "Orion Nebula",
                "start": "23:30",
                "end": "01:30",
                "altitude_deg": 39.2,
                "direction": "S",
                "imaging_notes": "Use 30s core + 180s HDR subs",
            },
        ]
    )
    parsed = list(csv.DictReader(io.StringIO(rows)))
    assert len(parsed) == 2
    assert parsed[0]["Catalog ID"] == "M031"
    assert parsed[0]["RA"].startswith("00h 42m")
    assert parsed[0]["Dec"].startswith("+41° 16'")
    assert parsed[0]["Constellation"] == "AND"
    assert parsed[0]["Window"].startswith("'=")
    assert parsed[0]["Notes"].startswith("'=")
    assert parsed[1]["Catalog ID"] == "M042"
    assert parsed[1]["Window"] == "23:30-01:30"
    assert parsed[1]["Dec"].startswith("-05° 23'")


def test_export_endpoints_tolerate_malformed_night_weather_and_reject_nonfinite() -> None:
    r_weird = client.post(
        "/api/export/session",
        json={
            "format": "ics",
            "obs_date": "2026-11-17",
            "night": {"timezone": 123, "moon_illumination_pct": "unknown"},
            "weather": {"available": True, "cloud_cover_pct": "NaN"},
            "plan": [],
        },
    )
    assert r_weird.status_code == 200
    assert "BEGIN:VCALENDAR" in r_weird.text

    r_inf = client.post(
        "/api/export/booking",
        content=(
            '{"booking_id":"BK-1","telescope_name":"CDK","start_utc":"2026-11-17T20:00:00Z",'
            '"end_utc":"2026-11-17T22:00:00Z","aperture_mm":Infinity}'
        ),
        headers={"Content-Type": "application/json"},
    )
    assert r_inf.status_code == 400


def test_export_plan_ics_handles_midnight_rollover_and_rfc5545_escaping() -> None:
    ics = export_plan_ics(
        [
            {
                "object_id": "M042",
                "name": "Orion Nebula; Core, Trap",
                "start": "23:30",
                "end": "01:30",
                "altitude_deg": 39.0,
                "direction": "S",
                "imaging_notes": "Dual-band filter, 120s subs",
            }
        ],
        obs_date=date(2026, 11, 17),
        location_name="Lyon, France",
        tz_name="Europe/Paris",
        night={"moon_illumination_pct": 18.0},
        weather={"available": True, "cloud_cover_pct": 10.0},
    )
    assert "\r\n" in ics
    assert ics.startswith("BEGIN:VCALENDAR\r\n")
    assert ics.endswith("END:VCALENDAR\r\n")
    # 2026-11-17 23:30 CET (UTC+1) -> 22:30Z; 01:30 CET next day -> 00:30Z on 2026-11-18
    assert "DTSTART:20261117T223000Z" in ics
    assert "DTEND:20261118T003000Z" in ics
    assert r"SUMMARY:Observe Orion Nebula\; Core\, Trap (M042)" in ics
    assert r"LOCATION:Lyon\, France" in ics
    assert "Moon: 18%" in ics
    assert "Clouds: 10%" in ics


def test_export_booking_ics_validates_window_and_formats_event() -> None:
    ics = export_booking_ics(
        booking_id="BK-001",
        telescope_name="Planewave CDK 14",
        aperture_mm=356,
        start_utc="2026-11-17T20:00:00Z",
        end_utc="2026-11-17T22:00:00Z",
        site="Lyon Observatory",
        price_cents=4500,
    )
    assert "DTSTART:20261117T200000Z" in ics
    assert "DTEND:20261117T220000Z" in ics
    assert "SUMMARY:Telescope Session · Planewave CDK 14 (356mm) (BK-001)" in ics
    assert "€45" in ics

    with pytest.raises(ValueError, match="after start_utc"):
        export_booking_ics(
            booking_id="BK-BAD",
            telescope_name="Planewave CDK 14",
            start_utc="2026-11-17T22:00:00Z",
            end_utc="2026-11-17T20:00:00Z",
        )


def test_export_api_endpoints_return_downloadable_ics_and_csv() -> None:
    payload = {
        "format": "csv",
        "obs_date": "2026-11-17",
        "location_name": "Fontainebleau",
        "timezone": "Europe/Paris",
        "plan": [
            {
                "object_id": "M031",
                "name": "Andromeda Galaxy",
                "start": "21:00",
                "end": "23:00",
                "altitude_deg": 75.0,
                "direction": "SE",
                "imaging_notes": "Wide-field APO",
            }
        ],
    }
    r_csv = client.post("/api/export/session", json=payload)
    assert r_csv.status_code == 200
    assert r_csv.headers["content-type"].startswith("text/csv")
    assert 'filename="astropilote_nina_2026-11-17.csv"' in r_csv.headers["content-disposition"]
    assert "M031" in r_csv.text and "00h 42m" in r_csv.text

    r_ics = client.post("/api/export/session", json={**payload, "format": "ics"})
    assert r_ics.status_code == 200
    assert r_ics.headers["content-type"].startswith("text/calendar")
    assert 'filename="astropilote_session_2026-11-17.ics"' in r_ics.headers["content-disposition"]
    assert "BEGIN:VCALENDAR" in r_ics.text

    r_bk = client.post(
        "/api/export/booking",
        json={
            "booking_id": "BK-002",
            "telescope_name": "Takahashi FSQ-106",
            "aperture_mm": 106,
            "start_utc": "2026-11-17T21:00:00Z",
            "end_utc": "2026-11-17T23:00:00Z",
            "site": "Lyon Observatory",
            "price_cents": 3500,
        },
    )
    assert r_bk.status_code == 200
    assert r_bk.headers["content-type"].startswith("text/calendar")
    assert 'filename="astropilote_BK-002.ics"' in r_bk.headers["content-disposition"]
    assert "BK-002" in r_bk.text
