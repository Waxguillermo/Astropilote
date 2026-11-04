"""Regression tests covering AstroPilote S1–S3 audit fixes across optics, citations, LLM, sky, and API security."""

from __future__ import annotations

import base64
import io
import stat

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import astropilote.llm as llm_mod
from astropilote.agent.citations import extract_inline_citations
from astropilote.agent.schemas import AstroAnswer, Chunk, ChunkMetadata
from astropilote.agent.state import AstroState
from astropilote.agent.validator import scrub_unverified_citations, verify_astro_answer
from astropilote.app.main import app
from astropilote.llm import (
    _resolve_anthropic_direct_model,
    _resolve_openrouter_model,
    save_anthropic_key,
)
from astropilote.optics import (
    EQUIPMENT_PRESETS,
    SENSOR_SPECS,
    angular_separation_deg,
    compute_fov_arcmin,
    compute_fov_deg,
    compute_npf_seconds,
    compute_pixel_scale_arcsec,
)
from astropilote.tools.sky import compute_night_ephemeris


def _tiny_png_base64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=(10, 12, 20)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_optics_seestar_s50_and_smartphone_fov() -> None:
    """Verify exact rectilinear FOV for Seestar S50 (1/2.8-inch IMX462) and smartphone wide lens (5.1mm)."""
    presets_by_id = {p["preset_id"]: p for p in EQUIPMENT_PRESETS}

    # 1. ZWO Seestar S50: 250mm f/5 on 1/2.8-inch (5.57 x 3.13 mm) -> ~76.6' x 43.0' (~1.28° x 0.72°)
    seestar = presets_by_id["seestar_s50"]
    assert seestar["sensor_format"] == "1/2.8-inch"
    assert seestar["focal_length_mm"] == 250.0
    assert SENSOR_SPECS["1/2.8-inch"] == (5.57, 3.13)

    s50_w_arcmin, s50_h_arcmin = compute_fov_arcmin(
        seestar["focal_length_mm"], seestar["sensor_format"]
    )
    assert s50_w_arcmin == pytest.approx(76.6, abs=0.15)
    assert s50_h_arcmin == pytest.approx(43.0, abs=0.15)

    s50_w_deg, s50_h_deg = compute_fov_deg(250.0, *SENSOR_SPECS["1/2.8-inch"])
    assert s50_w_deg == pytest.approx(1.28, abs=0.02)
    assert s50_h_deg == pytest.approx(0.72, abs=0.02)
    assert compute_pixel_scale_arcsec(250.0, 2.9) == pytest.approx(2.39, abs=0.02)

    # 2. Smartphone Night Mode: physical 5.1mm wide lens on 7.6 x 5.7 mm (~24mm eq., ~73.4° x 58.4°)
    phone = presets_by_id["smartphone_night"]
    assert phone["sensor_format"] == "smartphone"
    assert phone["focal_length_mm"] == pytest.approx(5.1)
    assert SENSOR_SPECS["smartphone"] == (7.6, 5.7)

    ph_w_deg, ph_h_deg = compute_fov_deg(phone["focal_length_mm"], *SENSOR_SPECS["smartphone"])
    assert 70.0 < ph_w_deg < 77.0
    assert 55.0 < ph_h_deg < 62.0
    ph_w_arcmin, ph_h_arcmin = compute_fov_arcmin(phone["focal_length_mm"], "smartphone")
    assert ph_w_arcmin == pytest.approx(ph_w_deg * 60.0, abs=0.1)
    assert ph_h_arcmin == pytest.approx(ph_h_deg * 60.0, abs=0.1)


def test_optics_npf_rule_and_haversine_angular_separation() -> None:
    """Verify Sahavre's NPF untracked exposure rule and numerically stable Haversine separation."""
    # NPF at celestial equator: t = (35 * (50/28) + 30 * 3.76) / 50 = 3.506s -> 3.5s
    t_eq = compute_npf_seconds(
        focal_length_mm=50.0, aperture_mm=28.0, pixel_size_um=3.76, dec_deg=0.0
    )
    assert t_eq == pytest.approx(3.5, abs=0.15)

    # At Dec +60° (cos 60° = 0.5), permissible untracked exposure doubles (~7.0s)
    t_dec60 = compute_npf_seconds(
        focal_length_mm=50.0, aperture_mm=28.0, pixel_size_um=3.76, dec_deg=60.0
    )
    assert t_dec60 == pytest.approx(7.0, abs=0.2)
    assert t_dec60 > t_eq

    # Circumpolar declination remains safely clamped <= 30.0s
    t_pole = compute_npf_seconds(
        focal_length_mm=24.0, aperture_mm=14.0, pixel_size_um=5.9, dec_deg=89.5
    )
    assert 0.5 <= t_pole <= 30.0

    # Finer pixel pitch requires shorter exposures than coarse full-frame pixels
    t_fine = compute_npf_seconds(focal_length_mm=135.0, aperture_mm=67.5, pixel_size_um=2.4)
    t_coarse = compute_npf_seconds(focal_length_mm=135.0, aperture_mm=67.5, pixel_size_um=5.9)
    assert t_fine < t_coarse

    # Haversine great-circle angular separation
    assert angular_separation_deg(10.6847, 41.2687, 10.6847, 41.2687) == 0.0
    assert angular_separation_deg(359.5, 0.0, 0.5, 0.0) == pytest.approx(1.0, abs=1e-5)
    assert angular_separation_deg(0.0, 90.0, 180.0, -90.0) == pytest.approx(180.0, abs=1e-5)
    # M31 to M32 (~24.2 arcmin = ~0.404 deg)
    sep_m31_m32 = angular_separation_deg(10.6847, 41.2687, 10.6742, 40.8652)
    assert sep_m31_m32 == pytest.approx(0.4036, abs=0.01)


def test_grouped_inline_citations_and_markdown_link_exclusion() -> None:
    """Verify extraction of grouped `[M031, CON_AND, M999]` citations while ignoring Markdown links."""
    sample = (
        "Andromeda [M031, CON_AND, M999] is best framed at wide focal lengths; "
        "compare with [C100](https://example.com) and [M081; M082] under [see note below] "
        "Bortle 4 skies [GUIDE_BORTLE]."
    )
    extracted = extract_inline_citations(sample)
    assert extracted == ["M031", "CON_AND", "M999", "M081", "M082", "GUIDE_BORTLE"]
    assert "C100" not in extracted

    # Validator must catch unretrieved M999 inside grouped `[M031, CON_AND, M999]`
    m31_chunk = Chunk(
        chunk_id="M031#0",
        doc_id="M031",
        title="Andromeda Galaxy",
        section="Overview",
        text="M31 in Andromeda.",
        metadata=ChunkMetadata(
            doc_id="M031", catalog="Messier", object_type="galaxy", constellation="AND"
        ),
    )
    con_chunk = Chunk(
        chunk_id="CON_AND#0",
        doc_id="CON_AND",
        title="Andromeda (constellation)",
        section="Overview",
        text="Andromeda constellation.",
        metadata=ChunkMetadata(
            doc_id="CON_AND",
            catalog="Constellation",
            object_type="constellation",
            constellation="AND",
        ),
    )
    state: AstroState = {
        "retrieved": [m31_chunk, con_chunk],
        "calculated_altitudes": {},
        "night_window": None,
    }
    ans = AstroAnswer(
        answer="Observe Andromeda [M031, CON_AND, M999] and read [C100](https://example.com).",
        sources=["M031", "CON_AND"],
        abstained=False,
    )
    checked = verify_astro_answer(ans, state)
    assert checked.verified is False
    assert any("M999" in err for err in checked.verification_errors)
    assert not any("C100" in err for err in checked.verification_errors)

    scrubbed = scrub_unverified_citations(checked, state)
    assert scrubbed.verified is False
    assert "[M031, CON_AND]" in scrubbed.answer
    assert "M999" not in scrubbed.answer
    assert "[C100](https://example.com)" in scrubbed.answer


def test_llm_save_anthropic_key_crlf_rejection_permissions_and_model_resolution() -> None:
    """Verify CRLF injection rejection, atomic 0o600 `.env` permissions, and Sonnet 4.6 resolution."""
    # 1. CRLF / control-character injection must be strictly rejected without touching ENV_PATH
    before_text = llm_mod.ENV_PATH.read_text(encoding="utf-8")
    for malicious in (
        "sk-ant-12345678\nOLLAMA_BASE_URL=http://evil",
        "sk-ant-12345678\rOLLAMA_BASE_URL=http://evil",
        "not-an-api-key-12345678",
    ):
        with pytest.raises(ValueError):
            save_anthropic_key(malicious)
    assert llm_mod.ENV_PATH.read_text(encoding="utf-8") == before_text
    assert "http://evil" not in llm_mod.ENV_PATH.read_text(encoding="utf-8")

    # 2. Valid key writes to isolated temporary ENV_PATH with strict 0o600 permissions
    valid_key = "sk-ant-api03-validtestkey1234567890"
    status = save_anthropic_key(valid_key)
    assert status["anthropic_ready"] is True
    assert status["configured_mode"] == "local"
    file_mode = stat.S_IMODE(llm_mod.ENV_PATH.stat().st_mode)
    assert file_mode == 0o600
    saved_content = llm_mod.ENV_PATH.read_text(encoding="utf-8")
    assert f"ANTHROPIC_API_KEY={valid_key}" in saved_content

    # 3. Legacy Claude Sonnet 4 identifier resolves to current Claude Sonnet 4.6
    assert _resolve_anthropic_direct_model("claude-sonnet-4-20250514") == "claude-sonnet-4-6"
    assert _resolve_openrouter_model("claude-sonnet-4-20250514") == "anthropic/claude-sonnet-4.6"


def test_sky_exact_timezones_and_tromso_midnight_sun() -> None:
    """Verify exact civil timezone offsets (including half-hour IST & southern DST) and Tromsø Midnight Sun."""
    # London in summer (BST = UTC+1)
    eph_london = compute_night_ephemeris(51.5074, -0.1278, "2026-06-21")
    assert eph_london["tz_offset_hours"] == pytest.approx(1.0)

    # Paris in summer (CEST = UTC+2)
    eph_paris = compute_night_ephemeris(48.8566, 2.3522, "2026-06-21")
    assert eph_paris["tz_offset_hours"] == pytest.approx(2.0)

    # New Delhi year-round (IST = UTC+5.5)
    eph_delhi = compute_night_ephemeris(28.6139, 77.2090, "2026-10-15")
    assert eph_delhi["tz_offset_hours"] == pytest.approx(5.5)

    # Santiago, Chile in January (CLST = UTC-3)
    eph_santiago = compute_night_ephemeris(-33.4489, -70.6693, "2026-01-15")
    assert eph_santiago["tz_offset_hours"] == pytest.approx(-3.0)

    # Midnight Sun in Tromsø (+69.65°N) on summer solstice -> no astronomical darkness
    eph_tromso = compute_night_ephemeris(69.6492, 18.9553, "2026-06-21")
    assert eph_tromso["has_astronomical_darkness"] is False
    assert eph_tromso["astronomical_dark_hours"] == 0.0
    assert eph_tromso["min_sun_altitude_deg"] > 0.0


def test_vision_traversal_filename_key_validation_and_security_headers() -> None:
    """Verify safe handling of `filename='..'`, rejection of invalid API keys, CSV sanitization, and `nosniff`."""
    from astropilote.vision.visualize import _sanitize_csv_cell

    test_client = TestClient(app, raise_server_exceptions=False)

    # 1. Security headers on API & HTML responses
    r_health = test_client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.headers.get("X-Content-Type-Options") == "nosniff"

    # 2. Invalid / CRLF API key rejected with 400/422 (never 500 or saved)
    r_bad_key = test_client.post(
        "/api/config/key",
        json={"api_key": "sk-ant-12345678\nOLLAMA_BASE_URL=http://evil"},
    )
    assert r_bad_key.status_code in (400, 422)

    r_bad_prefix = test_client.post(
        "/api/config/key",
        json={"api_key": "invalid-prefix-key-12345"},
    )
    assert r_bad_prefix.status_code in (400, 422)

    # 3. `filename=".."` in `/api/vision/identify` never triggers 500 IsADirectoryError:
    #    - corrupt/non-image payload with `filename=".."` -> clean 400/422
    #    - valid PNG payload with `filename=".."` -> safely neutralized to `upload.png` (200) or rejected (400/422)
    r_dotdot_bad_bytes = test_client.post(
        "/api/vision/identify",
        json={
            "filename": "..",
            "image_base64": base64.b64encode(b"not-an-image").decode("ascii"),
            "ra_hint_deg": 10.6847,
            "dec_hint_deg": 41.2687,
            "field_width_deg": 4.0,
        },
    )
    assert r_dotdot_bad_bytes.status_code in (400, 422)

    r_dotdot_png = test_client.post(
        "/api/vision/identify",
        json={
            "filename": "..",
            "image_base64": _tiny_png_base64(),
            "ra_hint_deg": 10.6847,
            "dec_hint_deg": 41.2687,
            "field_width_deg": 4.0,
        },
    )
    assert r_dotdot_png.status_code in (200, 400, 422)

    r_bad_preset = test_client.post(
        "/api/vision/identify",
        json={"preset_id": "../../etc/passwd"},
    )
    assert r_bad_preset.status_code in (400, 404, 422)

    # 4. CSV formula injection neutralization in `visualize.py`
    assert _sanitize_csv_cell("=CMD|' /C calc'!A0") == "'=CMD|' /C calc'!A0"
    assert _sanitize_csv_cell(41.2687) == 41.2687


def test_eval_runner_defaults_offline_isolation_and_polarity_penalty() -> None:
    """Verify `eval/run_eval.py` S3.4 defaults, offline path isolation, polarity cap, and city|date fixtures."""
    import importlib.util
    from pathlib import Path

    from astropilote.tools.world import weather

    eval_path = Path(__file__).resolve().parents[1] / "eval" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("run_eval_mod", eval_path)
    assert spec is not None and spec.loader is not None
    run_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_eval)

    # 1. Default CLI flags pin REFERENCE_DATE = 2026-10-15 and strict_corpus_guard = True
    args = run_eval.parse_args([])
    assert run_eval.REFERENCE_DATE == "2026-10-15"
    assert args.reference_date == "2026-10-15"
    assert args.strict_corpus_guard is True
    assert run_eval.parse_args(["--no-strict-corpus-guard"]).strict_corpus_guard is False

    # 2. Offline mode never overwrites results.jsonl / summary_metrics.json
    _, res_off, sum_off = run_eval._resolve_output_paths(Path("eval"), "offline")
    assert res_off.name == "results_offline.jsonl"
    assert sum_off.name == "summary_metrics_offline.json"
    _, res_off_explicit, sum_off_explicit = run_eval._resolve_output_paths(
        Path("eval/results.jsonl"), "offline"
    )
    assert res_off_explicit.name == "results_offline.jsonl"
    assert sum_off_explicit.name == "summary_metrics_offline.json"

    # 3. Polarity contradiction severely capped <= 0.15 despite high lexical & citation overlap
    gold_never_rises = (
        "No. Omega Centauri [C080] at declination -47.5 degrees never rises from Paris "
        "and stays below the horizon all night; it cannot be observed from 48.9N."
    )
    pred_contradiction = (
        "Yes! Omega Centauri [C080] at declination -47.5 degrees is well-positioned from Paris "
        "above 35 degrees and can be observed all night from 48.9N."
    )
    contradiction_score = run_eval._score_judge_correctness(
        pred_answer=pred_contradiction,
        gold_answer=gold_never_rises,
        cited=["C080"],
        gold_docs=["C080"],
        stop_reason="completed",
    )
    assert contradiction_score <= 0.15

    # Consistent negative answer scores high
    pred_consistent = (
        "No: Omega Centauri [C080] at Dec -47.5° never rises from Paris (48.9°N) and "
        "remains below the horizon, so it cannot be observed."
    )
    consistent_score = run_eval._score_judge_correctness(
        pred_answer=pred_consistent,
        gold_answer=gold_never_rises,
        cited=["C080"],
        gold_docs=["C080"],
        stop_reason="completed",
    )
    assert consistent_score >= 0.85

    # 4. City-indexed weather fixtures ("marseille|2026-10-17" vs "paris|2026-10-17")
    wx_marseille = weather(43.2965, 5.3698, "2026-10-17", place_name="Marseille")
    wx_paris = weather(48.8566, 2.3522, "2026-10-17", place_name="Paris")
    assert wx_marseille["cloud_cover_pct"] == 8
    assert wx_paris["cloud_cover_pct"] == 10


def test_located_session_paris_works_in_agent_and_rag_modes() -> None:
    """Ensure explicit city queries like 'what should i look tonight at paris' geocode and schedule in both agent and rag modes."""
    import json

    test_client = TestClient(app)
    for exec_mode in ("agent", "rag"):
        resp = test_client.post(
            "/api/chat/stream",
            json={
                "question": "what should i look tonight at paris",
                "mode": exec_mode,
                "date_override": "2026-10-17",
            },
        )
        assert resp.status_code == 200
        events = [
            json.loads(line[6:])
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]
        done_ev = next(ev for ev in events if ev.get("event") == "done")
        assert done_ev["intent"]["location_name"] == "Paris"
        assert done_ev["answer"]["conditions"] is not None
        assert "Paris" in done_ev["answer"]["conditions"]
        assert len(done_ev["answer"]["plan"]) >= 1
        tool_names = [t["name"] for t in done_ev["tool_observations"]]
        assert "geocode" in tool_names
        assert "night" in tool_names
        assert "weather" in tool_names
        assert "visibility" in tool_names
