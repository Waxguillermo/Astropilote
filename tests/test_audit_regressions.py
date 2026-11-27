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


def test_versioned_claude_model_is_never_silently_downgraded() -> None:
    """A configured `claude-<family>-<major>.<minor>` must reach each provider under its own slug."""
    from astropilote.tracking.tracker import compute_cost_usd

    for configured in ("claude-sonnet-5.5", "claude-sonnet-5-5"):
        assert _resolve_anthropic_direct_model(configured) == "claude-sonnet-5-5"
        assert _resolve_openrouter_model(configured) == "anthropic/claude-sonnet-5.5"
    assert _resolve_openrouter_model("anthropic/claude-sonnet-5.5") == "anthropic/claude-sonnet-5.5"

    # List prices per million input / output tokens: Sonnet 5.5 $2 / $10, Haiku 4.5 $1 / $5
    for backend in ("claude-sonnet-5-5", "anthropic/claude-sonnet-5.5"):
        assert compute_cost_usd(backend, 1_000_000, 1_000_000) == 12.0
    for backend in ("claude-haiku-4-5-20251001", "anthropic/claude-haiku-4.5"):
        assert compute_cost_usd(backend, 1_000_000, 1_000_000) == 6.0


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


def _load_run_eval():
    import importlib.util
    from pathlib import Path

    eval_path = Path(__file__).resolve().parents[1] / "eval" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("run_eval_mod", eval_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_runner_defaults_offline_isolation_and_polarity_penalty() -> None:
    """Verify `eval/run_eval.py` S3.4 defaults, offline path isolation, polarity cap, and city|date fixtures."""
    from pathlib import Path

    from astropilote.tools.world import weather

    run_eval = _load_run_eval()

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
            json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")
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


def test_eval_runner_refuses_to_run_without_the_requested_backend(monkeypatch) -> None:
    """A results file must only ever be produced by the model its summary names."""
    run_eval = _load_run_eval()
    api_status = {
        "anthropic_ready": True,
        "anthropic_model": "claude-sonnet-5-5",
        "router_model": "claude-haiku-4-5-20251001",
    }
    monkeypatch.setattr(run_eval.settings, "ollama_model", "gemma4")
    monkeypatch.setattr(
        run_eval.settings, "anthropic_api_key", "sk-ant-api03-validtestkey1234567890"
    )

    # Another tag of the same family is not the configured model.
    for installed in ([], ["gemma4:e2b"]):
        monkeypatch.setattr(
            run_eval, "check_llm_status", lambda tags=installed: {"ollama_installed_models": tags}
        )
        with pytest.raises(SystemExit, match="ollama pull gemma4"):
            run_eval._require_backend("local")

    monkeypatch.setattr(
        run_eval, "check_llm_status", lambda: {"ollama_installed_models": ["gemma4:latest"]}
    )
    assert run_eval._require_backend("local") == {
        "model": "ollama/gemma4",
        "router_model": "ollama/gemma4",
        "provider": "Ollama",
    }

    monkeypatch.setattr(run_eval, "check_llm_status", lambda: api_status)
    assert run_eval._require_backend("api") == {
        "model": "claude-sonnet-5-5",
        "router_model": "claude-haiku-4-5-20251001",
        "provider": "Anthropic API",
    }
    monkeypatch.setattr(run_eval, "check_llm_status", lambda: {"anthropic_ready": False})
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        run_eval._require_backend("api")


def test_llm_comparison_only_pairs_runs_of_the_same_campaign(tmp_path) -> None:
    """Each side keeps the model its run recorded; unattributed or mismatched runs are not paired."""
    import json

    run_eval = _load_run_eval()
    summary = {
        "questions": "eval/golden.jsonl",
        "reference_date": "2026-10-15",
        "strict_corpus_guard": True,
        "total_questions": 69,
        "answerable_questions": 48,
        "unanswerable_or_trap_questions": 21,
        "retrieval_recall": 0.99,
        "retrieval_mrr": 0.89,
        "prudence_abstention_refusal_accuracy": 1.0,
        "judged_correctness": 0.9,
        "latency_median_s": 6.4,
        "latency_p95_s": 9.5,
        "avg_tokens_in": 1397,
        "avg_tokens_out": 301,
        "total_cost_usd": 0.69,
    }
    api_path = tmp_path / "summary_metrics.json"
    local_path = tmp_path / "summary_metrics_local.json"
    comparison_path = tmp_path / "llm_comparison.json"

    def write(path, **fields) -> None:
        path.write_text(json.dumps({**summary, **fields}))

    # A legacy summary that never recorded its model is not attributed to the current settings.
    write(api_path)
    write(local_path, model="ollama/gemma4", total_cost_usd=0)
    run_eval._write_llm_comparison(tmp_path)
    assert not comparison_path.exists()

    # Runs anchored on different dates are different campaigns.
    write(api_path, model="claude-sonnet-5-5", reference_date="2026-11-01")
    run_eval._write_llm_comparison(tmp_path)
    assert not comparison_path.exists()

    write(api_path, model="claude-sonnet-5-5", provider="Anthropic API")
    run_eval._write_llm_comparison(tmp_path)
    models = json.loads(comparison_path.read_text())["models"]
    assert models["api"]["model_name"] == "claude-sonnet-5-5"
    assert models["api"]["cost_per_query_usd"] == 0.01
    assert models["local"]["model_name"] == "ollama/gemma4"
    assert models["local"]["prudence_accuracy"] == 1.0


def _isolate_eval_settings(monkeypatch, run_eval) -> None:
    """`run_eval.main()` mutates global settings: make pytest restore them afterwards."""
    for name in (
        "llm_backend_fallback",
        "reference_date",
        "strict_corpus_guard",
        "eval_offline_mode",
        "ollama_model",
        "anthropic_api_key",
    ):
        monkeypatch.setattr(run_eval.settings, name, getattr(run_eval.settings, name))
    monkeypatch.setattr(run_eval, "_refresh_env_settings", lambda: None)


def test_eval_main_attributes_golden_results_to_the_backend_that_ran(tmp_path, monkeypatch) -> None:
    import json
    import sys

    from astropilote.agent.schemas import EvalResultRow

    run_eval = _load_run_eval()
    _isolate_eval_settings(monkeypatch, run_eval)
    monkeypatch.setattr(run_eval.settings, "ollama_model", "gemma4")
    monkeypatch.setattr(
        run_eval, "check_llm_status", lambda: {"ollama_installed_models": ["gemma4:latest"]}
    )
    questions = tmp_path / "golden.jsonl"
    questions.write_text(
        json.dumps(
            {"id": "q1", "question": "Where is M31?", "gold_docs": ["M031"], "answer": "Andromeda"}
        )
        + "\n"
    )
    seen: dict = {}

    def fake_run_agent(question: str, **kwargs):
        seen.update(kwargs, fallback=run_eval.settings.llm_backend_fallback)
        row = EvalResultRow(
            id="q1",
            answer="[M031] lies in Andromeda.",
            cited_sources=["M031"],
            retrieved_ids=["M031"],
            tokens_in=10,
            tokens_out=5,
            latency_s=0.5,
            tool_calls=[],
            stop_reason="answered",
        )
        return None, None, row

    monkeypatch.setattr(run_eval, "run_agent", fake_run_agent)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval.py",
            "--questions",
            str(questions),
            "--out",
            str(out_dir),
            "--llm-mode",
            "local",
        ],
    )
    run_eval.main()

    assert seen["llm_mode"] == "local"
    assert seen["fallback"] is False
    summary = json.loads((out_dir / "summary_metrics_local.json").read_text())
    assert summary["model"] == "ollama/gemma4"
    assert summary["provider"] == "Ollama"
    assert summary["mode"] == "local"
    assert summary["questions"] == str(questions)
    assert summary["retrieval_recall"] == 1.0
    assert (out_dir / "results_local.jsonl").read_text().count("\n") == 1


def test_eval_main_preflights_the_scenario_mode_it_resolved(tmp_path, monkeypatch) -> None:
    """Without `--llm-mode`, a configured key means a paid API run: it is checked and attributed."""
    import sys
    from datetime import date

    import astropilote.agentic.scenario_eval as scenario_eval

    run_eval = _load_run_eval()
    _isolate_eval_settings(monkeypatch, run_eval)
    captured: dict = {}
    monkeypatch.setattr(
        scenario_eval,
        "evaluate_scenarios",
        lambda path, out, **kwargs: captured.update(kwargs) or {"failed": 0},
    )
    monkeypatch.setattr(
        run_eval,
        "check_llm_status",
        lambda: {
            "anthropic_ready": True,
            "anthropic_model": "claude-sonnet-5-5",
            "router_model": "claude-sonnet-5-5",
        },
    )
    monkeypatch.setattr(
        sys, "argv", ["run_eval.py", "--scenarios", "eval/scenarios.jsonl", "--out", str(tmp_path)]
    )

    monkeypatch.setattr(
        run_eval.settings, "anthropic_api_key", "sk-ant-api03-validtestkey1234567890"
    )
    run_eval.main()
    assert captured["llm_mode"] == "api"
    assert captured["backend"]["model"] == "claude-sonnet-5-5"

    monkeypatch.setattr(run_eval.settings, "anthropic_api_key", "")
    run_eval.main()
    assert captured["llm_mode"] == "offline"
    assert captured["backend"] == {
        "model": None,
        "provider": "rule-based controller",
        "generated_on": date.today().isoformat(),
    }


def test_backend_fallback_can_be_disabled_for_attributable_runs(monkeypatch) -> None:
    """With fallback off, a local-mode call is never answered (and billed) by the cloud API."""
    import httpx

    api_calls: list[str] = []

    def ollama_down(*_args, **_kwargs):
        raise httpx.ConnectError("daemon down")

    def api_answer(*_args, **_kwargs):
        api_calls.append("api")
        return "Answer from the API", 10, 5, "claude-sonnet-5-5"

    monkeypatch.setattr(
        llm_mod.settings, "anthropic_api_key", "sk-ant-api03-validtestkey1234567890"
    )
    monkeypatch.setattr(llm_mod, "_call_ollama_llm", ollama_down)
    monkeypatch.setattr(llm_mod, "_call_api_llm", api_answer)

    *_, backend = llm_mod.generate_grounded_answer("system", "user", mode_override="local")
    assert backend == "claude-sonnet-5-5"

    monkeypatch.setattr(llm_mod.settings, "llm_backend_fallback", False)
    with pytest.raises(llm_mod.LLMUnavailableError):
        llm_mod.generate_grounded_answer("system", "user", mode_override="local")
    assert api_calls == ["api"]


@pytest.mark.parametrize(
    ("local_fn", "api_fn", "call"),
    [
        (
            "_call_ollama_router",
            "_call_api_router",
            lambda: llm_mod.route_query_with_llm("q", "2026-11-14", "Saturday", "local"),
        ),
        (
            "_call_ollama_react_planner",
            "_call_api_structured_tool",
            lambda: llm_mod.plan_react_step_with_llm({}, mode_override="local"),
        ),
        (
            "_stream_ollama_llm",
            "_stream_api_llm",
            lambda: list(llm_mod.stream_grounded_answer("system", "user", mode_override="local")),
        ),
    ],
)
def test_router_planner_and_stream_honour_the_disabled_fallback(
    monkeypatch, local_fn, api_fn, call
) -> None:
    """Routing, planning and streamed answers of a local run are never served by the cloud API."""
    import httpx

    def ollama_down(*_args, **_kwargs):
        raise httpx.ConnectError("daemon down")

    def api_must_not_answer(*_args, **_kwargs):
        raise AssertionError("the cloud API served a local-mode call")

    monkeypatch.setattr(
        llm_mod.settings, "anthropic_api_key", "sk-ant-api03-validtestkey1234567890"
    )
    monkeypatch.setattr(llm_mod.settings, "llm_backend_fallback", False)
    monkeypatch.setattr(llm_mod, local_fn, ollama_down)
    monkeypatch.setattr(llm_mod, api_fn, api_must_not_answer)
    with pytest.raises(llm_mod.LLMUnavailableError):
        call()


def test_structured_calls_can_be_pinned_to_the_selected_backend(monkeypatch) -> None:
    """An acting agent passes `allow_fallback=False`: Ollama being down never reaches the API."""
    import httpx

    api_calls: list[str] = []

    def ollama_down(*_args, **_kwargs):
        raise httpx.ConnectError("daemon down")

    def api_plan(**_kwargs):
        api_calls.append("api")
        return {"decision": "finish"}, 10, 5, "claude-sonnet-5-5"

    monkeypatch.setattr(
        llm_mod.settings, "anthropic_api_key", "sk-ant-api03-validtestkey1234567890"
    )
    monkeypatch.setattr(llm_mod.settings, "llm_backend_fallback", True)
    monkeypatch.setattr(llm_mod, "_iter_ollama_ndjson", ollama_down)
    monkeypatch.setattr(llm_mod, "_call_api_structured_tool", api_plan)
    request = {
        "user_msg": "u",
        "sys_msg": "s",
        "tool_name": "t",
        "tool_description": "d",
        "tool_schema": {},
        "ollama_fields_hint": "",
        "mode_override": "local",
    }

    assert llm_mod._backend_allowed("api", "local") is True
    assert llm_mod._backend_allowed("api", "local", allow_fallback=False) is False
    assert llm_mod._backend_allowed("local", "local", allow_fallback=False) is True

    assert llm_mod.call_structured_llm(**request)[3] == "claude-sonnet-5-5"
    with pytest.raises(llm_mod.LLMUnavailableError):
        llm_mod.call_structured_llm(**request, allow_fallback=False)
    assert api_calls == ["api"]
