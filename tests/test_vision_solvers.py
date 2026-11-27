"""Tests for Sprint 4: ASTAP local solver, nova.astrometry.net API + SHA-256 cache, and 3-tier Vision fallback."""

from __future__ import annotations

import subprocess
from pathlib import Path

import httpx
from PIL import Image

import astropilote.vision.solvers as solvers_mod
from astropilote.vision.service import run_vision_pipeline
from astropilote.vision.solvers import (
    BlindSolveResult,
    build_wcs_from_blind_solution,
    detect_available_solvers,
    solve_with_astap,
    solve_with_astrometry_net,
)


def _make_sky_png(path: Path, width: int = 400, height: int = 300) -> Path:
    im = Image.new("RGB", (width, height), color=(5, 8, 16))
    im.save(path, format="PNG")
    return path


def test_solve_with_astap_parses_wcs_header_cleanly(tmp_path: Path, monkeypatch) -> None:
    img = _make_sky_png(tmp_path / "m31.png")
    monkeypatch.setattr(solvers_mod, "find_astap_binary", lambda: "/usr/local/bin/astap_cli")
    monkeypatch.setattr(solvers_mod, "find_astap_database_dir", lambda: tmp_path)

    def _fake_run(cmd, **_kw):
        assert "-wcs" in cmd
        assert "-fov" in cmd and cmd[cmd.index("-fov") + 1] == "0"
        assert "-d" in cmd and cmd[cmd.index("-d") + 1] == str(tmp_path)
        out_idx = cmd.index("-o") + 1
        wcs_file = Path(cmd[out_idx]).with_suffix(".wcs")
        wcs_file.write_text(
            "SIMPLE  = T\n"
            "NAXIS   = 2\n"
            "CTYPE1  = 'RA---TAN'\n"
            "CTYPE2  = 'DEC--TAN'\n"
            "CRPIX1  = 200.5\n"
            "CRPIX2  = 150.5\n"
            "CRVAL1  = 10.6847\n"
            "CRVAL2  = 41.2687\n"
            "CD1_1   = -0.0125\n"
            "CD1_2   = 0.0\n"
            "CD2_1   = 0.0\n"
            "CD2_2   = 0.0125\n"
            "END\n",
            encoding="ascii",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="Solved", stderr="")

    monkeypatch.setattr(solvers_mod.subprocess, "run", _fake_run)
    sol = solve_with_astap(img)
    assert sol.method == "astap_local"
    assert abs(sol.ra_deg - 10.6847) < 1e-4
    assert abs(sol.dec_deg - 41.2687) < 1e-4
    assert abs(sol.pixel_scale_arcsec - 45.0) < 0.1

    wcs = build_wcs_from_blind_solution(sol, width=400, height=300)
    x, y = wcs.world_to_pixel_values(10.6847, 41.2687)
    assert abs(float(x) - 199.5) < 1.0 and abs(float(y) - 149.5) < 1.0


def test_solve_with_astrometry_net_polls_and_caches_by_sha256(tmp_path: Path) -> None:
    img = _make_sky_png(tmp_path / "orion.png")
    cache_dir = tmp_path / "cache"
    call_counts = {"login": 0, "upload": 0, "cal": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        p = request.url.path
        if p.endswith("/login"):
            call_counts["login"] += 1
            return httpx.Response(200, json={"status": "success", "session": "sess-xyz"})
        if p.endswith("/upload"):
            call_counts["upload"] += 1
            return httpx.Response(200, json={"status": "success", "subid": 101})
        if p.endswith("/submissions/101"):
            return httpx.Response(200, json={"jobs": [202]})
        if p.endswith("/jobs/202"):
            return httpx.Response(200, json={"status": "success"})
        if p.endswith("/jobs/202/calibration/"):
            call_counts["cal"] += 1
            return httpx.Response(
                200,
                json={
                    "ra": 83.8221,
                    "dec": -5.3911,
                    "pixscale": 36.0,
                    "orientation": 12.5,
                    "parity": 1.0,
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(_handler)
    with httpx.Client(transport=transport) as client:
        sol1 = solve_with_astrometry_net(
            img, api_key="test-key", poll_interval_s=0.0, cache_dir=cache_dir, client=client
        )
        sol2 = solve_with_astrometry_net(
            img, api_key="test-key", poll_interval_s=0.0, cache_dir=cache_dir, client=client
        )

    assert sol1.method == "nova_astrometry_net" and sol1.cached is False
    assert abs(sol1.ra_deg - 83.8221) < 1e-4
    assert sol2.cached is True
    assert call_counts == {"login": 1, "upload": 1, "cal": 1}


def test_run_vision_pipeline_uses_blind_solver_without_manual_ra_dec(
    tmp_path: Path, monkeypatch
) -> None:
    img = _make_sky_png(tmp_path / "blind_m31.png", width=800, height=600)
    monkeypatch.setattr(
        "astropilote.vision.service.run_blind_solver_chain",
        lambda *_a, **_kw: (
            BlindSolveResult(
                method="astap_local",
                ra_deg=10.6847,
                dec_deg=41.2687,
                pixel_scale_arcsec=22.95,
            ),
            [],
        ),
    )
    res = run_vision_pipeline(img, metadata=None, is_preset=False)
    payload = res["result"]
    assert payload["astrometry"]["method"] == "astap_local"
    matched_ids = {obj["doc_id"] for obj in payload["objects"]}
    assert {"M031", "M032", "M110"}.issubset(matched_ids)
    assert detect_available_solvers(astrometry_api_key="abc")["nova_astrometry_net"] is True
