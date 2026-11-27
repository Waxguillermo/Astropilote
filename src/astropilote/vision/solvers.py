"""Blind plate-solving adapters: local ASTAP CLI and cloud `nova.astrometry.net` API.

Provides a clean 3-tier resolution chain for Sky Vision:
    1. Local `astap_cli` / `astap` (fast ~1-3s, offline, compact star database)
    2. Cloud `nova.astrometry.net` API (zero local install, SHA-256 cached)
    3. Manual gnomonic TAN WCS fallback in `service.py`
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from astropy.io import fits
from astropy.wcs import WCS

from astropilote.config import settings
from astropilote.vision.image_catalog import SolverFailureError

ASTROMETRY_NET_API_BASE = "https://nova.astrometry.net/api"


@dataclass(frozen=True)
class BlindSolveResult:
    """Normalized astrometric solution returned by ASTAP or `nova.astrometry.net`."""

    method: str  # "astap_local" | "nova_astrometry_net"
    ra_deg: float
    dec_deg: float
    pixel_scale_arcsec: float
    orientation_deg: float = 0.0
    parity: float = 1.0
    wcs_header_text: str | None = None
    cached: bool = False


def find_astap_binary() -> str | None:
    """Return path to `astap_cli` or `astap` in PATH, `data/astap/`, or standard macOS/Linux locations."""
    from_path = shutil.which("astap_cli") or shutil.which("astap")
    if from_path:
        return from_path
    candidates = [
        settings.data_dir / "astap" / "astap_cli",
        settings.data_dir / "astap" / "astap",
        Path("/Applications/ASTAP.app/Contents/MacOS/astap"),
        Path("/usr/local/bin/astap_cli"),
        Path("/opt/homebrew/bin/astap_cli"),
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def find_astap_database_dir() -> Path | None:
    """Return directory containing ASTAP star database files (`*.290` / `*.1476`), if present."""
    candidates = [
        settings.data_dir / "astap",
        Path("/usr/local/opt/astap"),
        Path("/opt/astap"),
        Path("/Applications/ASTAP.app/Contents/MacOS"),
    ]
    for folder in candidates:
        if folder.is_dir() and (any(folder.glob("*.290")) or any(folder.glob("*.1476"))):
            return folder
    return None


def detect_available_solvers(astrometry_api_key: str | None = None) -> dict[str, bool]:
    """Report which blind plate-solving backends are currently available."""
    key = (astrometry_api_key or settings.astrometry_api_key or "").strip()
    return {
        "astap": find_astap_binary() is not None and find_astap_database_dir() is not None,
        "solve_field": shutil.which("solve-field") is not None,
        "nova_astrometry_net": bool(key),
    }


def _parse_wcs_header_file(wcs_path: Path) -> BlindSolveResult:
    """Parse an ASTAP `.wcs` FITS-header or text file into a `BlindSolveResult`."""
    raw_text = wcs_path.read_text(encoding="latin-1", errors="ignore")
    if "\n" in raw_text:
        header = fits.Header.fromstring(raw_text, sep="\n")
    else:
        header = fits.Header.fromstring(raw_text)
    if "CRVAL1" not in header or "CRVAL2" not in header:
        raise SolverFailureError("ASTAP .wcs output did not contain CRVAL1/CRVAL2.")

    ra = float(header["CRVAL1"]) % 360.0
    dec = float(header["CRVAL2"])
    # Derive pixel scale (arcsec/px) from CD matrix or CDELT
    if all(k in header for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")):
        cd11, cd12 = float(header["CD1_1"]), float(header["CD1_2"])
        cd21, cd22 = float(header["CD2_1"]), float(header["CD2_2"])
        det = abs(cd11 * cd22 - cd12 * cd21)
        scale_deg = math.sqrt(det) if det > 0 else abs(float(header.get("CDELT1", 0.001)))
        orientation = math.degrees(math.atan2(cd12, cd22))
        parity = -1.0 if (cd11 * cd22 - cd12 * cd21) < 0 else 1.0
    else:
        scale_deg = abs(float(header.get("CDELT1", 0.001)))
        orientation = float(header.get("CROTA2", 0.0))
        parity = 1.0

    scale_arcsec = scale_deg * 3600.0
    if not (math.isfinite(scale_arcsec) and scale_arcsec > 0):
        raise SolverFailureError("ASTAP produced an invalid pixel scale.")

    return BlindSolveResult(
        method="astap_local",
        ra_deg=ra,
        dec_deg=dec,
        pixel_scale_arcsec=scale_arcsec,
        orientation_deg=orientation,
        parity=parity,
        wcs_header_text=raw_text,
    )


def solve_with_astap(
    image_path: Path,
    *,
    fov_deg: float | None = None,
    ra_hint_deg: float | None = None,
    dec_hint_deg: float | None = None,
    timeout_s: float = 60.0,
) -> BlindSolveResult:
    """Run local `astap_cli`/`astap` to blind-solve `image_path` into a WCS solution."""
    binary = find_astap_binary()
    if not binary:
        raise SolverFailureError("ASTAP CLI binary (`astap_cli` or `astap`) is not installed.")

    db_dir = find_astap_database_dir()
    with tempfile.TemporaryDirectory(prefix="astropilote-astap-") as tmpdir:
        out_base = Path(tmpdir) / "solved"
        fov_arg = (
            f"{float(fov_deg):.3f}"
            if (fov_deg is not None and math.isfinite(fov_deg) and fov_deg > 0)
            else "0"
        )
        cmd = [
            binary,
            "-f",
            str(image_path),
            "-o",
            str(out_base),
            "-r",
            "180",
            "-fov",
            fov_arg,
            "-z",
            "0",
            "-wcs",
        ]
        if db_dir is not None:
            cmd.extend(["-d", str(db_dir)])
        if ra_hint_deg is not None and dec_hint_deg is not None:
            cmd.extend(["-ra", f"{(float(ra_hint_deg) % 360.0) / 15.0:.4f}"])
            cmd.extend(["-spd", f"{float(dec_hint_deg) + 90.0:.4f}"])

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SolverFailureError(f"ASTAP timed out after {timeout_s:.0f}s.") from exc
        except OSError as exc:
            raise SolverFailureError(f"Failed to invoke ASTAP: {exc}") from exc

        wcs_path = out_base.with_suffix(".wcs")
        if proc.returncode != 0 or not wcs_path.exists():
            detail = (proc.stderr or proc.stdout or "").strip()[:240]
            raise SolverFailureError(f"ASTAP did not converge (exit={proc.returncode}): {detail}")

        return _parse_wcs_header_file(wcs_path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def solve_with_astrometry_net(
    image_path: Path,
    *,
    api_key: str,
    poll_interval_s: float = 2.5,
    timeout_s: float = 90.0,
    cache_dir: Path | None = None,
    client: httpx.Client | None = None,
) -> BlindSolveResult:
    """Blind-solve `image_path` via `nova.astrometry.net` API with SHA-256 disk caching."""
    clean_key = api_key.strip()
    if not clean_key:
        raise SolverFailureError("An Astrometry.net API key is required for cloud plate-solving.")

    resolved_cache = cache_dir or (settings.data_dir / "vision_cache")
    resolved_cache.mkdir(parents=True, exist_ok=True)
    digest = _sha256_file(image_path)
    cache_file = resolved_cache / f"{digest}.json"
    if cache_file.exists():
        try:
            cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
            return BlindSolveResult(
                method="nova_astrometry_net",
                ra_deg=float(cached_data["ra_deg"]),
                dec_deg=float(cached_data["dec_deg"]),
                pixel_scale_arcsec=float(cached_data["pixel_scale_arcsec"]),
                orientation_deg=float(cached_data.get("orientation_deg", 0.0)),
                parity=float(cached_data.get("parity", 1.0)),
                cached=True,
            )
        except (ValueError, KeyError, OSError):
            pass

    owns_client = client is None
    http = client or httpx.Client(timeout=30.0)
    deadline = time.monotonic() + timeout_s
    try:
        # 1. Login
        login_resp = http.post(
            f"{ASTROMETRY_NET_API_BASE}/login",
            data={"request-json": json.dumps({"apikey": clean_key})},
        )
        login_resp.raise_for_status()
        login_json = login_resp.json()
        if login_json.get("status") != "success" or not login_json.get("session"):
            raise SolverFailureError("Astrometry.net login failed (check ASTROMETRY_API_KEY).")
        session_key = str(login_json["session"])

        # 2. Upload image
        upload_payload = {
            "session": session_key,
            "publicly_visible": "n",
            "allow_commercial_use": "n",
            "allow_modifications": "n",
        }
        with image_path.open("rb") as img_fp:
            up_resp = http.post(
                f"{ASTROMETRY_NET_API_BASE}/upload",
                data={"request-json": json.dumps(upload_payload)},
                files={"file": (image_path.name, img_fp, "application/octet-stream")},
            )
        up_resp.raise_for_status()
        up_json = up_resp.json()
        if up_json.get("status") != "success" or not up_json.get("subid"):
            raise SolverFailureError("Astrometry.net image upload was rejected.")
        sub_id = up_json["subid"]

        # 3. Poll submission -> job_id
        job_id: int | None = None
        while time.monotonic() < deadline:
            sub_resp = http.get(f"{ASTROMETRY_NET_API_BASE}/submissions/{sub_id}")
            sub_resp.raise_for_status()
            jobs = sub_resp.json().get("jobs") or []
            valid_jobs = [j for j in jobs if j is not None]
            if valid_jobs:
                job_id = int(valid_jobs[0])
                break
            time.sleep(poll_interval_s)

        if job_id is None:
            raise SolverFailureError("Astrometry.net submission timed out waiting for a job.")

        # 4. Poll job status -> success
        while time.monotonic() < deadline:
            job_resp = http.get(f"{ASTROMETRY_NET_API_BASE}/jobs/{job_id}")
            job_resp.raise_for_status()
            status = str(job_resp.json().get("status") or "").lower()
            if status == "success":
                break
            if status == "failure":
                raise SolverFailureError(
                    "nova.astrometry.net could not match star fields in this photo."
                )
            time.sleep(poll_interval_s)
        else:
            raise SolverFailureError(f"nova.astrometry.net job {job_id} timed out.")

        # 5. Fetch calibration
        cal_resp = http.get(f"{ASTROMETRY_NET_API_BASE}/jobs/{job_id}/calibration/")
        cal_resp.raise_for_status()
        cal = cal_resp.json()
        result = BlindSolveResult(
            method="nova_astrometry_net",
            ra_deg=float(cal["ra"]) % 360.0,
            dec_deg=float(cal["dec"]),
            pixel_scale_arcsec=float(cal["pixscale"]),
            orientation_deg=float(cal.get("orientation", 0.0)),
            parity=float(cal.get("parity", 1.0)),
            cached=False,
        )
        cache_file.write_text(
            json.dumps(
                {
                    "ra_deg": result.ra_deg,
                    "dec_deg": result.dec_deg,
                    "pixel_scale_arcsec": result.pixel_scale_arcsec,
                    "orientation_deg": result.orientation_deg,
                    "parity": result.parity,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return result
    except httpx.HTTPError as exc:
        raise SolverFailureError(f"nova.astrometry.net HTTP error: {exc}") from exc
    finally:
        if owns_client:
            http.close()


def build_wcs_from_blind_solution(
    sol: BlindSolveResult,
    *,
    width: int,
    height: int,
) -> WCS:
    """Build an `astropy.wcs.WCS` from an ASTAP `.wcs` header or `nova.astrometry.net` calibration."""
    if sol.wcs_header_text:
        try:
            header = (
                fits.Header.fromstring(sol.wcs_header_text, sep="\n")
                if "\n" in sol.wcs_header_text
                else fits.Header.fromstring(sol.wcs_header_text)
            )
            return WCS(header).celestial
        except (ValueError, KeyError, OSError):
            pass

    scale_deg = sol.pixel_scale_arcsec / 3600.0
    theta = math.radians(sol.orientation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    parity = -1.0 if sol.parity < 0 else 1.0

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [(width + 1) / 2.0, (height + 1) / 2.0]
    w.wcs.crval = [sol.ra_deg, sol.dec_deg]
    w.wcs.cd = [
        [-parity * scale_deg * cos_t, scale_deg * sin_t],
        [parity * scale_deg * sin_t, scale_deg * cos_t],
    ]
    return w


def run_blind_solver_chain(
    image_path: Path,
    *,
    astrometry_api_key: str | None = None,
    fov_deg: float | None = None,
    ra_hint_deg: float | None = None,
    dec_hint_deg: float | None = None,
) -> tuple[BlindSolveResult | None, list[str]]:
    """Try ASTAP locally first, then `nova.astrometry.net` if configured; collect non-fatal warnings."""
    warnings: list[str] = []
    if find_astap_binary() is not None and find_astap_database_dir() is not None:
        try:
            return (
                solve_with_astap(
                    image_path,
                    fov_deg=fov_deg,
                    ra_hint_deg=ra_hint_deg,
                    dec_hint_deg=dec_hint_deg,
                ),
                warnings,
            )
        except SolverFailureError as exc:
            warnings.append(f"Local ASTAP did not converge: {exc}")

    key = (astrometry_api_key or settings.astrometry_api_key or "").strip()
    if key:
        try:
            return solve_with_astrometry_net(image_path, api_key=key), warnings
        except SolverFailureError as exc:
            warnings.append(f"nova.astrometry.net did not converge: {exc}")

    return None, warnings
