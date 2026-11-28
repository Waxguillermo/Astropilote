"""Web & API service bridge for the Sky Vision module (`image_catalog.py` & `visualize.py`).

Uses `astropilote.vision.image_catalog` and `astropilote.vision.visualize` directly.
When `solve-field` (Astrometry.net local binary + index files) is unavailable on the host machine,
provides an exact spherical gnomonic (`RA---TAN`, `DEC--TAN`) WCS projection backed by optical and
pointing metadata so the web UI works out-of-the-box on any machine.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from astropilote.vision import image_catalog as _ic
from astropilote.vision.image_catalog import (
    DEFAULT_INDEXES,
    DEFAULT_ROOT,
    MAX_BYTES,
    MAX_PIXELS,
    PhotoError,
    PhotoMetadata,
    SolverFailureError,
    check_dimensions,
    discover_chunks,
    inside_image,
    load_catalog,
)
from astropilote.vision.visualize import visualize

HERE = Path(__file__).resolve().parent

Image.MAX_IMAGE_PIXELS = MAX_PIXELS

# ============================================================================
# Curated Sky Photo Presets for 1-Click Testing in the Web Frontend
# ============================================================================
VISION_PRESETS: list[dict[str, Any]] = [
    {
        "id": "andromeda_widefield",
        "label": "Andromeda System (M31 · M32 · M110)",
        "subtitle": "RedCat 51 (250 mm) · APS-C · 5.10° wide · 3 galaxies",
        "badge": "3 Galaxies",
        "ra_hint_deg": 10.6847,
        "dec_hint_deg": 41.2687,
        "focal_length_mm": 250.0,
        "sensor_width_mm": 22.3,
        "field_width_deg": 5.10,
        "telescope": "William Optics RedCat 51 (250mm f/4.9)",
        "camera": "ZWO ASI2600MC Pro (APS-C)",
        "exposure_seconds": 180.0,
        "expected_objects": ["M031", "M032", "M110"],
    },
    {
        "id": "orion_sword",
        "label": "Orion's Sword Complex (M42 · M43 · Running Man)",
        "subtitle": "72ED APO (420 mm) · Full Frame · 4.90° wide · 6 objects",
        "badge": "6 Objects",
        "ra_hint_deg": 83.8187,
        "dec_hint_deg": -5.2000,
        "focal_length_mm": 420.0,
        "sensor_width_mm": 36.0,
        "field_width_deg": 4.90,
        "telescope": "Sky-Watcher Evostar 72ED (420mm f/5.8)",
        "camera": "ZWO ASI6200MC Pro (Full Frame)",
        "exposure_seconds": 120.0,
        "expected_objects": ["M042", "M043", "NGC1973", "NGC1975", "NGC1980", "NGC1981"],
    },
    {
        "id": "sagittarius_lagoon_trifid",
        "label": "Lagoon & Trifid Wide-Field (M8 · M20 · M21)",
        "subtitle": "FRA300 Pro (300 mm) · APS-C · 4.48° wide · 5 objects",
        "badge": "5 Objects",
        "ra_hint_deg": 271.1000,
        "dec_hint_deg": -23.5000,
        "focal_length_mm": 300.0,
        "sensor_width_mm": 23.5,
        "field_width_deg": 4.48,
        "telescope": "Askar FRA300 Pro (300mm f/5)",
        "camera": "Sony A6400 Astro-Modified (APS-C)",
        "exposure_seconds": 240.0,
        "expected_objects": ["M008", "M020", "M021", "NGC6530", "NGC6546"],
    },
    {
        "id": "bodes_cigar_galaxies",
        "label": "Bode's & Cigar Galaxies (M81 · M82)",
        "subtitle": "Evostar 80ED (600 mm) · APS-C · 2.13° wide · 2 galaxies",
        "badge": "2 Galaxies",
        "ra_hint_deg": 148.9290,
        "dec_hint_deg": 69.3720,
        "focal_length_mm": 600.0,
        "sensor_width_mm": 22.3,
        "field_width_deg": 2.13,
        "telescope": "Sky-Watcher Evostar 80ED (600mm f/7.5)",
        "camera": "Canon EOS R7 (APS-C)",
        "exposure_seconds": 300.0,
        "expected_objects": ["M081", "M082"],
    },
]


# ============================================================================
# Exact Spherical Gnomonic (TAN) WCS Projection
# ============================================================================
def gnomonic_sky_to_pixel(
    ra_deg: float,
    dec_deg: float,
    ra0_deg: float,
    dec0_deg: float,
    cx: float,
    cy: float,
    cdelt1_deg: float,
    cdelt2_deg: float,
) -> tuple[float, float] | None:
    """Project (ra_deg, dec_deg) onto 0-based pixel coordinates (x, y) via TAN projection."""
    if not (
        math.isfinite(cdelt1_deg)
        and abs(cdelt1_deg) > 1e-10
        and math.isfinite(cdelt2_deg)
        and abs(cdelt2_deg) > 1e-10
    ):
        raise PhotoError(
            "Degenerate WCS pixel scale (CDELT1 and CDELT2 must be finite and non-zero)."
        )

    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    ra0 = math.radians(ra0_deg)
    dec0 = math.radians(dec0_deg)

    dra = ra - ra0
    cos_c = math.sin(dec0) * math.sin(dec) + math.cos(dec0) * math.cos(dec) * math.cos(dra)
    if cos_c <= 1e-7:
        return None

    xi_deg = math.degrees((math.cos(dec) * math.sin(dra)) / cos_c)
    eta_deg = math.degrees(
        (math.cos(dec0) * math.sin(dec) - math.sin(dec0) * math.cos(dec) * math.cos(dra)) / cos_c
    )
    x = cx + xi_deg / cdelt1_deg
    y = cy + eta_deg / cdelt2_deg
    return x, y


def gnomonic_pixel_to_sky(
    x: float,
    y: float,
    ra0_deg: float,
    dec0_deg: float,
    cx: float,
    cy: float,
    cdelt1_deg: float,
    cdelt2_deg: float,
) -> tuple[float, float]:
    """Invert 0-based pixel (x, y) back to (ra_deg, dec_deg) via TAN projection."""
    xi = math.radians((x - cx) * cdelt1_deg)
    eta = math.radians((y - cy) * cdelt2_deg)
    rho = math.hypot(xi, eta)
    if rho < 1e-12:
        return ra0_deg % 360.0, dec0_deg

    c = math.atan(rho)
    dec0 = math.radians(dec0_deg)
    ra0 = math.radians(ra0_deg)

    sin_dec = math.cos(c) * math.sin(dec0) + (eta * math.sin(c) * math.cos(dec0)) / rho
    sin_dec = max(-1.0, min(1.0, sin_dec))
    dec = math.asin(sin_dec)

    num = xi * math.sin(c)
    den = rho * math.cos(dec0) * math.cos(c) - eta * math.sin(dec0) * math.sin(c)
    ra = (ra0 + math.atan2(num, den)) % (2.0 * math.pi)
    return math.degrees(ra), math.degrees(dec)


def angular_separation_arcsec(
    ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float
) -> float:
    """Compute exact spherical angular separation in arcseconds."""
    r1, d1 = math.radians(ra1_deg), math.radians(dec1_deg)
    r2, d2 = math.radians(ra2_deg), math.radians(dec2_deg)
    dlon = r2 - r1
    dlat = d2 - d1
    a = math.sin(dlat / 2) ** 2 + math.cos(d1) * math.cos(d2) * math.sin(dlon / 2) ** 2
    c = 2.0 * math.asin(min(1.0, math.sqrt(max(0.0, a))))
    return math.degrees(c) * 3600.0


def _resolve_field_width_deg(metadata: PhotoMetadata, width_px: int) -> float:
    """Derive horizontal field width in degrees from PhotoMetadata optical hints."""
    if metadata.field_width_deg is not None:
        return float(metadata.field_width_deg)
    if metadata.pixel_scale_arcsec is not None:
        return (float(metadata.pixel_scale_arcsec) * width_px) / 3600.0
    if metadata.focal_length_mm is not None:
        if metadata.sensor_width_mm is not None:
            sensor_w = float(metadata.sensor_width_mm)
        elif metadata.pixel_size_um is not None:
            sensor_w = (float(metadata.pixel_size_um) * width_px) / 1000.0
        else:
            sensor_w = 22.3
        return 2.0 * math.degrees(math.atan(sensor_w / (2.0 * float(metadata.focal_length_mm))))
    return 4.0


def _read_checked_bytes(path: Path) -> bytes:
    """Read file bytes after verifying non-empty size within MAX_BYTES."""
    try:
        stat_res = path.stat()
    except OSError as exc:
        raise PhotoError(f"Cannot read image file: {exc}") from exc
    if stat_res.st_size <= 0:
        raise PhotoError("Image file is empty.")
    if stat_res.st_size > MAX_BYTES:
        raise PhotoError(f"Image file is too large: maximum {MAX_BYTES} bytes.")
    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_BYTES + 1)
    except OSError as exc:
        raise PhotoError(f"Cannot read image file: {exc}") from exc
    if not payload:
        raise PhotoError("Image file is empty.")
    if len(payload) > MAX_BYTES:
        raise PhotoError(f"Image file is too large: maximum {MAX_BYTES} bytes.")
    return payload


def _read_fits_2d(payload: bytes) -> tuple[np.ndarray, fits.Header]:
    """Read a 2D primary FITS image via astropy.io.fits (applying BZERO/BSCALE natively)."""
    try:
        with fits.open(io.BytesIO(payload), memmap=False) as hdus:
            primary = hdus[0]
            header = primary.header.copy()
            if header.get("NAXIS") != 2:
                raise PhotoError(
                    "Unsupported FITS file: a 2D primary image is required "
                    "(data cubes and image extensions are not supported)."
                )
            width = int(header.get("NAXIS1", 0))
            height = int(header.get("NAXIS2", 0))
            check_dimensions(width, height)
            if primary.data is None:
                raise PhotoError("FITS primary HDU contains no pixel data.")
            pixels = np.asarray(primary.data, dtype=np.float32)
    except PhotoError:
        raise
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        raise PhotoError(f"Unable to decode FITS image: {exc}") from exc

    if pixels.ndim != 2:
        raise PhotoError("A 2D pixel array is required.")
    finite = np.isfinite(pixels)
    if not finite.any():
        raise PhotoError("Image contains no finite numeric pixels.")
    if not finite.all():
        replacement = float(np.median(pixels[finite]))
        pixels = np.where(finite, pixels, replacement).astype(np.float32)
    return pixels, header


# ============================================================================
# Synthetic Astrophotography Renderer for the 4 Curated Presets
# ============================================================================
def _render_preset_sky_image(
    preset: dict[str, Any],
    entries: list[dict[str, Any]],
    output_png: Path,
    width: int = 960,
    height: int = 640,
) -> Path:
    """Render an astrophysically accurate sky photo with catalog objects at exact TAN coordinates."""
    check_dimensions(width, height)
    if output_png.exists() and output_png.stat().st_size > 0:
        return output_png
    output_png.parent.mkdir(parents=True, exist_ok=True)

    ra0 = float(preset["ra_hint_deg"])
    dec0 = float(preset["dec_hint_deg"])
    fov_w = float(preset["field_width_deg"])
    scale_deg = fov_w / width
    if not (math.isfinite(scale_deg) and scale_deg > 1e-10):
        raise PhotoError("Degenerate preset field width or pixel scale.")
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    cdelt1 = -scale_deg
    cdelt2 = +scale_deg

    seed = int(hashlib.sha256(str(preset["id"]).encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    # Deep space night sky background gradient + subtle cosmic dust noise
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    bg_r = 7.0 + 4.0 * (1.0 - yy / height) + rng.normal(0, 1.1, (height, width))
    bg_g = 10.0 + 5.0 * (1.0 - yy / height) + rng.normal(0, 1.1, (height, width))
    bg_b = 18.0 + 8.0 * (1.0 - yy / height) + rng.normal(0, 1.3, (height, width))

    # Find catalog objects that fall inside this preset's frame
    in_frame: list[tuple[dict[str, Any], float, float]] = []
    for entry in entries:
        proj = gnomonic_sky_to_pixel(
            entry["ra_deg"], entry["dec_deg"], ra0, dec0, cx, cy, cdelt1, cdelt2
        )
        if proj is not None and inside_image(proj[0], proj[1], width, height):
            in_frame.append((entry, proj[0], proj[1]))

    # Render realistic deep-sky surface brightness profiles at exact projected (x, y)
    for entry, ox, oy in in_frame:
        did = entry["doc_id"]
        otype = str(entry["metadata"].get("object_type") or "").lower()
        size_arcmin = float(entry["metadata"].get("size_major_arcmin") or 18.0)
        radius_px = max(14.0, min(210.0, (size_arcmin / 60.0) / scale_deg * 0.42))

        dx = xx - ox
        dy = yy - oy

        if did == "M031":
            # Tilted grand-design spiral disk (Andromeda M31)
            angle = math.radians(38.0)
            rx = dx * math.cos(angle) + dy * math.sin(angle)
            ry = (-dx * math.sin(angle) + dy * math.cos(angle)) * 2.65
            r2 = rx * rx + ry * ry
            core = np.exp(-r2 / (2.0 * (radius_px * 0.22) ** 2))
            disk = np.exp(-r2 / (2.0 * (radius_px * 0.85) ** 2))
            dust = 1.0 - 0.32 * np.exp(-((np.sqrt(r2) - radius_px * 0.42) ** 2) / (2.0 * 14.0**2))
            bg_r += (core * 215.0 + disk * 95.0) * dust
            bg_g += (core * 195.0 + disk * 92.0) * dust
            bg_b += (core * 165.0 + disk * 125.0) * dust
        elif "galaxy" in otype:
            # Elliptical / spiral companion or pair (M32, M110, M81, M82)
            is_edge_on = did == "M082"
            angle = math.radians(-22.0 if not is_edge_on else 68.0)
            aspect = 3.2 if is_edge_on else 1.55
            rx = dx * math.cos(angle) + dy * math.sin(angle)
            ry = (-dx * math.sin(angle) + dy * math.cos(angle)) * aspect
            r2 = rx * rx + ry * ry
            sig = max(12.0, radius_px * 0.45)
            profile = np.exp(-r2 / (2.0 * sig**2))
            core = np.exp(-(dx * dx + dy * dy) / (2.0 * (sig * 0.35) ** 2))
            bg_r += core * 185.0 + profile * (120.0 if is_edge_on else 85.0)
            bg_g += core * 170.0 + profile * 78.0
            bg_b += core * 155.0 + profile * (75.0 if is_edge_on else 105.0)
        elif "nebula" in otype or "cluster_with_nebulosity" in otype:
            # H-II emission / reflection nebula (M42, M43, M8, M20, NGC1973/1975)
            sig = max(18.0, radius_px * 0.52)
            r2 = dx * dx + dy * dy
            cloud = np.exp(-r2 / (2.0 * sig**2))
            core = np.exp(-r2 / (2.0 * (sig * 0.28) ** 2))
            is_reflection = did in {"NGC1973", "NGC1975"}
            if is_reflection:
                bg_r += cloud * 55.0 + core * 95.0
                bg_g += cloud * 95.0 + core * 145.0
                bg_b += cloud * 165.0 + core * 210.0
            else:
                bg_r += cloud * 145.0 + core * 210.0
                bg_g += cloud * 58.0 + core * 140.0
                bg_b += cloud * 115.0 + core * 195.0
        else:
            # Open cluster glow + concentrated star core (M21, NGC6530, NGC6546, NGC1980, NGC1981)
            sig = max(14.0, radius_px * 0.38)
            r2 = dx * dx + dy * dy
            glow = np.exp(-r2 / (2.0 * sig**2))
            bg_r += glow * 65.0
            bg_g += glow * 85.0
            bg_b += glow * 135.0

    rgb = np.stack(
        [
            np.clip(bg_r, 0, 255),
            np.clip(bg_g, 0, 255),
            np.clip(bg_b, 0, 255),
        ],
        axis=-1,
    ).astype(np.uint8)

    img = Image.fromarray(rgb, mode="RGB").filter(ImageFilter.GaussianBlur(radius=1.1))
    draw = ImageDraw.Draw(img)

    # Sprinkle 420 field stars + cluster stars around each catalog object
    for _ in range(420):
        sx = float(rng.uniform(4, width - 4))
        sy = float(rng.uniform(4, height - 4))
        mag_weight = float(rng.pareto(2.4))
        r_star = min(3.2, 0.55 + 0.32 * mag_weight)
        tint = int(rng.integers(210, 255))
        warm = int(rng.integers(-20, 20))
        color = (
            min(255, max(180, tint + warm)),
            min(255, max(190, tint)),
            min(255, max(185, tint - warm)),
        )
        draw.ellipse((sx - r_star, sy - r_star, sx + r_star, sy + r_star), fill=color)
        if r_star >= 2.2:
            spike = int(r_star * 4.5)
            draw.line((sx - spike, sy, sx + spike, sy), fill=color, width=1)
            draw.line((sx, sy - spike, sx, sy + spike), fill=color, width=1)

    # Add concentrated cluster member stars at each catalog object center
    for _entry, ox, oy in in_frame:
        for _ in range(18):
            jx = float(ox + rng.normal(0, 11.0))
            jy = float(oy + rng.normal(0, 11.0))
            if 2 <= jx < width - 2 and 2 <= jy < height - 2:
                draw.ellipse((jx - 1.3, jy - 1.3, jx + 1.3, jy + 1.3), fill=(240, 246, 255))

    img.save(output_png, format="PNG")
    return output_png


def get_or_create_preset_image(
    preset_id: str, root: Path = DEFAULT_ROOT
) -> tuple[dict[str, Any], Path]:
    """Return preset config and path to its generated sky PNG in the system temp directory."""
    preset = next((p for p in VISION_PRESETS if p["id"] == preset_id), None)
    if preset is None:
        raise PhotoError(f"Unknown vision preset: {preset_id}")
    cache_dir = Path(tempfile.gettempdir()) / "astropilote_vision_presets"
    png_path = cache_dir / f"{preset_id}.png"
    if png_path.exists() and png_path.stat().st_size > 0:
        return preset, png_path
    chunks_path = discover_chunks(root)
    entries, _ = load_catalog(chunks_path)
    _render_preset_sky_image(preset, entries, png_path)
    return preset, png_path


# ============================================================================
# Unified Execution Pipeline (identify_image + Gnomonic Fallback + visualize.py)
# ============================================================================
def run_vision_pipeline(
    image_path: Path,
    metadata: PhotoMetadata | None = None,
    *,
    root: Path = DEFAULT_ROOT,
    trust_fits_wcs: bool = False,
    is_preset: bool = False,
) -> dict[str, Any]:
    """Run catalog identification and `visualize.py` annotation pipeline."""
    started = time.perf_counter()
    metadata = metadata if metadata is not None else PhotoMetadata()
    trace_steps: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    chunks_path = discover_chunks(root)
    entries, skipped = load_catalog(chunks_path)
    t_cat_ms = round((time.perf_counter() - t0) * 1000, 1)
    trace_steps.append(
        {
            "step": 1,
            "name": "load_catalog",
            "arguments": {"chunks": chunks_path.name},
            "latency_ms": t_cat_ms,
            "result": f"Loaded {len(entries)} point-localizable sky objects ({skipped} constellations excluded).",
        }
    )

    result: dict[str, Any] | None = None
    fallback_warnings: list[str] = []
    has_local_astrometry = (
        shutil.which("solve-field") is not None
        and DEFAULT_INDEXES.is_dir()
        and any(DEFAULT_INDEXES.glob("index-*.fits"))
    )

    # 1a. When trust_fits_wcs=True, delegate unconditionally to identify_image(..., trust_fits_wcs=True)
    #     and NEVER swallow PhotoError / ValueError / OSError (never fall back to metadata).
    if trust_fits_wcs:
        t_solve = time.perf_counter()
        result = _ic.identify_image(
            image_path,
            metadata=metadata,
            root=root,
            chunks=chunks_path,
            indexes=DEFAULT_INDEXES,
            trust_fits_wcs=True,
        )
        trace_steps.append(
            {
                "step": 2,
                "name": "identify_image",
                "arguments": {
                    "method": result["astrometry"]["method"],
                    "trust_fits_wcs": True,
                },
                "latency_ms": round((time.perf_counter() - t_solve) * 1000, 1),
                "result": (
                    f"WCS center RA={result['astrometry']['centre_icrs_deg']['ra']:.4f}°, "
                    f"Dec={result['astrometry']['centre_icrs_deg']['dec']:.4f}° → "
                    f"{result['object_count']} object(s) matched."
                ),
            }
        )

    # 1b. When trust_fits_wcs=False on an uploaded image and local Astrometry.net is available,
    #     run identify_image and ONLY catch SolverFailureError (never file/dimension/catalog PhotoErrors).
    elif not is_preset and has_local_astrometry:
        t_solve = time.perf_counter()
        try:
            result = _ic.identify_image(
                image_path,
                metadata=metadata,
                root=root,
                chunks=chunks_path,
                indexes=DEFAULT_INDEXES,
                trust_fits_wcs=False,
            )
        except SolverFailureError as first_exc:
            has_restrictive_hints = any(
                v is not None
                for v in (
                    metadata.ra_hint_deg,
                    metadata.dec_hint_deg,
                    metadata.field_width_deg,
                    metadata.pixel_scale_arcsec,
                    metadata.focal_length_mm,
                )
            )
            # If restrictive UI hints (e.g. leftover preset RA/Dec/FOV) caused solve-field to miss,
            # retry a blind Astrometry.net solve before falling back to explicit RA/Dec hints.
            if has_restrictive_hints:
                try:
                    result = _ic.identify_image(
                        image_path,
                        metadata=PhotoMetadata(
                            telescope=metadata.telescope,
                            camera=metadata.camera,
                            exposure_seconds=metadata.exposure_seconds,
                        ),
                        root=root,
                        chunks=chunks_path,
                        indexes=DEFAULT_INDEXES,
                        trust_fits_wcs=False,
                    )
                except SolverFailureError:
                    if metadata.ra_hint_deg is None or metadata.dec_hint_deg is None:
                        raise first_exc from None
                    fallback_warnings.append(
                        "Local Astrometry.net plate-solving did not converge; "
                        "falling back to gnomonic TAN projection from provided RA/Dec hints."
                    )
            else:
                if metadata.ra_hint_deg is None or metadata.dec_hint_deg is None:
                    raise
                fallback_warnings.append(
                    "Local Astrometry.net plate-solving did not converge; "
                    "falling back to gnomonic TAN projection from provided RA/Dec hints."
                )

        if result is not None:
            trace_steps.append(
                {
                    "step": 2,
                    "name": "identify_image",
                    "arguments": {
                        "method": result["astrometry"]["method"],
                        "trust_fits_wcs": False,
                    },
                    "latency_ms": round((time.perf_counter() - t_solve) * 1000, 1),
                    "result": (
                        f"WCS center RA={result['astrometry']['centre_icrs_deg']['ra']:.4f}°, "
                        f"Dec={result['astrometry']['centre_icrs_deg']['dec']:.4f}° → "
                        f"{result['object_count']} object(s) matched."
                    ),
                }
            )

    # 2. Otherwise use gnomonic TAN WCS projection from pointing & optical metadata
    if result is None:
        t_solve = time.perf_counter()
        payload = _read_checked_bytes(image_path)

        if payload.startswith(b"SIMPLE"):
            pixels, _ = _read_fits_2d(payload)
            image_format = "FITS"
            height, width = pixels.shape
            check_dimensions(width, height)
        else:
            try:
                with Image.open(io.BytesIO(payload)) as im:
                    if im.format not in {"JPEG", "PNG"}:
                        raise PhotoError(
                            "Unsupported image format: expected JPEG, PNG, or 2D primary FITS."
                        )
                    check_dimensions(im.width, im.height)
                    if getattr(im, "n_frames", 1) != 1:
                        raise PhotoError("Animated or multi-frame images are not supported.")
                    image_format = im.format
                    norm_im = ImageOps.exif_transpose(im)
                    width, height = norm_im.size
                    check_dimensions(width, height)
            except PhotoError:
                raise
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                raise PhotoError(f"Unable to decode image: {exc}") from exc

        if width <= 0 or height <= 0:
            raise PhotoError("Invalid image dimensions.")

        if metadata.ra_hint_deg is None or metadata.dec_hint_deg is None:
            raise PhotoError(
                "Local Astrometry.net (solve-field) is not installed on this machine. "
                "Please provide RA & Dec pointing coordinates (ra_hint_deg, dec_hint_deg) "
                "or choose one of the curated sky photo presets."
            )

        messages: list[str] = list(fallback_warnings)
        ra0 = float(metadata.ra_hint_deg)
        dec0 = float(metadata.dec_hint_deg)
        fov_w = _resolve_field_width_deg(metadata, width)
        scale_deg = fov_w / width
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        cdelt1 = -scale_deg
        cdelt2 = +scale_deg

        if not (
            math.isfinite(cdelt1)
            and abs(cdelt1) > 1e-10
            and math.isfinite(cdelt2)
            and abs(cdelt2) > 1e-10
        ):
            raise PhotoError("Degenerate WCS pixel scale derived from optical metadata.")

        method = "gnomonic_tan_metadata"
        if not has_local_astrometry:
            messages.append(
                "WCS gnomonic TAN constructed from pointing & optical metadata "
                f'(FOV = {fov_w:.2f}° × {fov_w * height / width:.2f}°, scale = {scale_deg * 3600:.2f}"/px).'
            )

        wcs_header = {
            "WCSAXES": 2,
            "CRPIX1": round(cx + 1.0, 4),
            "CRPIX2": round(cy + 1.0, 4),
            "CDELT1": round(cdelt1, 8),
            "CDELT2": round(cdelt2, 8),
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRVAL1": round(ra0, 6),
            "CRVAL2": round(dec0, 6),
            "RADESYS": "FK5",
            "EQUINOX": 2000.0,
        }

        objects: list[dict[str, Any]] = []
        projection_errors: list[str] = []

        for entry in entries:
            proj = gnomonic_sky_to_pixel(
                entry["ra_deg"], entry["dec_deg"], ra0, dec0, cx, cy, cdelt1, cdelt2
            )
            if proj is None:
                continue
            x, y = proj
            if not inside_image(x, y, width, height):
                continue
            rec_ra, rec_dec = gnomonic_pixel_to_sky(x, y, ra0, dec0, cx, cy, cdelt1, cdelt2)
            residual = angular_separation_arcsec(entry["ra_deg"], entry["dec_deg"], rec_ra, rec_dec)
            if not math.isfinite(residual) or residual > 1.0:
                projection_errors.append(entry["doc_id"])
                continue
            objects.append(
                {
                    "doc_id": entry["doc_id"],
                    "title": entry["title"],
                    "relation": "catalogue_center_in_image",
                    "detected_in_pixels": None,
                    "pixel": {
                        "x": round(x, 2),
                        "y": round(y, 2),
                        "distance_to_edge_px": round(
                            min(x + 0.5, y + 0.5, width - 0.5 - x, height - 0.5 - y), 2
                        ),
                    },
                    "metadata": entry["metadata"],
                    "evidence": entry["evidence"],
                }
            )

        result = {
            "schema_version": "1.0",
            "status": "partial" if projection_errors else "ok",
            "meaning": "catalogue_centres_in_image_not_pixel_detections",
            "image": {
                "name": image_path.name,
                "format": image_format,
                "width_px": width,
                "height_px": height,
                "pixel_coordinates": (
                    "zero_based_normalized_array: x=column, y=row; "
                    "JPEG/PNG EXIF orientation applied; FITS row order unchanged"
                ),
            },
            "input_metadata": metadata.model_dump(mode="json"),
            "astrometry": {
                "method": method,
                "centre_icrs_deg": {"ra": round(ra0, 6), "dec": round(dec0, 6)},
                "wcs_header": wcs_header,
            },
            "catalogue": {
                "chunks_file": chunks_path.name,
                "coordinate_convention": "FK5 J2000 as declared by the corpus",
                "eligible_documents": len(entries),
                "excluded_documents": skipped,
            },
            "object_count": len(objects),
            "objects": objects,
            "projection_errors": projection_errors,
            "warnings": messages,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

        trace_steps.append(
            {
                "step": 2,
                "name": "match_catalog",
                "arguments": {
                    "ra_deg": round(ra0, 4),
                    "dec_deg": round(dec0, 4),
                    "scale_arcsec_px": round(abs(cdelt1) * 3600.0, 2),
                },
                "latency_ms": round((time.perf_counter() - t_solve) * 1000, 1),
                "result": (
                    f"Projected {len(entries)} J2000 objects onto {width}×{height} px TAN grid → "
                    f"{len(objects)} inside footprint ({', '.join(o['doc_id'] for o in objects) or 'none'})."
                ),
            }
        )

    # 3. Run `visualize(image_path, result_path, output_dir)` to generate
    #    the annotated PNG (cyan circles + yellow labels) and CSV table!
    t_vis = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="astropilote-vis-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        vis_input_path = image_path
        vis_result = dict(result)

        # If input is FITS, render a stretched PNG copy so `visualize.py` (which accepts JPEG/PNG) can annotate it
        if result["image"]["format"] == "FITS":
            raw_bytes = _read_checked_bytes(image_path)
            arr, _ = _read_fits_2d(raw_bytes)
            vmin, vmax = float(np.percentile(arr, 5)), float(np.percentile(arr, 99.5))
            norm = np.clip((arr - vmin) / max(1e-5, vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
            vis_input_path = tmp_path / "fits_preview.png"
            Image.fromarray(norm, mode="L").convert("RGB").save(vis_input_path, format="PNG")
            vis_result["image"] = {**result["image"], "format": "PNG"}

        result_json_path = tmp_path / "vision_result.json"
        result_json_path.write_text(json.dumps(vis_result, ensure_ascii=False), encoding="utf-8")
        out_dir = tmp_path / "annotated_out"

        annotated_png_path, csv_path = visualize(
            image_path=vis_input_path,
            result_path=result_json_path,
            output_dir=out_dir,
        )

        orig_bytes = _read_checked_bytes(vis_input_path)
        orig_mime = "image/jpeg" if vis_result["image"]["format"] == "JPEG" else "image/png"
        orig_b64 = base64.b64encode(orig_bytes).decode("ascii")
        ann_b64 = base64.b64encode(_read_checked_bytes(annotated_png_path)).decode("ascii")
        csv_text = csv_path.read_text(encoding="utf-8-sig")

    trace_steps.append(
        {
            "step": 3,
            "name": "visualize_overlay",
            "arguments": {
                "annotated_objects": result["object_count"],
                "outputs": ["annotated.png", "objects.csv"],
            },
            "latency_ms": round((time.perf_counter() - t_vis) * 1000, 1),
            "result": (
                f"Rendered cyan target markers & yellow doc_id labels ({result['object_count']} objects) "
                "and exported CSV table."
            ),
        }
    )

    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return {
        "result": result,
        "original_image_data_url": f"data:{orig_mime};base64,{orig_b64}",
        "annotated_image_data_url": f"data:image/png;base64,{ann_b64}",
        "csv_content": csv_text,
        "trace_steps": trace_steps,
    }
