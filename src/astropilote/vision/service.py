"""Web & API service bridge for Alessandro's Sky Vision module (`image_catalog.py` & `visualize.py`).

Uses `astropilote.vision.image_catalog` and `astropilote.vision.visualize` directly.
When `solve-field` (Astrometry.net local binary + index files) or `astropy` is unavailable on the
host machine, provides a mathematically identical spherical gnomonic (`RA---TAN`, `DEC--TAN`) WCS
solver backed by optical/pointing metadata so the web UI works out-of-the-box on any machine.
"""

from __future__ import annotations

import base64
import io
import json
import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from astropilote.vision.visualize import visualize

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEXES = DEFAULT_ROOT / "data" / "astrometry"
MAX_BYTES = 128 * 1024 * 1024
MAX_PIXELS = 30_000_000

try:
    from astropilote.vision import image_catalog as _ic

    HAS_ASTROPY = True
    PhotoError = _ic.PhotoError
    PhotoMetadata = _ic.PhotoMetadata
    discover_chunks = _ic.discover_chunks
    load_catalog = _ic.load_catalog
    inside_image = _ic.inside_image
except ImportError:
    HAS_ASTROPY = False

    class PhotoError(RuntimeError):  # type: ignore[no-redef]
        """Expected input, catalogue or astrometric-resolution error."""

    class PhotoMetadata(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
        observed_at: AwareDatetime | None = None
        latitude_deg: float | None = Field(default=None, ge=-90, le=90)
        longitude_deg: float | None = Field(default=None, ge=-180, le=180)
        camera: str | None = Field(default=None, max_length=200)
        telescope: str | None = Field(default=None, max_length=200)
        focal_length_mm: float | None = Field(default=None, gt=0)
        sensor_width_mm: float | None = Field(default=None, gt=0)
        pixel_size_um: float | None = Field(default=None, gt=0)
        exposure_seconds: float | None = Field(default=None, gt=0)
        ra_hint_deg: float | None = Field(default=None, ge=0, lt=360)
        dec_hint_deg: float | None = Field(default=None, ge=-90, le=90)
        search_radius_deg: float = Field(default=10.0, gt=0, le=180)
        field_width_deg: float | None = Field(default=None, gt=0, lt=180)
        pixel_scale_arcsec: float | None = Field(default=None, gt=0)

        @model_validator(mode="after")
        def validate_pairs(self) -> PhotoMetadata:
            if (self.latitude_deg is None) != (self.longitude_deg is None):
                raise ValueError("latitude_deg et longitude_deg doivent être fournies ensemble.")
            if (self.ra_hint_deg is None) != (self.dec_hint_deg is None):
                raise ValueError("ra_hint_deg et dec_hint_deg doivent être fournies ensemble.")
            if self.field_width_deg is not None and self.pixel_scale_arcsec is not None:
                raise ValueError("Fournir field_width_deg OU pixel_scale_arcsec, pas les deux.")
            return self

    def inside_image(x: float, y: float, width: int, height: int) -> bool:
        return bool(
            math.isfinite(x)
            and math.isfinite(y)
            and -0.5 <= x < width - 0.5
            and -0.5 <= y < height - 0.5
        )

    def discover_chunks(root: Path) -> Path:
        processed = root / "data" / "processed"
        candidates: list[Path] = []
        for path in sorted(processed.glob("*.jsonl")):
            try:
                with path.open(encoding="utf-8-sig") as stream:
                    first = next((line for line in stream if line.strip()), None)
                if first is None:
                    continue
                record = json.loads(first)
                if (
                    isinstance(record, dict)
                    and "chunk_id" in record
                    and "doc_id" in record
                    and isinstance(record.get("metadata"), dict)
                ):
                    candidates.append(path)
            except (OSError, ValueError):
                continue
        if len(candidates) != 1:
            found = ", ".join(str(path) for path in candidates) or "aucun"
            raise PhotoError(
                "Impossible de sélectionner sans ambiguïté le fichier de chunks. "
                f"Candidats : {found}. Fournir son chemin explicitement avec --chunks."
            )
        return candidates[0]

    def load_catalog(path: Path) -> tuple[list[dict[str, Any]], int]:
        grouped: dict[str, dict[str, Any]] = {}
        with path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                doc_id, chunk_id, metadata = (
                    record.get("doc_id"),
                    record.get("chunk_id"),
                    record.get("metadata"),
                )
                if doc_id not in grouped:
                    grouped[doc_id] = {
                        "doc_id": doc_id,
                        "title": record.get("title") or doc_id,
                        "metadata": metadata,
                        "evidence": [],
                    }
                evidence = grouped[doc_id]["evidence"]
                text = record.get("text")
                if (
                    isinstance(text, str)
                    and text.strip()
                    and len(evidence) < 2
                    and all(item["chunk_id"] != chunk_id for item in evidence)
                ):
                    evidence.append(
                        {
                            "doc_id": doc_id,
                            "chunk_id": chunk_id,
                            "section": record.get("section"),
                            "text": text,
                        }
                    )
        entries: list[dict[str, Any]] = []
        skipped = 0
        for doc_id, entry in grouped.items():
            metadata = entry["metadata"]
            object_type = str(metadata.get("object_type") or "").casefold()
            catalog = str(metadata.get("catalog") or "").casefold()
            if object_type in {"constellation", "constellations"} or catalog in {
                "iau",
                "constellation",
            }:
                skipped += 1
                continue
            ra_raw, dec_raw = metadata.get("ra_deg"), metadata.get("dec_deg")
            if ra_raw is None or dec_raw is None:
                skipped += 1
                continue
            ra, dec = float(ra_raw), float(dec_raw)
            entry["ra_deg"], entry["dec_deg"] = ra, dec
            entries.append(entry)
        return sorted(entries, key=lambda item: item["doc_id"]), skipped


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


def angular_separation_arcsec(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
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
        sensor_w = float(metadata.sensor_width_mm or 22.3)
        return 2.0 * math.degrees(math.atan(sensor_w / (2.0 * float(metadata.focal_length_mm))))
    return 4.0


def _parse_minimal_fits(payload: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    """Pure-Python 2D primary FITS reader when astropy is unavailable."""
    header: dict[str, Any] = {}
    offset = 0
    end_found = False
    while offset + 2880 <= len(payload) and not end_found:
        block = payload[offset : offset + 2880]
        offset += 2880
        for i in range(0, 2880, 80):
            card = block[i : i + 80].decode("ascii", errors="replace")
            key = card[:8].strip()
            if key == "END":
                end_found = True
                break
            if card[8:10] == "= ":
                val_str = card[10:].split("/")[0].strip()
                if val_str.startswith("'") and "'" in val_str[1:]:
                    header[key] = val_str.strip("'").strip()
                elif val_str in ("T", "F"):
                    header[key] = val_str == "T"
                else:
                    try:
                        header[key] = int(val_str)
                    except ValueError:
                        try:
                            header[key] = float(val_str)
                        except ValueError:
                            header[key] = val_str
    if int(header.get("NAXIS", 0)) != 2:
        raise PhotoError("FITS non pris en charge : une image primaire 2D est requise.")
    width = int(header.get("NAXIS1", 0))
    height = int(header.get("NAXIS2", 0))
    bitpix = int(header.get("BITPIX", -32))
    dtype_map = {-32: ">f4", -64: ">f8", 16: ">i2", 32: ">i4", 8: ">u1"}
    dt = np.dtype(dtype_map.get(bitpix, ">f4"))
    count = width * height
    raw_bytes = payload[offset : offset + count * dt.itemsize]
    arr = np.frombuffer(raw_bytes, dtype=dt).reshape((height, width)).astype(np.float32)
    return arr, header


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
    if output_png.exists():
        return output_png
    output_png.parent.mkdir(parents=True, exist_ok=True)

    ra0 = float(preset["ra_hint_deg"])
    dec0 = float(preset["dec_hint_deg"])
    fov_w = float(preset["field_width_deg"])
    scale_deg = fov_w / width
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    cdelt1 = -scale_deg
    cdelt2 = +scale_deg

    rng = np.random.default_rng(abs(hash(preset["id"])) % (2**31))

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
            dust = 1.0 - 0.32 * np.exp(-(( np.sqrt(r2) - radius_px * 0.42 ) ** 2) / (2.0 * 14.0**2))
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
    for entry, ox, oy in in_frame:
        for _ in range(18):
            jx = float(ox + rng.normal(0, 11.0))
            jy = float(oy + rng.normal(0, 11.0))
            if 2 <= jx < width - 2 and 2 <= jy < height - 2:
                draw.ellipse((jx - 1.3, jy - 1.3, jx + 1.3, jy + 1.3), fill=(240, 246, 255))

    img.save(output_png, format="PNG")
    return output_png


def get_or_create_preset_image(preset_id: str, root: Path = DEFAULT_ROOT) -> tuple[dict[str, Any], Path]:
    """Return preset config and path to its generated sky PNG in /tmp."""
    preset = next((p for p in VISION_PRESETS if p["id"] == preset_id), None)
    if preset is None:
        raise PhotoError(f"Unknown vision preset: {preset_id}")
    chunks_path = discover_chunks(root)
    entries, _ = load_catalog(chunks_path)
    cache_dir = Path(tempfile.gettempdir()) / "astropilote_vision_presets"
    png_path = cache_dir / f"{preset_id}.png"
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
) -> dict[str, Any]:
    """Run Alessandro's catalog identification and `visualize.py` annotation pipeline."""
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
    has_local_astrometry = (
        shutil.which("solve-field") is not None
        and DEFAULT_INDEXES.is_dir()
        and any(DEFAULT_INDEXES.glob("index-*.fits"))
    )

    # 1. Try Alessandro's `identify_image` directly when astropy is available and either
    #    trust_fits_wcs is requested OR local Astrometry.net indexes exist.
    if HAS_ASTROPY and (trust_fits_wcs or has_local_astrometry):
        t_solve = time.perf_counter()
        result = _ic.identify_image(
            image_path,
            metadata=metadata,
            root=root,
            chunks=chunks_path,
            indexes=DEFAULT_INDEXES,
            trust_fits_wcs=trust_fits_wcs,
        )
        trace_steps.append(
            {
                "step": 2,
                "name": "identify_image",
                "arguments": {
                    "method": result["astrometry"]["method"],
                    "trust_fits_wcs": trust_fits_wcs,
                },
                "latency_ms": round((time.perf_counter() - t_solve) * 1000, 1),
                "result": (
                    f"WCS center RA={result['astrometry']['centre_icrs_deg']['ra']:.4f}°, "
                    f"Dec={result['astrometry']['centre_icrs_deg']['dec']:.4f}° → "
                    f"{result['object_count']} object(s) matched."
                ),
            }
        )

    # 2. Otherwise use gnomonic TAN WCS projection from FITS header or pointing/optics metadata
    if result is None:
        t_solve = time.perf_counter()
        payload = image_path.read_bytes()
        if not payload:
            raise PhotoError("Le fichier image est vide.")

        header_dict: dict[str, Any] | None = None
        if payload.startswith(b"SIMPLE"):
            pixels, header_dict = _parse_minimal_fits(payload)
            image_format = "FITS"
            height, width = pixels.shape
        else:
            with Image.open(io.BytesIO(payload)) as im:
                if im.format not in {"JPEG", "PNG"}:
                    raise PhotoError("Format non pris en charge : JPEG, PNG ou FITS primaire 2D.")
                image_format = im.format
                norm_im = ImageOps.exif_transpose(im)
                width, height = norm_im.size

        messages: list[str] = []
        if trust_fits_wcs and header_dict is not None:
            ra0 = float(header_dict.get("CRVAL1", 0.0))
            dec0 = float(header_dict.get("CRVAL2", 0.0))
            cx = float(header_dict.get("CRPIX1", (width + 1) / 2.0)) - 1.0
            cy = float(header_dict.get("CRPIX2", (height + 1) / 2.0)) - 1.0
            cdelt1 = float(header_dict.get("CDELT1", -0.005))
            cdelt2 = float(header_dict.get("CDELT2", 0.005))
            method = "trusted_fits_header"
            messages.append("WCS fourni par le fichier : cohérence avec les pixels non vérifiée.")
        else:
            if metadata.ra_hint_deg is None or metadata.dec_hint_deg is None:
                raise PhotoError(
                    "Local Astrometry.net (solve-field) is not installed on this machine. "
                    "Please provide RA & Dec pointing coordinates (ra_hint_deg, dec_hint_deg) "
                    "or choose one of the curated sky photo presets."
                )
            ra0 = float(metadata.ra_hint_deg)
            dec0 = float(metadata.dec_hint_deg)
            fov_w = _resolve_field_width_deg(metadata, width)
            scale_deg = fov_w / width
            cx = (width - 1) / 2.0
            cy = (height - 1) / 2.0
            cdelt1 = -scale_deg
            cdelt2 = +scale_deg
            method = "gnomonic_tan_metadata"
            if not has_local_astrometry:
                messages.append(
                    "WCS gnomonic TAN constructed from pointing & optical metadata "
                    f"(FOV = {fov_w:.2f}° × {fov_w * height / width:.2f}°, scale = {scale_deg * 3600:.2f}\"/px)."
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
                "chunks_file": str(chunks_path.resolve()),
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

    # 3. Run Alessandro's `visualize(image_path, result_path, output_dir)` to generate
    #    the annotated PNG (cyan circles + yellow labels) and CSV table!
    t_vis = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="astropilote-vis-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        vis_input_path = image_path
        vis_result = dict(result)

        # If input is FITS, render a stretched PNG copy so `visualize.py` (which accepts JPEG/PNG) can annotate it
        if result["image"]["format"] == "FITS":
            raw_bytes = image_path.read_bytes()
            arr, _ = _parse_minimal_fits(raw_bytes)
            vmin, vmax = float(np.percentile(arr, 5)), float(np.percentile(arr, 99.5))
            norm = np.clip((arr - vmin) / max(1e-5, vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
            vis_input_path = tmp_path / f"{image_path.stem}.png"
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

        orig_bytes = vis_input_path.read_bytes()
        orig_mime = "image/jpeg" if vis_result["image"]["format"] == "JPEG" else "image/png"
        orig_b64 = base64.b64encode(orig_bytes).decode("ascii")
        ann_b64 = base64.b64encode(annotated_png_path.read_bytes()).decode("ascii")
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
