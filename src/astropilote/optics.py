"""Single Source of Truth for optical sensors, exact trigonometric FOV, NPF rule, and equipment presets."""

from __future__ import annotations

import math
from typing import Any

# Physical sensor dimensions (width_mm, height_mm)
SENSOR_SPECS: dict[str, tuple[float, float]] = {
    "full-frame": (36.0, 24.0),
    "APS-C": (23.5, 15.6),
    "micro-4/3": (17.3, 13.0),
    "1-inch": (13.2, 8.8),
    "1/2.8-inch": (5.57, 3.13),  # Sony IMX462 / IMX662 (Seestar S50, Dwarf)
    "smartphone": (
        7.6,
        5.7,
    ),  # Typical 1/1.65" main mobile sensor (~5.1 mm physical f -> ~24 mm eq.)
}

# Typical pixel pitch in micrometers when not explicitly provided
DEFAULT_PIXEL_UM: dict[str, float] = {
    "full-frame": 5.9,
    "APS-C": 3.76,
    "micro-4/3": 3.3,
    "1-inch": 3.76,
    "1/2.8-inch": 2.9,
    "smartphone": 1.4,
}

EQUIPMENT_PRESETS: list[dict[str, Any]] = [
    {
        "preset_id": "apo_80ed_eq",
        "rig_name": "80mm f/6 APO Refractor + APS-C + Equatorial GoTo",
        "telescope_or_lens": "80/480mm f/6 Apochromatic Refractor",
        "focal_length_mm": 480.0,
        "aperture_mm": 80.0,
        "camera_model": "Astro-modified APS-C Mirrorless / DSLR",
        "sensor_format": "APS-C",
        "pixel_size_um": 3.76,
        "mount_type": "equatorial_goto",
        "has_autoguiding": True,
        "filters": ["broadband", "dual_band"],
        "bortle_class": 4,
    },
    {
        "preset_id": "tele_135_tracker",
        "rig_name": "135mm f/2 Telephoto + Full-Frame + Star Tracker",
        "telescope_or_lens": "Samyang 135mm f/2 ED Telephoto Lens",
        "focal_length_mm": 135.0,
        "aperture_mm": 67.5,
        "camera_model": "Full-Frame 36×24 mm Camera",
        "sensor_format": "full-frame",
        "pixel_size_um": 5.9,
        "mount_type": "star_tracker",
        "has_autoguiding": False,
        "filters": ["broadband", "dual_band"],
        "bortle_class": 4,
    },
    {
        "preset_id": "newton_150_750",
        "rig_name": "150/750 f/5 Newtonian + Cooled APS-C Camera",
        "telescope_or_lens": "150/750mm f/5 Newtonian Reflector with Coma Corrector",
        "focal_length_mm": 750.0,
        "aperture_mm": 150.0,
        "camera_model": "ZWO ASI2600MC Pro (Cooled APS-C)",
        "sensor_format": "APS-C",
        "pixel_size_um": 3.76,
        "mount_type": "equatorial_goto",
        "has_autoguiding": True,
        "filters": ["broadband", "dual_band", "narrowband_sho"],
        "bortle_class": 5,
    },
    {
        "preset_id": "sct_c8_1280",
        "rig_name": 'Celestron C8 (1280mm f/6.3) + 1" Astro Camera',
        "telescope_or_lens": "Celestron C8 203mm + f/6.3 Focal Reducer (1280mm)",
        "focal_length_mm": 1280.0,
        "aperture_mm": 203.0,
        "camera_model": 'ZWO ASI533MC Pro (1" Sensor)',
        "sensor_format": "1-inch",
        "pixel_size_um": 3.76,
        "mount_type": "equatorial_goto",
        "has_autoguiding": True,
        "filters": ["broadband", "uhc_cls"],
        "bortle_class": 5,
    },
    {
        "preset_id": "dslr_50mm_tripod",
        "rig_name": "50mm f/1.8 Lens + APS-C Camera on Fixed Tripod",
        "telescope_or_lens": "50mm f/1.8 Prime Lens",
        "focal_length_mm": 50.0,
        "aperture_mm": 28.0,
        "camera_model": "APS-C Mirrorless / DSLR",
        "sensor_format": "APS-C",
        "pixel_size_um": 3.76,
        "mount_type": "tripod",
        "has_autoguiding": False,
        "filters": ["broadband"],
        "bortle_class": 4,
    },
    {
        "preset_id": "seestar_s50",
        "rig_name": "Smart Telescope 50/250mm (Seestar S50 / Alt-Az GoTo)",
        "telescope_or_lens": "Integrated 50/250mm f/5 APO Triplet (Seestar S50)",
        "focal_length_mm": 250.0,
        "aperture_mm": 50.0,
        "camera_model": 'Built-in Sony IMX462 1/2.8" CMOS Sensor',
        "sensor_format": "1/2.8-inch",
        "pixel_size_um": 2.9,
        "mount_type": "altaz_goto",
        "has_autoguiding": False,
        "filters": ["broadband", "dual_band"],
        "bortle_class": 6,
    },
    {
        "preset_id": "smartphone_night",
        "rig_name": "Smartphone on Tripod (Night / Astro Mode)",
        "telescope_or_lens": "Smartphone Main Wide Lens (5.1mm f/1.8 · 24mm eq.)",
        "focal_length_mm": 5.1,
        "aperture_mm": 2.8,
        "camera_model": "Smartphone Main Wide Sensor (7.6×5.7 mm)",
        "sensor_format": "smartphone",
        "pixel_size_um": 1.4,
        "mount_type": "tripod",
        "has_autoguiding": False,
        "filters": ["broadband"],
        "bortle_class": 5,
    },
]


def resolve_sensor_dimensions_mm(sensor_format: str | None) -> tuple[float, float]:
    """Return `(width_mm, height_mm)` for `sensor_format` (defaults to APS-C)."""
    return SENSOR_SPECS.get(sensor_format or "APS-C", SENSOR_SPECS["APS-C"])


def compute_fov_deg(
    focal_length_mm: float, sensor_width_mm: float, sensor_height_mm: float
) -> tuple[float, float]:
    """Compute exact rectilinear field of view `(fov_w_deg, fov_h_deg)` via `2 * atan(size / (2 * f))`."""
    f = max(focal_length_mm, 0.5)
    fov_w = 2.0 * math.degrees(math.atan(sensor_width_mm / (2.0 * f)))
    fov_h = 2.0 * math.degrees(math.atan(sensor_height_mm / (2.0 * f)))
    return fov_w, fov_h


def compute_fov_arcmin(
    focal_length_mm: float,
    sensor_format: str = "APS-C",
    *,
    sensor_width_mm: float | None = None,
    sensor_height_mm: float | None = None,
) -> tuple[float, float]:
    """Compute exact field of view `(fov_w_arcmin, fov_h_arcmin)` for a focal length and sensor."""
    sw, sh = resolve_sensor_dimensions_mm(sensor_format)
    w = sensor_width_mm if (sensor_width_mm is not None and sensor_width_mm > 0) else sw
    h = sensor_height_mm if (sensor_height_mm is not None and sensor_height_mm > 0) else sh
    deg_w, deg_h = compute_fov_deg(focal_length_mm, w, h)
    return deg_w * 60.0, deg_h * 60.0


def compute_pixel_scale_arcsec(focal_length_mm: float, pixel_size_um: float) -> float:
    """Compute image sampling in arcseconds per pixel (`206.265 * p_um / f_mm`)."""
    return 206.264806 * max(pixel_size_um, 0.5) / max(focal_length_mm, 0.5)


def compute_npf_seconds(
    focal_length_mm: float,
    aperture_mm: float | None = None,
    pixel_size_um: float | None = None,
    dec_deg: float | None = None,
    sensor_format: str = "APS-C",
) -> float:
    """Compute untracked tripod max sub-exposure (seconds) using Sahavre's NPF rule:

    `t = (35 * N + 30 * p) / (f * cos(dec))` (clamped at high declination so circumpolar stars stay reasonable).
    """
    f = max(focal_length_mm, 1.0)
    f_ratio = (f / aperture_mm) if (aperture_mm and aperture_mm > 0) else 4.0
    f_ratio = max(1.2, min(16.0, f_ratio))
    pitch = (
        pixel_size_um
        if (pixel_size_um and pixel_size_um > 0)
        else DEFAULT_PIXEL_UM.get(sensor_format, 3.76)
    )
    cos_dec = max(0.25, abs(math.cos(math.radians(dec_deg)))) if dec_deg is not None else 1.0
    raw = (35.0 * f_ratio + 30.0 * pitch) / (f * cos_dec)
    return round(max(0.5, min(30.0, raw)), 1)


def angular_separation_deg(
    ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float
) -> float:
    """Compute exact great-circle angular separation in degrees using the Haversine formula."""
    r1 = math.radians(ra1_deg)
    d1 = math.radians(dec1_deg)
    r2 = math.radians(ra2_deg)
    d2 = math.radians(dec2_deg)
    dra = r2 - r1
    ddec = d2 - d1
    a = math.sin(ddec / 2.0) ** 2 + math.cos(d1) * math.cos(d2) * math.sin(dra / 2.0) ** 2
    a = min(1.0, max(0.0, a))
    return math.degrees(2.0 * math.asin(math.sqrt(a)))
