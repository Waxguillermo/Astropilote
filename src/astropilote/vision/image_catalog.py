"""Identify catalogued object centres inside an astronomical image."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, NoConvergence
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from PIL import Image, ImageOps
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEXES = DEFAULT_ROOT / "data" / "astrometry"
MAX_BYTES = 128 * 1024 * 1024
MAX_PIXELS = 30_000_000

Image.MAX_IMAGE_PIXELS = MAX_PIXELS

_SOLVER_SEMAPHORE = threading.Semaphore(2)
_CATALOG_CACHE: dict[tuple[Path, int, int], tuple[list[dict[str, Any]], int]] = {}
_CATALOG_CACHE_LOCK = threading.Lock()


class PhotoError(RuntimeError):
    """Expected input, catalogue or astrometric-resolution error."""


class SolverFailureError(PhotoError):
    """Astrometry.net local solver unavailability, timeout, or non-convergence error."""


class PhotoMetadata(BaseModel):
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
            raise ValueError("latitude_deg and longitude_deg must be provided together.")
        if (self.ra_hint_deg is None) != (self.dec_hint_deg is None):
            raise ValueError("ra_hint_deg and dec_hint_deg must be provided together.")
        if self.field_width_deg is not None and self.pixel_scale_arcsec is not None:
            raise ValueError("Provide either field_width_deg or pixel_scale_arcsec, not both.")
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
        found = ", ".join(str(path) for path in candidates) or "none"
        raise PhotoError(
            "Unable to unambiguously select the chunks file. "
            f"Candidates: {found}. "
            "Provide the path explicitly with --chunks."
        )
    return candidates[0]


def load_catalog(path: Path) -> tuple[list[dict[str, Any]], int]:
    resolved_path = path.resolve()
    try:
        stat_res = resolved_path.stat()
    except OSError as exc:
        raise PhotoError(f"Cannot read catalog file: {exc}") from exc

    cache_key = (resolved_path, stat_res.st_mtime_ns, stat_res.st_size)
    with _CATALOG_CACHE_LOCK:
        cached = _CATALOG_CACHE.get(cache_key)
    if cached is not None:
        cached_entries, cached_skipped = cached
        return [{**e, "evidence": list(e["evidence"])} for e in cached_entries], cached_skipped

    grouped: dict[str, dict[str, Any]] = {}
    try:
        with resolved_path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError as exc:
                    raise PhotoError(
                        f"Invalid JSON in {resolved_path.name}, line {line_number}."
                    ) from exc
                if not isinstance(record, dict):
                    raise PhotoError(f"Chunk record is not a JSON object at line {line_number}.")
                doc_id, chunk_id, metadata = (
                    record.get("doc_id"),
                    record.get("chunk_id"),
                    record.get("metadata"),
                )
                if (
                    not isinstance(doc_id, str)
                    or not isinstance(chunk_id, str)
                    or not isinstance(metadata, dict)
                ):
                    raise PhotoError(f"Unrecognized chunk schema at line {line_number}.")
                if metadata.get("doc_id", doc_id) != doc_id:
                    raise PhotoError(f"Inconsistent doc_id for chunk {chunk_id}.")
                if doc_id not in grouped:
                    grouped[doc_id] = {
                        "doc_id": doc_id,
                        "title": record.get("title") or doc_id,
                        "metadata": metadata,
                        "evidence": [],
                    }
                else:
                    previous = grouped[doc_id]["metadata"]
                    for key in ("ra_deg", "dec_deg", "object_type", "catalog"):
                        if previous.get(key) != metadata.get(key):
                            raise PhotoError(f"Inconsistent metadata field '{key}' for {doc_id}.")
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
    except OSError as exc:
        raise PhotoError(f"Cannot read catalog file: {exc}") from exc

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
        if isinstance(ra_raw, bool) or isinstance(dec_raw, bool):
            raise PhotoError(f"Invalid boolean coordinates for {doc_id}.")
        try:
            ra, dec = float(ra_raw), float(dec_raw)
        except (TypeError, ValueError) as exc:
            raise PhotoError(f"Invalid coordinates for {doc_id}.") from exc
        if (
            not math.isfinite(ra)
            or not math.isfinite(dec)
            or not 0 <= ra < 360
            or not -90 <= dec <= 90
        ):
            raise PhotoError(f"Out-of-range coordinates for {doc_id}.")
        entry["ra_deg"], entry["dec_deg"] = ra, dec
        entries.append(entry)
    if not entries:
        raise PhotoError("No point-localizable objects found in the catalog.")

    sorted_entries = sorted(entries, key=lambda item: item["doc_id"])
    with _CATALOG_CACHE_LOCK:
        for stale_key in [k for k in _CATALOG_CACHE if k[0] == resolved_path and k != cache_key]:
            _CATALOG_CACHE.pop(stale_key, None)
        _CATALOG_CACHE[cache_key] = (sorted_entries, skipped)

    return [{**e, "evidence": list(e["evidence"])} for e in sorted_entries], skipped


def check_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise PhotoError("Invalid image dimensions.")
    if width * height > MAX_PIXELS:
        raise PhotoError(f"Image too large: maximum {MAX_PIXELS} pixels.")


def load_pixels(path: Path) -> tuple[np.ndarray, fits.Header | None, str]:
    with path.open("rb") as stream:
        payload = stream.read(MAX_BYTES + 1)
    if not payload:
        raise PhotoError("Image file is empty.")
    if len(payload) > MAX_BYTES:
        raise PhotoError(f"Image file is too large: maximum {MAX_BYTES} bytes.")
    header: fits.Header | None = None
    try:
        if payload.startswith(b"SIMPLE"):
            with fits.open(io.BytesIO(payload), memmap=False) as hdus:
                primary = hdus[0]
                header = primary.header.copy()
                if header.get("NAXIS") != 2:
                    raise PhotoError(
                        "Unsupported FITS file: a 2D primary image is required "
                        "(data cubes and image extensions are not supported)."
                    )
                width, height = (
                    int(header.get("NAXIS1", 0)),
                    int(header.get("NAXIS2", 0)),
                )
                check_dimensions(width, height)
                pixels = np.asarray(primary.data, dtype=np.float32)
            image_format = "FITS"
        else:
            with Image.open(io.BytesIO(payload)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise PhotoError(
                        "Unsupported image format: expected JPEG, PNG, or 2D primary FITS."
                    )
                check_dimensions(image.width, image.height)
                if getattr(image, "n_frames", 1) != 1:
                    raise PhotoError("Animated or multi-frame images are not supported.")
                image_format = image.format
                normalized = ImageOps.exif_transpose(image)
                check_dimensions(normalized.width, normalized.height)
                if normalized.mode in {"I", "I;16", "I;16B", "I;16L", "F"}:
                    pixels = np.asarray(normalized, dtype=np.float32)
                else:
                    pixels = np.asarray(normalized.convert("L"), dtype=np.float32)
    except PhotoError:
        raise
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        raise PhotoError(f"Unable to decode image: {exc}") from exc
    if pixels.ndim != 2:
        raise PhotoError("A 2D pixel array is required.")
    finite = np.isfinite(pixels)
    if not finite.any():
        raise PhotoError("Image contains no finite numeric pixels.")
    if not finite.all():
        replacement = float(np.median(pixels[finite]))
        pixels = np.where(finite, pixels, replacement).astype(np.float32)
    return pixels, header, image_format


def celestial_wcs(header: fits.Header) -> WCS:
    try:
        candidate = WCS(header, relax=True)
        if not candidate.has_celestial:
            raise PhotoError("FITS header does not contain a celestial WCS.")
        result = candidate.celestial
        matrix = np.asarray(result.pixel_scale_matrix, dtype=float)
        if (
            result.pixel_n_dim != 2
            or matrix.shape != (2, 2)
            or not np.isfinite(matrix).all()
            or np.linalg.det(matrix) == 0
        ):
            raise PhotoError("Degenerate or unsupported celestial WCS.")
        return result
    except PhotoError:
        raise
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        raise PhotoError(f"Unreadable or unsupported celestial WCS: {exc}") from exc


def solve_image(pixels: np.ndarray, metadata: PhotoMetadata, indexes: Path, timeout: int) -> WCS:
    if os.name != "posix":
        raise SolverFailureError(
            "Local Astrometry.net solver is only supported on macOS and Linux."
        )
    executable = shutil.which("solve-field")
    if executable is None:
        raise SolverFailureError("solve-field executable not found. Please install Astrometry.net.")
    indexes = indexes.resolve()
    if any(character.isspace() for character in str(indexes)):
        raise SolverFailureError(
            "Astrometry.net index directory path must not contain whitespace characters."
        )
    if not indexes.is_dir() or not any(indexes.glob("index-*.fits")):
        raise SolverFailureError(f"No Astrometry.net index files found in {indexes}.")

    with _SOLVER_SEMAPHORE, tempfile.TemporaryDirectory(prefix="astropilote-photo-") as temporary:
        work = Path(temporary)
        image_path, config_path = work / "image.fits", work / "astrometry.cfg"
        wcs_path, solved_path, log_path = (
            work / "solution.wcs",
            work / "solution.solved",
            work / "solver.log",
        )
        fits.writeto(image_path, pixels, overwrite=True)
        config_path.write_text(
            f"add_path {indexes}\nautoindex\ncpulimit {timeout}\n", encoding="utf-8"
        )
        command = [
            executable,
            "--config",
            str(config_path),
            "--dir",
            str(work),
            "--out",
            "solution",
            "--overwrite",
            "--no-plots",
            "--no-verify",
            "--fits-image",
            "--cpulimit",
            str(timeout),
            "--new-fits",
            "none",
            "--match",
            "none",
            "--rdls",
            "none",
            "--corr",
            "none",
            "--index-xyls",
            "none",
            "--wcs",
            str(wcs_path),
            "--solved",
            str(solved_path),
        ]
        if metadata.ra_hint_deg is not None:
            command.extend(
                [
                    "--ra",
                    str(metadata.ra_hint_deg),
                    "--dec",
                    str(metadata.dec_hint_deg),
                    "--radius",
                    str(metadata.search_radius_deg),
                ]
            )
        if metadata.field_width_deg is not None:
            command.extend(
                [
                    "--scale-units",
                    "degwidth",
                    "--scale-low",
                    str(metadata.field_width_deg * 0.8),
                    "--scale-high",
                    str(metadata.field_width_deg * 1.2),
                ]
            )
        elif metadata.pixel_scale_arcsec is not None:
            command.extend(
                [
                    "--scale-units",
                    "arcsecperpix",
                    "--scale-low",
                    str(metadata.pixel_scale_arcsec * 0.8),
                    "--scale-high",
                    str(metadata.pixel_scale_arcsec * 1.2),
                ]
            )
        command.append(str(image_path))
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command,
                cwd=work,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=timeout)
            except BaseException as exc:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                if isinstance(exc, subprocess.TimeoutExpired):
                    raise SolverFailureError(
                        f"Astrometric plate-solving timed out after {timeout} seconds. "
                        "Check index files, image quality, and pointing/scale hints."
                    ) from exc
                raise

        solved = (
            returncode == 0
            and solved_path.is_file()
            and solved_path.read_bytes()[:1] == b"\x01"
            and wcs_path.is_file()
        )

        if not solved:
            with log_path.open("rb") as log:
                log.seek(0, os.SEEK_END)
                log.seek(max(0, log.tell() - 6000))
                tail = log.read().decode("utf-8", errors="replace")
            raise SolverFailureError(
                f"No astrometric plate-solve solution found (exit code {returncode}).\n{tail}"
            )

        with fits.open(wcs_path, memmap=False) as hdus:
            solved_header = hdus[0].header.copy()

        solved_header["NAXIS"] = 2
        solved_header["NAXIS1"] = int(pixels.shape[1])
        solved_header["NAXIS2"] = int(pixels.shape[0])

        try:
            return celestial_wcs(solved_header)
        except PhotoError as exc:
            raise SolverFailureError(str(exc)) from exc


def match_catalog(
    entries: list[dict[str, Any]],
    wcs: WCS,
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Project J2000 catalogue positions; do not infer pixel-level detection."""

    objects: list[dict[str, Any]] = []
    projection_errors: list[str] = []

    for entry in entries:
        coordinate = SkyCoord(
            ra=entry["ra_deg"],
            dec=entry["dec_deg"],
            unit="deg",
            frame="fk5",
            equinox="J2000",
        )

        try:
            is_tan = all("TAN" in str(ctype).upper() for ctype in wcs.wcs.ctype)

            if is_tan:
                x_basic, y_basic = skycoord_to_pixel(
                    coordinate,
                    wcs,
                    origin=0,
                    mode="wcs",
                )

                if not (math.isfinite(float(x_basic)) and math.isfinite(float(y_basic))):
                    continue

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "error",
                    message=r"All-NaN slice encountered",
                    category=RuntimeWarning,
                )
                x_raw, y_raw = skycoord_to_pixel(
                    coordinate,
                    wcs,
                    origin=0,
                    mode="all",
                )

            x, y = float(x_raw), float(y_raw)

            if not (math.isfinite(x) and math.isfinite(y)):
                projection_errors.append(entry["doc_id"])
                continue

            if not inside_image(x, y, width, height):
                continue

            recovered = pixel_to_skycoord(x, y, wcs, origin=0, mode="all")
            residual = float(coordinate.separation(recovered).arcsec)

            if not math.isfinite(residual) or residual > 1.0:
                projection_errors.append(entry["doc_id"])
                continue

        except (NoConvergence, ValueError, RuntimeError, RuntimeWarning):
            projection_errors.append(entry["doc_id"])
            continue

        objects.append(
            {
                "doc_id": entry["doc_id"],
                "title": entry["title"],
                "relation": "catalogue_center_in_image",
                "detected_in_pixels": None,
                "pixel": {
                    "x": x,
                    "y": y,
                    "distance_to_edge_px": min(
                        x + 0.5,
                        y + 0.5,
                        width - 0.5 - x,
                        height - 0.5 - y,
                    ),
                },
                "metadata": entry["metadata"],
                "evidence": entry["evidence"],
            }
        )

    return objects, projection_errors


def identify_image(
    image: Path,
    *,
    metadata: PhotoMetadata | None = None,
    root: Path = DEFAULT_ROOT,
    chunks: Path | None = None,
    indexes: Path = DEFAULT_INDEXES,
    timeout: int = 180,
    trust_fits_wcs: bool = False,
) -> dict[str, Any]:
    """Public Python entry point; local inputs only, no LLM invocation."""

    if not 1 <= timeout <= 3600:
        raise PhotoError("timeout must be between 1 and 3600 seconds.")

    started = time.perf_counter()
    metadata = metadata if metadata is not None else PhotoMetadata()
    chunks_path = chunks if chunks is not None else discover_chunks(root)
    entries, skipped = load_catalog(chunks_path)
    pixels, header, image_format = load_pixels(image)
    height, width = pixels.shape
    messages: list[str] = []

    if trust_fits_wcs:
        if header is None:
            raise PhotoError("--trust-fits-wcs requires a FITS image with a valid celestial WCS.")
        wcs = celestial_wcs(header)
        method = "trusted_fits_header"
        messages.append(
            "WCS read from FITS header: consistency with image pixels was not verified."
        )
    else:
        wcs = solve_image(pixels, metadata, indexes, timeout)
        method = "local_astrometry_net"

    center = pixel_to_skycoord(
        (width - 1) / 2,
        (height - 1) / 2,
        wcs,
        origin=0,
        mode="all",
    ).icrs

    if not (math.isfinite(float(center.ra.deg)) and math.isfinite(float(center.dec.deg))):
        raise PhotoError("WCS does not define a finite celestial image center.")

    objects, projection_errors = match_catalog(entries, wcs, width, height)

    if projection_errors:
        messages.append(
            "Some catalog projections failed (see projection_errors); "
            "the matched object list may be incomplete."
        )

    return {
        "schema_version": "1.0",
        "status": "partial" if projection_errors else "ok",
        "meaning": "catalogue_centres_in_image_not_pixel_detections",
        "image": {
            "name": image.name,
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
            "centre_icrs_deg": {
                "ra": float(center.ra.deg),
                "dec": float(center.dec.deg),
            },
            "wcs_header": dict(wcs.to_header(relax=True)),
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Identify catalogued celestial object centers inside a sky image."
    )
    parser.add_argument("image", type=Path, nargs="?")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--chunks", type=Path)
    parser.add_argument("--indexes", type=Path, default=DEFAULT_INDEXES)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--trust-fits-wcs", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        suite = unittest.defaultTestLoader.discover(
            str(DEFAULT_ROOT / "tests"),
            pattern="test_vision.py",
        )
        if suite.countTestCases() == 0:
            parser.error("No tests found in tests/test_vision.py.")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

    if args.image is None:
        parser.error("Provide an image path or pass --self-test.")

    try:
        if args.output is not None and args.output.exists():
            raise PhotoError(
                f"Output file already exists: {args.output}. Choose a new output path."
            )

        metadata = (
            PhotoMetadata.model_validate_json(args.metadata.read_text(encoding="utf-8"))
            if args.metadata is not None
            else PhotoMetadata()
        )

        result = identify_image(
            args.image,
            metadata=metadata,
            root=args.root,
            chunks=args.chunks,
            indexes=args.indexes,
            timeout=args.timeout,
            trust_fits_wcs=args.trust_fits_wcs,
        )
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"

        if args.output is None:
            sys.stdout.write(payload)
        else:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload)

        return 0

    except (PhotoError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
