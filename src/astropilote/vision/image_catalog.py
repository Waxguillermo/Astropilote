# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "astropy==7.2.3",
#   "numpy==2.5.3",
#   "pillow==12.3.0",
#   "pydantic==2.14.0",
# ]
# ///

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
import time
import unittest
from pathlib import Path
from typing import Any
import warnings

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


class PhotoError(RuntimeError):
    """Expected input, catalogue or astrometric-resolution error."""


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
            raise ValueError(
                "latitude_deg et longitude_deg doivent être fournies ensemble."
            )
        if (self.ra_hint_deg is None) != (self.dec_hint_deg is None):
            raise ValueError(
                "ra_hint_deg et dec_hint_deg doivent être fournies ensemble."
            )
        if self.field_width_deg is not None and self.pixel_scale_arcsec is not None:
            raise ValueError(
                "Fournir field_width_deg OU pixel_scale_arcsec, pas les deux."
            )
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
            f"Candidats : {found}. "
            "Fournir son chemin explicitement avec --chunks."
        )
    return candidates[0]


def load_catalog(path: Path) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise PhotoError(
                    f"JSON invalide dans {path}, ligne {line_number}."
                ) from exc
            if not isinstance(record, dict):
                raise PhotoError(f"Chunk non objet, ligne {line_number}.")
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
                raise PhotoError(f"Schéma de chunk non reconnu, ligne {line_number}.")
            if metadata.get("doc_id", doc_id) != doc_id:
                raise PhotoError(f"doc_id incohérent pour le chunk {chunk_id}.")
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
                        raise PhotoError(f"Métadonnée {key} incohérente pour {doc_id}.")
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
        if isinstance(ra_raw, bool) or isinstance(dec_raw, bool):
            raise PhotoError(f"Coordonnées booléennes invalides pour {doc_id}.")
        try:
            ra, dec = float(ra_raw), float(dec_raw)
        except (TypeError, ValueError) as exc:
            raise PhotoError(f"Coordonnées invalides pour {doc_id}.") from exc
        if (
            not math.isfinite(ra)
            or not math.isfinite(dec)
            or not 0 <= ra < 360
            or not -90 <= dec <= 90
        ):
            raise PhotoError(f"Coordonnées hors domaine pour {doc_id}.")
        entry["ra_deg"], entry["dec_deg"] = ra, dec
        entries.append(entry)
    if not entries:
        raise PhotoError(
            "Aucun objet ponctuellement localisable trouvé dans le catalogue."
        )
    return sorted(entries, key=lambda item: item["doc_id"]), skipped


def check_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise PhotoError("Dimensions d'image invalides.")
    if width * height > MAX_PIXELS:
        raise PhotoError(f"Image trop grande : maximum {MAX_PIXELS} pixels.")


def load_pixels(path: Path) -> tuple[np.ndarray, fits.Header | None, str]:
    with path.open("rb") as stream:
        payload = stream.read(MAX_BYTES + 1)
    if not payload:
        raise PhotoError("Le fichier image est vide.")
    if len(payload) > MAX_BYTES:
        raise PhotoError(f"Fichier trop volumineux : maximum {MAX_BYTES} octets.")
    header: fits.Header | None = None
    try:
        if payload.startswith(b"SIMPLE"):
            with fits.open(io.BytesIO(payload), memmap=False) as hdus:
                primary = hdus[0]
                header = primary.header.copy()
                if header.get("NAXIS") != 2:
                    raise PhotoError(
                        "FITS non pris en charge : une image primaire 2D est requise, sans cube ni image en extension."
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
                        "Format non pris en charge : JPEG, PNG ou FITS primaire 2D."
                    )
                check_dimensions(image.width, image.height)
                if getattr(image, "n_frames", 1) != 1:
                    raise PhotoError(
                        "Les images animées ou multivues ne sont pas prises en charge."
                    )
                image_format = image.format
                normalized = ImageOps.exif_transpose(image)
                if normalized.mode in {"I", "I;16", "I;16B", "I;16L", "F"}:
                    pixels = np.asarray(normalized, dtype=np.float32)
                else:
                    pixels = np.asarray(normalized.convert("L"), dtype=np.float32)
    except PhotoError:
        raise
    except Exception as exc:
        raise PhotoError(f"Impossible de décoder l'image : {exc}") from exc
    if pixels.ndim != 2:
        raise PhotoError("Une matrice de pixels 2D est requise.")
    finite = np.isfinite(pixels)
    if not finite.any():
        raise PhotoError("L'image ne contient aucun pixel numérique fini.")
    if not finite.all():
        replacement = float(np.median(pixels[finite]))
        pixels = np.where(finite, pixels, replacement).astype(np.float32)
    return pixels, header, image_format


def celestial_wcs(header: fits.Header) -> WCS:
    try:
        candidate = WCS(header, relax=True)
        if not candidate.has_celestial:
            raise PhotoError("Le fichier ne contient pas de WCS céleste.")
        result = candidate.celestial
        matrix = np.asarray(result.pixel_scale_matrix, dtype=float)
        if (
            result.pixel_n_dim != 2
            or matrix.shape != (2, 2)
            or not np.isfinite(matrix).all()
            or np.linalg.det(matrix) == 0
        ):
            raise PhotoError("WCS céleste dégénéré ou non pris en charge.")
        return result
    except PhotoError:
        raise
    except Exception as exc:
        raise PhotoError(
            f"WCS céleste illisible ou non pris en charge : {exc}"
        ) from exc


def solve_image(
    pixels: np.ndarray, metadata: PhotoMetadata, indexes: Path, timeout: int
) -> WCS:
    if os.name != "posix":
        raise PhotoError("Cette implémentation du solveur local cible macOS et Linux.")
    executable = shutil.which("solve-field")
    if executable is None:
        raise PhotoError("solve-field introuvable. Installer Astrometry.net.")
    indexes = indexes.resolve()
    if any(character.isspace() for character in str(indexes)):
        raise PhotoError(
            "Le chemin du dossier d'index ne doit contenir aucun espace ni caractère d'espacement dans cette version."
        )
    if not indexes.is_dir() or not any(indexes.glob("index-*.fits")):
        raise PhotoError(f"Aucun index Astrometry.net trouvé dans {indexes}.")
    with tempfile.TemporaryDirectory(prefix="astropilote-photo-") as temporary:
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
                    raise PhotoError(
                        f"Résolution interrompue après {timeout} secondes. "
                        "Vérifier les index, l'image et les indications fournies."
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
            raise PhotoError(
                "Aucune solution astrométrique obtenue. "
                f"Code de sortie : {returncode}.\n{tail}"
            )

        with fits.open(wcs_path, memmap=False) as hdus:
            solved_header = hdus[0].header.copy()

        solved_header["NAXIS"] = 2
        solved_header["NAXIS1"] = int(pixels.shape[1])
        solved_header["NAXIS2"] = int(pixels.shape[0])

        return celestial_wcs(solved_header)


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

                if not (
                    math.isfinite(float(x_basic)) and math.isfinite(float(y_basic))
                ):
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
        raise PhotoError("timeout doit être compris entre 1 et 3600 secondes.")

    started = time.perf_counter()
    metadata = metadata if metadata is not None else PhotoMetadata()
    chunks_path = chunks if chunks is not None else discover_chunks(root)
    entries, skipped = load_catalog(chunks_path)
    pixels, header, image_format = load_pixels(image)
    height, width = pixels.shape
    messages: list[str] = []

    if trust_fits_wcs:
        if header is None:
            raise PhotoError(
                "--trust-fits-wcs nécessite une image FITS avec un WCS céleste."
            )
        wcs = celestial_wcs(header)
        method = "trusted_fits_header"
        messages.append(
            "WCS fourni par le fichier : cohérence avec les pixels non vérifiée."
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

    if not (
        math.isfinite(float(center.ra.deg)) and math.isfinite(float(center.dec.deg))
    ):
        raise PhotoError("Le WCS ne définit pas de centre céleste fini.")

    objects, projection_errors = match_catalog(entries, wcs, width, height)

    if projection_errors:
        messages.append(
            "Certaines projections ont échoué : consulter projection_errors. "
            "La liste des résultats peut être incomplète."
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Identifier les centres d'objets du catalogue dans une image du ciel."
        )
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
            parser.error("Aucun test trouvé dans tests/test_vision.py.")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

    if args.image is None:
        parser.error("Fournir une image ou utiliser --self-test.")

    try:
        if args.output is not None and args.output.exists():
            raise PhotoError(
                f"Le fichier de sortie existe déjà : {args.output}. "
                "Choisir un nouveau chemin."
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
        payload = (
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )

        if args.output is None:
            sys.stdout.write(payload)
        else:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload)

        return 0

    except (PhotoError, OSError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Traitement interrompu.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
