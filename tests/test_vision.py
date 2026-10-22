"""Deterministic fixtures, not real-sky detection or accuracy benchmarks."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, NoConvergence
from PIL import Image
from pydantic import ValidationError

from astropilote.vision.image_catalog import (
    PhotoError,
    PhotoMetadata,
    celestial_wcs,
    check_dimensions,
    discover_chunks,
    identify_image,
    inside_image,
    load_catalog,
    load_pixels,
    match_catalog,
    solve_image,
)


def make_wcs() -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [50.5, 40.5]
    wcs.wcs.crval = [359.9, 30.0]
    wcs.wcs.cdelt = [-0.01, 0.01]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.radesys = "FK5"
    wcs.wcs.equinox = 2000.0
    return wcs


def record(
    doc_id: str = "TEST_IN",
    ra: float | None = 359.9,
    dec: float | None = 30.0,
    *,
    catalog: str = "Messier",
    object_type: str = "galaxy",
) -> dict:
    return {
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}#fixture-0",
        "title": doc_id,
        "section": "Fixture",
        "text": "Texte de test, sans information astronomique réelle.",
        "metadata": {
            "doc_id": doc_id,
            "catalog": catalog,
            "object_type": object_type,
            "ra_deg": ra,
            "dec_deg": dec,
        },
    }


def write_chunks(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )


class PhotoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.chunks = self.root / "chunks.jsonl"
        write_chunks(self.chunks, [record()])

    def test_pixel_boundaries(self) -> None:
        self.assertTrue(inside_image(-0.5, -0.5, 100, 80))
        self.assertTrue(inside_image(99.499, 79.499, 100, 80))
        self.assertFalse(inside_image(99.5, 40, 100, 80))
        self.assertFalse(inside_image(40, 79.5, 100, 80))
        self.assertFalse(inside_image(float("nan"), 40, 100, 80))
        self.assertFalse(inside_image(float("inf"), 40, 100, 80))

    def test_invalid_metadata(self) -> None:
        invalid = [
            {"latitude_deg": 48.0},
            {"ra_hint_deg": 10.0},
            {"observed_at": "2026-01-01T21:00:00"},
            {"field_width_deg": 3, "pixel_scale_arcsec": 2},
            {"focal_length_mm": -1},
            {"pixel_scale_arcsec": float("nan")},
            {"unrecognized": True},
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                PhotoMetadata(**values)

    def test_valid_metadata(self) -> None:
        metadata = PhotoMetadata(
            observed_at="2026-01-01T21:00:00Z",
            latitude_deg=48,
            longitude_deg=2,
        )
        self.assertIsNotNone(metadata.observed_at.utcoffset())

    def test_dimension_limits(self) -> None:
        with self.assertRaises(PhotoError):
            check_dimensions(0, 80)
        with self.assertRaises(PhotoError):
            check_dimensions(10000, 10000)

    def test_catalogue_grouping_and_exclusions(self) -> None:
        duplicate = record()
        duplicate["chunk_id"] = "TEST_IN#fixture-1"
        write_chunks(
            self.chunks,
            [
                record(),
                duplicate,
                record("CON_TEST", catalog="Constellation"),
                record("NO_COORD", ra=None),
            ],
        )
        entries, excluded = load_catalog(self.chunks)
        self.assertEqual(len(entries), 1)
        self.assertEqual(len(entries[0]["evidence"]), 2)
        self.assertEqual(excluded, 2)

    def test_inconsistent_catalogue_rejected(self) -> None:
        write_chunks(self.chunks, [record(), record(ra=10)])
        with self.assertRaises(PhotoError):
            load_catalog(self.chunks)

    def test_invalid_coordinates_rejected(self) -> None:
        write_chunks(self.chunks, [record(ra=True)])
        with self.assertRaises(PhotoError):
            load_catalog(self.chunks)

    def test_malformed_json_rejected(self) -> None:
        self.chunks.write_text("not JSON\n", encoding="utf-8")
        with self.assertRaises(PhotoError):
            load_catalog(self.chunks)

    def test_discovery_and_ambiguity(self) -> None:
        processed = self.root / "data" / "processed"
        processed.mkdir(parents=True)
        write_chunks(processed / "chunks.jsonl", [record()])
        self.assertEqual(discover_chunks(self.root), processed / "chunks.jsonl")
        write_chunks(processed / "other.jsonl", [record()])
        with self.assertRaises(PhotoError):
            discover_chunks(self.root)

    def test_png_and_exif_orientation(self) -> None:
        path = self.root / "image.png"
        image = Image.new("RGB", (100, 80))
        exif = Image.Exif()
        exif[274] = 6
        image.save(path, exif=exif)
        pixels, header, image_format = load_pixels(path)
        self.assertEqual(pixels.shape, (100, 80))
        self.assertIsNone(header)
        self.assertEqual(image_format, "PNG")

    def test_fits_reading_and_nonfinite_replacement(self) -> None:
        path = self.root / "image.fits"
        pixels = np.ones((80, 100), dtype=np.float32)
        pixels[0, 0] = np.nan
        fits.writeto(path, pixels, make_wcs().to_header())
        loaded, header, image_format = load_pixels(path)
        self.assertTrue(np.isfinite(loaded).all())
        self.assertEqual(image_format, "FITS")
        self.assertTrue(celestial_wcs(header).has_celestial)

    def test_cube_rejected(self) -> None:
        path = self.root / "cube.fits"
        fits.writeto(path, np.ones((3, 8, 10)))
        with self.assertRaises(PhotoError):
            load_pixels(path)

    def test_missing_wcs_rejected(self) -> None:
        with self.assertRaises(PhotoError):
            celestial_wcs(fits.Header())

    def test_projection_and_ra_wrap(self) -> None:
        write_chunks(
            self.chunks,
            [record(), record("TEST_WRAP", ra=0.1), record("TEST_OUT", ra=180)],
        )
        entries, _ = load_catalog(self.chunks)
        objects, errors = match_catalog(entries, make_wcs(), 100, 80)
        self.assertEqual(errors, [])
        self.assertEqual(
            {item["doc_id"] for item in objects},
            {"TEST_IN", "TEST_WRAP"},
        )
        center = next(item for item in objects if item["doc_id"] == "TEST_IN")
        self.assertAlmostEqual(center["pixel"]["x"], 49.5, places=5)
        self.assertAlmostEqual(center["pixel"]["y"], 39.5, places=5)
        self.assertIsNone(center["detected_in_pixels"])

    def test_icrs_wcs_frame_conversion(self) -> None:
        coordinate = SkyCoord(359.9, 30, unit="deg", frame="fk5", equinox="J2000")
        icrs = coordinate.icrs
        wcs = make_wcs()
        wcs.wcs.radesys = "ICRS"
        wcs.wcs.equinox = float("nan")
        wcs.wcs.crval = [icrs.ra.deg, icrs.dec.deg]
        entries, _ = load_catalog(self.chunks)
        objects, errors = match_catalog(entries, wcs, 100, 80)
        self.assertEqual(errors, [])
        self.assertAlmostEqual(objects[0]["pixel"]["x"], 49.5, places=5)

    def test_trusted_fits_pipeline_without_solver(self) -> None:
        path = self.root / "image.fits"
        fits.writeto(path, np.ones((80, 100)), make_wcs().to_header())
        with patch("astropilote.vision.image_catalog.solve_image") as solver:
            result = identify_image(path, chunks=self.chunks, trust_fits_wcs=True)
        solver.assert_not_called()
        self.assertEqual(result["object_count"], 1)
        self.assertEqual(result["astrometry"]["method"], "trusted_fits_header")
        self.assertTrue(result["warnings"])
        json.dumps(result, allow_nan=False)

    def test_png_cannot_supply_trusted_wcs(self) -> None:
        path = self.root / "image.png"
        Image.new("RGB", (100, 80)).save(path)
        with self.assertRaises(PhotoError):
            identify_image(path, chunks=self.chunks, trust_fits_wcs=True)

    def test_solver_wrapper_and_success_marker(self) -> None:
        indexes = self.root / "indexes"
        indexes.mkdir()
        (indexes / "index-test.fits").touch()
        metadata = PhotoMetadata(
            ra_hint_deg=10,
            dec_hint_deg=20,
            pixel_scale_arcsec=3,
        )

        def fake_launch(command: list[str], **kwargs):
            self.assertIn("--no-verify", command)
            self.assertIn("arcsecperpix", command)
            self.assertEqual(command[command.index("--ra") + 1], "10.0")
            self.assertIs(kwargs["start_new_session"], True)
            fits.writeto(
                Path(command[command.index("--wcs") + 1]),
                np.zeros((1, 1)),
                make_wcs().to_header(),
            )
            Path(command[command.index("--solved") + 1]).write_bytes(b"\x01")

            class FakeProcess:
                def wait(self, timeout=None):
                    return 0

            return FakeProcess()

        with (
            patch(
                "astropilote.vision.image_catalog.shutil.which",
                return_value="/fake/solve-field",
            ),
            patch(
                "astropilote.vision.image_catalog.subprocess.Popen",
                side_effect=fake_launch,
            ),
        ):
            result = solve_image(np.zeros((80, 100)), metadata, indexes, 5)
        self.assertTrue(result.has_celestial)

    def test_png_16bit_preserved(self) -> None:
        path = self.root / "16bit.png"
        Image.fromarray(np.array([[0, 1000, 65535]], dtype=np.uint16)).save(path)
        pixels, _, _ = load_pixels(path)
        self.assertEqual(pixels.tolist(), [[0.0, 1000.0, 65535.0]])

    def test_empty_image_rejected(self) -> None:
        path = self.root / "empty.png"
        path.touch()
        with self.assertRaises(PhotoError):
            load_pixels(path)

    def test_projection_failure_reported(self) -> None:
        entries, _ = load_catalog(self.chunks)
        with patch(
            "astropilote.vision.image_catalog.skycoord_to_pixel",
            side_effect=NoConvergence("test"),
        ):
            objects, errors = match_catalog(entries, make_wcs(), 100, 80)
        self.assertEqual(objects, [])
        self.assertEqual(errors, ["TEST_IN"])

    def test_solver_no_solution_rejected(self) -> None:
        indexes = self.root / "indexes"
        indexes.mkdir()
        (indexes / "index-test.fits").touch()
        with (
            patch(
                "astropilote.vision.image_catalog.shutil.which",
                return_value="/fake/solve-field",
            ),
            patch("astropilote.vision.image_catalog.subprocess.Popen") as launch,
        ):
            launch.return_value.wait.return_value = 0
            with self.assertRaises(PhotoError):
                solve_image(np.zeros((80, 100)), PhotoMetadata(), indexes, 5)

    def test_solver_timeout(self) -> None:
        indexes = self.root / "indexes"
        indexes.mkdir()
        (indexes / "index-test.fits").touch()
        with (
            patch(
                "astropilote.vision.image_catalog.shutil.which",
                return_value="/fake/solve-field",
            ),
            patch("astropilote.vision.image_catalog.subprocess.Popen") as launch,
            patch("astropilote.vision.image_catalog.os.killpg") as kill,
        ):
            launch.return_value.pid = 12345
            launch.return_value.wait.side_effect = [
                subprocess.TimeoutExpired("solve-field", 1),
                -9,
            ]
            with self.assertRaises(PhotoError):
                solve_image(np.zeros((80, 100)), PhotoMetadata(), indexes, 1)
            kill.assert_called_once()

    def test_undefined_tan_projection_skips_sip(self) -> None:
        entries, _ = load_catalog(self.chunks)

        with patch(
            "astropilote.vision.image_catalog.skycoord_to_pixel",
            return_value=(np.nan, np.nan),
        ) as project:
            objects, errors = match_catalog(entries, make_wcs(), 100, 80)

        self.assertEqual(objects, [])
        self.assertEqual(errors, [])
        self.assertEqual(project.call_count, 1)
        self.assertEqual(project.call_args.kwargs["mode"], "wcs")

    def test_sip_nan_warning_is_reported(self) -> None:
        entries, _ = load_catalog(self.chunks)

        def project(*args, **kwargs):
            if kwargs["mode"] == "wcs":
                return 49.5, 39.5

            warnings.warn(
                "All-NaN slice encountered",
                RuntimeWarning,
            )
            return np.nan, np.nan

        with patch(
            "astropilote.vision.image_catalog.skycoord_to_pixel",
            side_effect=project,
        ):
            objects, errors = match_catalog(entries, make_wcs(), 100, 80)

        self.assertEqual(objects, [])
        self.assertEqual(errors, ["TEST_IN"])


if __name__ == "__main__":
    unittest.main()
