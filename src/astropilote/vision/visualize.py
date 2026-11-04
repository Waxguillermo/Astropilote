"""Annotate an astronomical photograph from Sky Vision JSON output and export a CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from astropilote.vision.image_catalog import (
    MAX_BYTES,
    MAX_PIXELS,
    PhotoError,
    check_dimensions,
    inside_image,
)

Image.MAX_IMAGE_PIXELS = MAX_PIXELS

_CSV_DANGEROUS_PREFIXES = {"=", "+", "-", "@", "\t", "\r"}
_CSV_DANGEROUS_LSTRIP = {"=", "+", "-", "@"}


def _sanitize_csv_cell(value: Any) -> Any:
    """Neutralize spreadsheet formula injection on string cells while preserving numeric types."""
    if isinstance(value, str) and (
        value[:1] in _CSV_DANGEROUS_PREFIXES or value.lstrip()[:1] in _CSV_DANGEROUS_LSTRIP
    ):
        return f"'{value}"
    return value


def visualize(
    image_path: Path,
    result_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Render target markers and labels onto `image_path` and write an accompanying CSV file."""
    result = json.loads(result_path.read_text(encoding="utf-8"))

    if result.get("schema_version") != "1.0":
        raise ValueError("Unsupported vision JSON schema version.")

    if result["image"]["format"] not in {"JPEG", "PNG"}:
        raise ValueError("Visualization only supports JPEG and PNG images.")

    stat_res = image_path.stat()
    if stat_res.st_size <= 0:
        raise ValueError("Input image file is empty.")
    if stat_res.st_size > MAX_BYTES:
        raise ValueError(f"Input image file is too large: maximum {MAX_BYTES} bytes.")

    with Image.open(image_path) as source:
        if source.format not in {"JPEG", "PNG"}:
            raise ValueError("Input image must be a JPEG or PNG file.")
        check_dimensions(source.width, source.height)
        image = ImageOps.exif_transpose(source).convert("RGB")
        check_dimensions(image.width, image.height)

    expected = (
        result["image"]["width_px"],
        result["image"]["height_px"],
    )
    if image.size != expected:
        raise ValueError(
            f"Incompatible image dimensions: image={image.size}, JSON={expected}. "
            "Use the original unresized photograph."
        )

    objects = result["objects"]

    if result["object_count"] != len(objects):
        raise ValueError("object_count does not match the number of objects in results.")

    for obj in objects:
        x = float(obj["pixel"]["x"])
        y = float(obj["pixel"]["y"])

        if not inside_image(x, y, image.width, image.height):
            raise ValueError(f"Invalid pixel position for {obj['doc_id']}.")

    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_path = output_dir / f"{result_path.stem}_annotated.png"
    csv_path = output_dir / f"{result_path.stem}_objects.csv"

    for path in (annotated_path, csv_path):
        if path.exists():
            raise FileExistsError(
                f"Output file already exists: {path}. "
                "Choose another output directory or move the existing file."
            )

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)

    for obj in objects:
        x = float(obj["pixel"]["x"])
        y = float(obj["pixel"]["y"])
        label = str(obj["doc_id"])

        draw.ellipse(
            (x - 7, y - 7, x + 7, y + 7),
            outline="cyan",
            width=2,
        )

        left, top, right, bottom = draw.textbbox(
            (0, 0),
            label,
            font=font,
            stroke_width=2,
        )
        text_width = right - left
        text_height = bottom - top

        tx = max(3, min(x + 10, image.width - text_width - 3))
        ty = max(3, min(y - 20, image.height - text_height - 3))

        draw.text(
            (tx - left, ty - top),
            label,
            font=font,
            fill="yellow",
            stroke_width=2,
            stroke_fill="black",
        )

    image.save(annotated_path, format="PNG")

    with csv_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "doc_id",
                "title",
                "object_type",
                "magnitude",
                "x_px",
                "y_px",
                "distance_to_edge_px",
            ]
        )

        for obj in objects:
            metadata = obj["metadata"]
            pixel = obj["pixel"]

            writer.writerow(
                [
                    _sanitize_csv_cell(obj["doc_id"]),
                    _sanitize_csv_cell(obj["title"]),
                    _sanitize_csv_cell(metadata.get("object_type")),
                    _sanitize_csv_cell(metadata.get("magnitude")),
                    _sanitize_csv_cell(pixel["x"]),
                    _sanitize_csv_cell(pixel["y"]),
                    _sanitize_csv_cell(pixel["distance_to_edge_px"]),
                ]
            )

    print(f"Result status: {result['status']}")
    print(f"Annotated objects: {len(objects)}")
    print(f"Projection errors: {len(result['projection_errors'])}")

    for warning in result.get("warnings", []):
        print(f"Warning: {warning}")

    print(f"Annotated image: {annotated_path}")
    print(f"Objects CSV table: {csv_path}")

    return annotated_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate catalogued object centers on a JPEG/PNG sky photograph "
            "and export their CSV summary table."
        )
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        visualize(
            image_path=args.image.expanduser(),
            result_path=args.result.expanduser(),
            output_dir=args.output_dir.expanduser(),
        )
    except (OSError, ValueError, KeyError, TypeError, PhotoError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
