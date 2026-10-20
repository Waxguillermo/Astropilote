"""Annoter une photographie à partir du JSON du module vision."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def visualize(
    image_path: Path,
    result_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    result = json.loads(result_path.read_text(encoding="utf-8"))

    if result.get("schema_version") != "1.0":
        raise ValueError("Version du JSON non prise en charge.")

    if result["image"]["format"] not in {"JPEG", "PNG"}:
        raise ValueError("Cette visualisation accepte uniquement JPEG et PNG.")

    with Image.open(image_path) as source:
        if source.format not in {"JPEG", "PNG"}:
            raise ValueError("L'image fournie doit être un JPEG ou un PNG.")

        image = ImageOps.exif_transpose(source).convert("RGB")

    expected = (
        result["image"]["width_px"],
        result["image"]["height_px"],
    )
    if image.size != expected:
        raise ValueError(
            f"Dimensions incompatibles : image={image.size}, JSON={expected}. "
            "Utiliser la photographie originale, sans redimensionnement."
        )

    objects = result["objects"]

    if result["object_count"] != len(objects):
        raise ValueError("object_count ne correspond pas au nombre de résultats.")

    for obj in objects:
        x = float(obj["pixel"]["x"])
        y = float(obj["pixel"]["y"])

        if not (
            math.isfinite(x)
            and math.isfinite(y)
            and -0.5 <= x < image.width - 0.5
            and -0.5 <= y < image.height - 0.5
        ):
            raise ValueError(f"Position pixel invalide pour {obj['doc_id']}.")

    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_path = output_dir / f"{result_path.stem}_annotated.png"
    csv_path = output_dir / f"{result_path.stem}_objects.csv"

    for path in (annotated_path, csv_path):
        if path.exists():
            raise FileExistsError(
                f"Le fichier existe déjà : {path}. "
                "Choisir un autre dossier de sortie ou déplacer ce fichier."
            )

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)

    for obj in objects:
        x = float(obj["pixel"]["x"])
        y = float(obj["pixel"]["y"])
        label = obj["doc_id"]

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
                    obj["doc_id"],
                    obj["title"],
                    metadata.get("object_type"),
                    metadata.get("magnitude"),
                    pixel["x"],
                    pixel["y"],
                    pixel["distance_to_edge_px"],
                ]
            )

    print(f"Statut du résultat : {result['status']}")
    print(f"Objets annotés : {len(objects)}")
    print(f"Erreurs de projection : {len(result['projection_errors'])}")

    for warning in result.get("warnings", []):
        print(f"Avertissement : {warning}")

    print(f"Image annotée : {annotated_path}")
    print(f"Tableau des objets : {csv_path}")

    return annotated_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Annoter les centres catalogués sur une photographie JPEG/PNG "
            "et exporter leur tableau CSV."
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
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Erreur : {exc}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
