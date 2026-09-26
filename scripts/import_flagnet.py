"""Import annotated real-world flag crops from the Flagnet dataset.

The importer downloads the public Flagnet repository archive, reads its Pascal-VOC
bounding boxes, crops each annotated flag, and stores provenance for every derived
image. Dataset files remain local and are ignored by Git.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import tempfile
from urllib.request import urlretrieve
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image
import yaml


FLAGNET_ARCHIVE = (
    "https://github.com/iamvukasin/flagnet/"
    "archive/refs/heads/master.zip"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path(
            "data/source_metadata/flagnet_credits.csv"
        ),
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.10,
        help="Fractional padding added around each annotated bounding box.",
    )
    return parser.parse_args()


def padded_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    padding: float,
) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = box
    width = xmax - xmin
    height = ymax - ymin

    pad_x = round(width * padding)
    pad_y = round(height * padding)

    image_width, image_height = image_size

    return (
        max(0, xmin - pad_x),
        max(0, ymin - pad_y),
        min(image_width, xmax + pad_x),
        min(image_height, ymax + pad_y),
    )


def read_boxes(annotation_path: Path) -> list[tuple[int, int, int, int]]:
    root = ET.parse(annotation_path).getroot()
    boxes: list[tuple[int, int, int, int]] = []

    for object_node in root.findall("object"):
        box_node = object_node.find("bndbox")

        if box_node is None:
            continue

        values = tuple(
            int(float(box_node.findtext(name, "0")))
            for name in ("xmin", "ymin", "xmax", "ymax")
        )

        xmin, ymin, xmax, ymax = values

        if xmax > xmin and ymax > ymin:
            boxes.append(values)

    return boxes


def load_credit_map(country_dir: Path) -> dict[str, dict[str, str]]:
    credits_path = country_dir / "credits.yml"

    if not credits_path.is_file():
        return {}

    with credits_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        credits = yaml.safe_load(handle) or {}

    return {
        str(photo.get("filename", "")): {
            "author": str(photo.get("author", "")),
            "license": str(photo.get("license", "")),
            "source_url": str(photo.get("url", "")),
            "download_url": str(photo.get("download_url", "")),
        }
        for photo in credits.get("photos", [])
    }


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.padding <= 0.50:
        raise ValueError("--padding must be between 0.0 and 0.50.")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_rows: list[dict[str, str | int]] = []

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary = Path(
            temporary_directory
        )
        archive_path = (
            temporary / "flagnet.zip"
        )

        print("Downloading Flagnet dataset archive...")
        urlretrieve(
            FLAGNET_ARCHIVE,
            archive_path,
        )

        with zipfile.ZipFile(
            archive_path
        ) as archive:
            archive.extractall(
                temporary
            )

        source_root = (
            temporary
            / "flagnet-master"
            / "dataset"
        )

        if not source_root.is_dir():
            raise RuntimeError(
                "Flagnet dataset directory was not found in the archive."
            )

        country_dirs = sorted(
            path
            for path in source_root.iterdir()
            if path.is_dir()
        )

        for country_dir in country_dirs:
            country_code = (
                country_dir.name.lower()
            )
            output_country_dir = (
                args.output_dir
                / country_code
            )
            output_country_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            credit_map = load_credit_map(
                country_dir
            )

            annotation_paths = sorted(
                country_dir.glob("*.xml")
            )

            country_crops = 0

            for annotation_path in annotation_paths:
                image_path = (
                    country_dir
                    / (
                        annotation_path.stem
                        + ".jpg"
                    )
                )

                if not image_path.is_file():
                    continue

                boxes = read_boxes(
                    annotation_path
                )

                if not boxes:
                    continue

                credit = credit_map.get(
                    image_path.name,
                    {},
                )

                with Image.open(
                    image_path
                ) as source_image:
                    source_image = (
                        source_image
                        .convert("RGB")
                    )

                    for object_index, box in enumerate(boxes):
                        crop_box = padded_box(
                            box=box,
                            image_size=source_image.size,
                            padding=args.padding,
                        )
                        crop = source_image.crop(
                            crop_box
                        )

                        crop_name = (
                            f"{image_path.stem}"
                            f"_obj{object_index:02d}.jpg"
                        )
                        output_path = (
                            output_country_dir
                            / crop_name
                        )

                        crop.save(
                            output_path,
                            quality=95,
                        )

                        metadata_rows.append(
                            {
                                "derived_path": (
                                    output_path.as_posix()
                                ),
                                "country_code": (
                                    country_code
                                ),
                                "source_filename": (
                                    image_path.name
                                ),
                                "object_index": (
                                    object_index
                                ),
                                "author": credit.get(
                                    "author",
                                    "",
                                ),
                                "license": credit.get(
                                    "license",
                                    "",
                                ),
                                "source_url": credit.get(
                                    "source_url",
                                    "",
                                ),
                                "download_url": credit.get(
                                    "download_url",
                                    "",
                                ),
                                "source_dataset": (
                                    "iamvukasin/flagnet"
                                ),
                            }
                        )
                        country_crops += 1

            print(
                f"{country_code}: "
                f"{country_crops} crop(s)"
            )

    if not metadata_rows:
        raise RuntimeError(
            "No annotated flag crops were imported."
        )

    fieldnames = list(
        metadata_rows[0].keys()
    )

    with args.metadata_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(
            metadata_rows
        )

    print(
        f"\nImported crops : {len(metadata_rows)}"
    )
    print(
        f"Country folders: "
        f"{len(set(row['country_code'] for row in metadata_rows))}"
    )
    print(
        f"Provenance CSV : {args.metadata_path}"
    )


if __name__ == "__main__":
    main()
