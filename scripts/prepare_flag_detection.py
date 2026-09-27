"""Prepare a one-class flag-detection dataset from Flagnet annotations.

This dataset is separate from the recognition challenge set. It downloads the
original Flagnet images and Pascal-VOC boxes, then writes a detection manifest
with image paths and bounding boxes for generic flag localization.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlretrieve
import xml.etree.ElementTree as ET
import zipfile


FLAGNET_ARCHIVE = (
    "https://github.com/iamvukasin/flagnet/"
    "archive/refs/heads/master.zip"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/flag_detection"),
    )
    return parser.parse_args()


def read_boxes(annotation_path: Path):
    root = ET.parse(annotation_path).getroot()
    boxes = []

    for object_node in root.findall("object"):
        box = object_node.find("bndbox")
        if box is None:
            continue

        xmin = int(float(box.findtext("xmin", "0")))
        ymin = int(float(box.findtext("ymin", "0")))
        xmax = int(float(box.findtext("xmax", "0")))
        ymax = int(float(box.findtext("ymax", "0")))

        if xmax > xmin and ymax > ymin:
            boxes.append((xmin, ymin, xmax, ymax))

    return boxes


def main():
    args = parse_args()
    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        archive_path = tmp / "flagnet.zip"

        print("Downloading Flagnet...")
        urlretrieve(FLAGNET_ARCHIVE, archive_path)

        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(tmp)

        source_root = tmp / "flagnet-master" / "dataset"
        if not source_root.is_dir():
            raise RuntimeError("Flagnet dataset not found in archive.")

        image_index = 0

        for country_dir in sorted(
            path for path in source_root.iterdir() if path.is_dir()
        ):
            for annotation_path in sorted(country_dir.glob("*.xml")):
                image_path = country_dir / f"{annotation_path.stem}.jpg"
                if not image_path.is_file():
                    continue

                boxes = read_boxes(annotation_path)
                if not boxes:
                    continue

                output_name = (
                    f"{image_index:05d}_"
                    f"{country_dir.name.lower()}_"
                    f"{image_path.name}"
                )
                output_path = images_dir / output_name
                shutil.copy2(image_path, output_path)

                for box_index, (xmin, ymin, xmax, ymax) in enumerate(boxes):
                    rows.append({
                        "image_path": output_path.as_posix(),
                        "box_index": box_index,
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "label": "flag",
                    })

                image_index += 1

    manifest = args.output_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path", "box_index",
                "xmin", "ymin", "xmax", "ymax", "label",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Detection images : {image_index}")
    print(f"Flag boxes       : {len(rows)}")
    print(f"Manifest         : {manifest}")


if __name__ == "__main__":
    main()
