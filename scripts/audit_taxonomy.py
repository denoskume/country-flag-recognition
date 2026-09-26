"""Audit flag taxonomy for identical and near-identical visual classes.

The audit uses canonical images from the local training tree when available.
It computes normalized visual fingerprints and pairwise distances so benchmark
errors caused by taxonomy ambiguity can be separated from genuine model errors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from flag_recognition.inference import load_inference_bundle


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/models/worldwide_mobilenet_v3_small.pt"),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/metrics"),
    )
    parser.add_argument(
        "--near-threshold",
        type=float,
        default=0.035,
        help="Normalized visual distance threshold for near-identical flags.",
    )
    return parser.parse_args()


def canonical_candidate(class_dir: Path) -> Path:
    preferred = [
        class_dir / "canonical_00.jpg",
        class_dir / "canonical_00.png",
        class_dir / "canonical_landscape.jpg",
        class_dir / "canonical_landscape.png",
    ]
    for path in preferred:
        if path.is_file():
            return path

    images = sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and "canonical" in path.stem.lower()
    )
    if images:
        return images[0]

    fallback = sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if fallback:
        return fallback[0]

    raise FileNotFoundError(f"No image found in {class_dir}")


def normalized_rgb(path: Path, size=(192, 128)) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = ImageOps.contain(
            image,
            size,
            method=Image.Resampling.LANCZOS,
        )
        canvas = Image.new("RGB", size, (245, 245, 245))
        x = (size[0] - image.width) // 2
        y = (size[1] - image.height) // 2
        canvas.paste(image, (x, y))
        return np.asarray(canvas, dtype=np.float32) / 255.0


def exact_fingerprint(array: np.ndarray) -> str:
    quantized = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    return hashlib.sha256(quantized.tobytes()).hexdigest()


def visual_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_inference_bundle(args.checkpoint, device="cpu")
    classes = [
        bundle.index_to_class[index]
        for index in sorted(bundle.index_to_class)
    ]

    images: dict[str, np.ndarray] = {}
    source_paths: dict[str, str] = {}
    missing: list[str] = []

    for code in classes:
        class_dir = args.raw_dir / code
        if not class_dir.is_dir():
            missing.append(code)
            continue
        try:
            path = canonical_candidate(class_dir)
            images[code] = normalized_rgb(path)
            source_paths[code] = path.as_posix()
        except FileNotFoundError:
            missing.append(code)

    exact_groups: dict[str, list[str]] = {}
    for code, image in images.items():
        exact_groups.setdefault(exact_fingerprint(image), []).append(code)

    identical_groups = [
        sorted(group)
        for group in exact_groups.values()
        if len(group) > 1
    ]
    identical_pairs = {
        tuple(sorted(pair))
        for group in identical_groups
        for pair in itertools.combinations(group, 2)
    }

    pair_rows = []
    near_pairs = []

    for left, right in itertools.combinations(sorted(images), 2):
        distance = visual_distance(images[left], images[right])
        relation = "distinct"

        if (left, right) in identical_pairs:
            relation = "identical"
        elif distance <= args.near_threshold:
            relation = "near_identical"
            near_pairs.append(
                {
                    "class_a": left,
                    "class_b": right,
                    "distance": distance,
                }
            )

        pair_rows.append(
            {
                "class_a": left,
                "class_b": right,
                "distance": distance,
                "relation": relation,
                "source_a": source_paths[left],
                "source_b": source_paths[right],
            }
        )

    pair_rows.sort(key=lambda row: row["distance"])
    near_pairs.sort(key=lambda row: row["distance"])

    csv_path = args.output_dir / "taxonomy_visual_similarity.csv"
    json_path = args.output_dir / "taxonomy_ambiguity_audit.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class_a",
                "class_b",
                "distance",
                "relation",
                "source_a",
                "source_b",
            ],
        )
        writer.writeheader()
        writer.writerows(pair_rows)

    report = {
        "checkpoint": args.checkpoint.as_posix(),
        "classes_expected": len(classes),
        "classes_audited": len(images),
        "missing_classes": missing,
        "near_threshold": args.near_threshold,
        "identical_groups": identical_groups,
        "near_identical_pairs": near_pairs,
        "closest_pairs": pair_rows[:50],
        "outputs": {
            "pairwise_csv": csv_path.as_posix(),
            "report_json": json_path.as_posix(),
        },
    }

    json_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
