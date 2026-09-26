"""Reproducible class-aware seen/unseen dataset splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    """Dataset split parameters."""

    unseen_fraction: float = 0.20
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    minimum_images_per_seen_class: int = 5
    seed: int = 42


def _validate_fraction(name: str, value: float) -> None:
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in [0, 1).")


def choose_unseen_classes(
    classes: list[str],
    unseen_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Partition country labels into seen and unseen classes."""
    _validate_fraction("unseen_fraction", unseen_fraction)

    unique_classes = sorted(set(classes))

    if len(unique_classes) < 2:
        raise ValueError("At least two country classes are required.")

    unseen_count = max(
        1,
        round(len(unique_classes) * unseen_fraction),
    )
    unseen_count = min(
        unseen_count,
        len(unique_classes) - 1,
    )

    rng = random.Random(seed)
    shuffled = unique_classes.copy()
    rng.shuffle(shuffled)

    unseen = sorted(shuffled[:unseen_count])
    seen = sorted(shuffled[unseen_count:])

    return seen, unseen


def split_seen_class(
    image_paths: list[Path],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[Path]]:
    """Split source images from one seen class before augmentation."""
    _validate_fraction("validation_fraction", validation_fraction)
    _validate_fraction("test_fraction", test_fraction)

    if validation_fraction + test_fraction >= 1.0:
        raise ValueError(
            "validation_fraction + test_fraction must be smaller than 1."
        )

    ordered = sorted(Path(path) for path in image_paths)

    if len(ordered) < 3:
        raise ValueError(
            "At least three source images are required for a seen class."
        )

    rng = random.Random(seed)
    shuffled = ordered.copy()
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_validation = max(1, round(n_total * validation_fraction))
    n_test = max(1, round(n_total * test_fraction))

    while n_validation + n_test >= n_total:
        if n_validation >= n_test and n_validation > 1:
            n_validation -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            raise ValueError(
                "Not enough source images to form train/validation/test."
            )

    validation = shuffled[:n_validation]
    test = shuffled[n_validation : n_validation + n_test]
    train = shuffled[n_validation + n_test :]

    return {
        "train": sorted(train),
        "validation": sorted(validation),
        "test": sorted(test),
    }


def build_split_manifest(
    class_to_images: dict[str, list[Path]],
    config: SplitConfig,
) -> pd.DataFrame:
    """Create the full seen/unseen manifest without image-level leakage."""
    if not class_to_images:
        raise ValueError("No country classes were provided.")

    classes = sorted(class_to_images)
    seen_classes, unseen_classes = choose_unseen_classes(
        classes=classes,
        unseen_fraction=config.unseen_fraction,
        seed=config.seed,
    )

    rows: list[dict[str, str]] = []

    for class_index, country in enumerate(seen_classes):
        image_paths = class_to_images[country]

        if len(image_paths) < config.minimum_images_per_seen_class:
            raise ValueError(
                f"Seen class '{country}' has {len(image_paths)} image(s); "
                f"minimum is {config.minimum_images_per_seen_class}."
            )

        partitions = split_seen_class(
            image_paths=image_paths,
            validation_fraction=config.validation_fraction,
            test_fraction=config.test_fraction,
            seed=config.seed + class_index,
        )

        for partition, paths in partitions.items():
            rows.extend(
                {
                    "path": path.as_posix(),
                    "country": country,
                    "regime": "seen",
                    "partition": partition,
                }
                for path in paths
            )

    for country in unseen_classes:
        rows.extend(
            {
                "path": Path(path).as_posix(),
                "country": country,
                "regime": "unseen",
                "partition": "test",
            }
            for path in sorted(class_to_images[country])
        )

    manifest = pd.DataFrame(rows)

    if manifest.empty:
        raise RuntimeError("The generated split manifest is empty.")

    return manifest.sort_values(
        ["regime", "country", "partition", "path"]
    ).reset_index(drop=True)
