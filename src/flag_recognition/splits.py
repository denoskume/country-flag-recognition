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
        raise ValueError(
            f"{name} must be in [0, 1)."
        )


def is_canonical_variant(
    path: Path,
) -> bool:
    """Identify generated coverage samples by their reserved filename prefix."""
    return Path(path).name.startswith(
        "canonical_"
    )


def choose_unseen_classes(
    classes: list[str],
    unseen_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Partition country labels into seen and unseen classes."""
    _validate_fraction(
        "unseen_fraction",
        unseen_fraction,
    )

    unique_classes = sorted(
        set(classes)
    )

    if len(unique_classes) < 2:
        raise ValueError(
            "At least two country classes are required."
        )

    unseen_count = max(
        1,
        round(
            len(unique_classes)
            * unseen_fraction
        ),
    )
    unseen_count = min(
        unseen_count,
        len(unique_classes) - 1,
    )

    rng = random.Random(seed)
    shuffled = unique_classes.copy()
    rng.shuffle(shuffled)

    unseen = sorted(
        shuffled[:unseen_count]
    )
    seen = sorted(
        shuffled[unseen_count:]
    )

    return seen, unseen


def split_seen_class(
    image_paths: list[Path],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[Path]]:
    """Split independent real-world source images for one seen class."""
    _validate_fraction(
        "validation_fraction",
        validation_fraction,
    )
    _validate_fraction(
        "test_fraction",
        test_fraction,
    )

    if (
        validation_fraction
        + test_fraction
        >= 1.0
    ):
        raise ValueError(
            "validation_fraction + test_fraction "
            "must be smaller than 1."
        )

    ordered = sorted(
        Path(path)
        for path in image_paths
    )

    if len(ordered) < 3:
        raise ValueError(
            "At least three independent source images "
            "are required for train/validation/test."
        )

    rng = random.Random(seed)
    shuffled = ordered.copy()
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_validation = max(
        1,
        round(
            n_total
            * validation_fraction
        ),
    )
    n_test = max(
        1,
        round(
            n_total
            * test_fraction
        ),
    )

    while (
        n_validation
        + n_test
        >= n_total
    ):
        if (
            n_validation >= n_test
            and n_validation > 1
        ):
            n_validation -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            raise ValueError(
                "Not enough independent source images "
                "to form train/validation/test."
            )

    validation = shuffled[
        :n_validation
    ]
    test = shuffled[
        n_validation:
        n_validation + n_test
    ]
    train = shuffled[
        n_validation + n_test:
    ]

    return {
        "train": sorted(train),
        "validation": sorted(
            validation
        ),
        "test": sorted(test),
    }


def _row(
    path: Path,
    country: str,
    regime: str,
    partition: str,
    source_type: str,
    evaluation_scope: str,
) -> dict[str, str]:
    return {
        "path": Path(path).as_posix(),
        "country": country,
        "regime": regime,
        "partition": partition,
        "source_type": source_type,
        "evaluation_scope": evaluation_scope,
    }


def build_split_manifest(
    class_to_images: dict[
        str,
        list[Path],
    ],
    config: SplitConfig,
) -> pd.DataFrame:
    """Create a leakage-aware worldwide seen/unseen manifest.

    Generated canonical presentation variants are training-only for seen
    classes. Validation/test for seen classes use independent real-world
    images whenever available. Unseen classes are absent from supervised
    training by construction.
    """
    if not class_to_images:
        raise ValueError(
            "No country classes were provided."
        )

    classes = sorted(
        class_to_images
    )

    (
        seen_classes,
        unseen_classes,
    ) = choose_unseen_classes(
        classes=classes,
        unseen_fraction=(
            config.unseen_fraction
        ),
        seed=config.seed,
    )

    rows: list[
        dict[str, str]
    ] = []

    for (
        class_index,
        country,
    ) in enumerate(seen_classes):
        all_paths = sorted(
            Path(path)
            for path
            in class_to_images[
                country
            ]
        )

        canonical_paths = [
            path
            for path in all_paths
            if is_canonical_variant(
                path
            )
        ]
        real_paths = [
            path
            for path in all_paths
            if not is_canonical_variant(
                path
            )
        ]

        if (
            len(all_paths)
            < config.minimum_images_per_seen_class
        ):
            raise ValueError(
                f"Seen class '{country}' has "
                f"{len(all_paths)} image(s); "
                "minimum is "
                f"{config.minimum_images_per_seen_class}."
            )

        # Canonical presentation variants support training only.
        rows.extend(
            _row(
                path=path,
                country=country,
                regime="seen",
                partition="train",
                source_type=(
                    "canonical_augmented"
                ),
                evaluation_scope=(
                    "training_support"
                ),
            )
            for path
            in canonical_paths
        )

        if len(real_paths) >= 3:
            partitions = (
                split_seen_class(
                    image_paths=(
                        real_paths
                    ),
                    validation_fraction=(
                        config.validation_fraction
                    ),
                    test_fraction=(
                        config.test_fraction
                    ),
                    seed=(
                        config.seed
                        + class_index
                    ),
                )
            )

            for (
                partition,
                paths,
            ) in partitions.items():
                scope = (
                    "real_world_benchmark"
                    if partition
                    in {
                        "validation",
                        "test",
                    }
                    else "training_real_world"
                )

                rows.extend(
                    _row(
                        path=path,
                        country=country,
                        regime="seen",
                        partition=(
                            partition
                        ),
                        source_type=(
                            "real_world"
                        ),
                        evaluation_scope=scope,
                    )
                    for path
                    in paths
                )
        else:
            # Coverage-only classes can still be learned, but they are not
            # treated as independently validated real-world classes.
            rows.extend(
                _row(
                    path=path,
                    country=country,
                    regime="seen",
                    partition="train",
                    source_type=(
                        "real_world"
                    ),
                    evaluation_scope=(
                        "training_real_world"
                    ),
                )
                for path
                in real_paths
            )

    for country in unseen_classes:
        for path in sorted(
            Path(path)
            for path
            in class_to_images[
                country
            ]
        ):
            canonical = (
                is_canonical_variant(
                    path
                )
            )

            rows.append(
                _row(
                    path=path,
                    country=country,
                    regime="unseen",
                    partition="test",
                    source_type=(
                        "canonical_augmented"
                        if canonical
                        else "real_world"
                    ),
                    evaluation_scope=(
                        "controlled_unknown"
                        if canonical
                        else "real_world_unknown"
                    ),
                )
            )

    manifest = pd.DataFrame(
        rows
    )

    if manifest.empty:
        raise RuntimeError(
            "The generated split manifest is empty."
        )

    return manifest.sort_values(
        [
            "regime",
            "country",
            "partition",
            "source_type",
            "path",
        ]
    ).reset_index(
        drop=True
    )
