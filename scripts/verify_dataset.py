"""Validate the local worldwide dataset before training."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from flag_recognition.dataset import discover_country_images
from flag_recognition.splits import is_canonical_variant


RAW_DIR = Path("data/raw")
MANIFEST_PATH = Path("data/splits/split_manifest.csv")
EXPECTED_CLASSES = 250


def main() -> None:
    class_to_images = discover_country_images(
        RAW_DIR
    )

    if len(class_to_images) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_CLASSES} classes, "
            f"found {len(class_to_images)}."
        )

    total_images = sum(
        len(paths)
        for paths in class_to_images.values()
    )

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Missing split manifest: {MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    required_columns = {
        "path",
        "country",
        "regime",
        "partition",
        "source_type",
        "evaluation_scope",
    }

    missing = required_columns - set(
        manifest.columns
    )

    if missing:
        raise RuntimeError(
            "Manifest missing columns: "
            + ", ".join(sorted(missing))
        )

    missing_files = [
        path
        for path in manifest["path"]
        if not Path(path).is_file()
    ]

    if missing_files:
        raise RuntimeError(
            f"{len(missing_files)} manifest path(s) do not exist."
        )

    if not manifest["path"].is_unique:
        raise RuntimeError(
            "The split manifest contains duplicate paths."
        )

    unseen = manifest[
        manifest["regime"] == "unseen"
    ]

    if set(unseen["partition"]) != {"test"}:
        raise RuntimeError(
            "Unseen classes must be test-only."
        )

    seen_evaluation = manifest[
        (manifest["regime"] == "seen")
        & manifest["partition"].isin(
            ["validation", "test"]
        )
    ]

    leaked_canonical = [
        path
        for path in seen_evaluation["path"]
        if is_canonical_variant(
            Path(path)
        )
    ]

    if leaked_canonical:
        raise RuntimeError(
            "Canonical variants leaked into seen validation/test."
        )

    print("Dataset verification passed.")
    print(f"Classes              : {len(class_to_images)}")
    print(f"Images               : {total_images}")
    print(
        "Seen classes         : "
        f"{manifest.loc[manifest['regime'] == 'seen', 'country'].nunique()}"
    )
    print(
        "Unseen classes       : "
        f"{manifest.loc[manifest['regime'] == 'unseen', 'country'].nunique()}"
    )

    for (
        regime,
        partition,
    ), group in manifest.groupby(
        ["regime", "partition"]
    ):
        print(
            f"{regime:6s}/{partition:10s}: {len(group)}"
        )


if __name__ == "__main__":
    main()
