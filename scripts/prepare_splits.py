"""Generate a reproducible seen/unseen split manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from flag_recognition.dataset import discover_country_images
from flag_recognition.splits import SplitConfig, build_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data_cfg = config["data"]
    project_cfg = config["project"]

    class_to_images = discover_country_images(
        raw_dir=Path(data_cfg["raw_dir"]),
        allowed_extensions=set(
            data_cfg["allowed_extensions"]
        ),
    )

    split_config = SplitConfig(
        unseen_fraction=float(
            data_cfg["unseen_fraction"]
        ),
        validation_fraction=float(
            data_cfg["validation_fraction"]
        ),
        test_fraction=float(
            data_cfg["test_fraction"]
        ),
        minimum_images_per_seen_class=int(
            data_cfg["minimum_images_per_seen_class"]
        ),
        seed=int(project_cfg["seed"]),
    )

    manifest = build_split_manifest(
        class_to_images=class_to_images,
        config=split_config,
    )

    output_path = Path(
        data_cfg["split_manifest"]
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    manifest.to_csv(
        output_path,
        index=False,
    )

    seen_classes = sorted(
        manifest.loc[
            manifest["regime"] == "seen",
            "country",
        ].unique()
    )
    unseen_classes = sorted(
        manifest.loc[
            manifest["regime"] == "unseen",
            "country",
        ].unique()
    )

    print(f"Country classes discovered : {len(class_to_images)}")
    print(f"Seen classes              : {len(seen_classes)}")
    print(f"Unseen classes            : {len(unseen_classes)}")
    print(f"Manifest rows             : {len(manifest)}")
    print(f"Saved                     : {output_path}")


if __name__ == "__main__":
    main()
