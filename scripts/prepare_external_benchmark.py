"""Prepare the external benchmark folder structure from the deployment checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from flag_recognition.inference import load_inference_bundle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/models/worldwide_mobilenet_v3_small.pt"),
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("data/external_benchmark"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = load_inference_bundle(args.checkpoint, device="cpu")

    known_root = args.benchmark_dir / "known"
    unknown_root = args.benchmark_dir / "unknown"

    known_root.mkdir(parents=True, exist_ok=True)
    unknown_root.mkdir(parents=True, exist_ok=True)

    classes = [
        bundle.index_to_class[index]
        for index in sorted(bundle.index_to_class)
    ]

    for country in classes:
        (known_root / country).mkdir(parents=True, exist_ok=True)

    for category in [
        "non_flags",
        "logos",
        "objects",
        "scenes",
        "text_and_graphics",
    ]:
        (unknown_root / category).mkdir(parents=True, exist_ok=True)

    print(f"Known classes prepared : {len(classes)}")
    print(f"Known root             : {known_root}")
    print(f"Unknown root           : {unknown_root}")
    print()
    print("Recommended minimum:")
    print("  known   : >= 3 genuinely external images per class")
    print("  unknown : >= 100 diverse non-flag images overall")
    print()
    print("Do not copy images from data/raw.")


if __name__ == "__main__":
    main()
