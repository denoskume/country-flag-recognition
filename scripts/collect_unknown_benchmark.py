"""Collect an external unknown benchmark from CIFAR-100 test images.

The images are natural non-flag examples from a dataset that is independent of the
flag training sources. They are stored locally under the external benchmark tree.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random

from torchvision.datasets import CIFAR100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/external_benchmark/unknown/non_flags"),
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/external_sources/cifar100"),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.count < 100:
        raise ValueError("--count should be at least 100 for a useful unknown benchmark.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.download_dir.mkdir(parents=True, exist_ok=True)

    dataset = CIFAR100(
        root=args.download_dir,
        train=False,
        download=True,
    )

    if args.count > len(dataset):
        raise ValueError(
            f"--count={args.count} exceeds CIFAR-100 test size {len(dataset)}."
        )

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), args.count)

    saved = 0
    class_counts: dict[str, int] = {}

    for output_index, dataset_index in enumerate(indices):
        image, target = dataset[dataset_index]
        class_name = dataset.classes[target]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

        path = (
            args.output_dir
            / f"{output_index:04d}_{class_name}_{dataset_index:05d}.png"
        )
        image.save(path, format="PNG")
        saved += 1

    print(f"Unknown images saved : {saved}")
    print(f"Distinct CIFAR classes: {len(class_counts)}")
    print(f"Output directory      : {args.output_dir}")
    print()
    print("Source: CIFAR-100 test split")
    print("Purpose: external non-flag rejection benchmark")
    print("These images are not used for training.")


if __name__ == "__main__":
    main()
