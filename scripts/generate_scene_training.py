"""Generate scene-aware training images without touching the external benchmark.

Canonical/current training flags are composited onto independent CIFAR-100 train
backgrounds at varied scales, positions, rotations and photometric conditions.
The Wikimedia real-world challenge set remains strictly evaluation-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision.datasets import CIFAR100

from flag_recognition.dataset import discover_country_images
from flag_recognition.splits import is_canonical_variant


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/scene_train"),
    )
    parser.add_argument(
        "--background-dir",
        type=Path,
        default=Path("data/external_sources/cifar100_train"),
    )
    parser.add_argument("--images-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def choose_flag(paths: list[Path]) -> Path:
    canonical = [path for path in paths if is_canonical_variant(path)]
    return canonical[0] if canonical else paths[0]


def make_scene(
    flag: Image.Image,
    background: Image.Image,
    rng: random.Random,
) -> Image.Image:
    canvas_w = rng.choice([384, 448, 512])
    canvas_h = rng.choice([256, 320, 384])

    background = ImageOps.fit(
        background.convert("RGB"),
        (canvas_w, canvas_h),
        method=Image.Resampling.BICUBIC,
    )

    background = ImageEnhance.Brightness(background).enhance(
        rng.uniform(0.75, 1.20)
    )
    background = ImageEnhance.Contrast(background).enhance(
        rng.uniform(0.80, 1.20)
    )
    background = background.filter(
        ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 1.2))
    )

    flag = flag.convert("RGB")
    max_w = int(canvas_w * rng.uniform(0.18, 0.55))
    max_h = int(canvas_h * rng.uniform(0.16, 0.48))

    flag = ImageOps.contain(
        flag,
        (max_w, max_h),
        method=Image.Resampling.LANCZOS,
    )

    flag = ImageEnhance.Brightness(flag).enhance(
        rng.uniform(0.72, 1.18)
    )
    flag = ImageEnhance.Contrast(flag).enhance(
        rng.uniform(0.85, 1.20)
    )

    angle = rng.uniform(-22, 22)
    rotated = flag.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(0, 0, 0),
    )

    # Build an opacity mask from the rotated rectangle so fill pixels disappear.
    mask = Image.new("L", flag.size, 255).rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=0,
    )

    if rng.random() < 0.35:
        rotated = rotated.filter(
            ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.0))
        )

    if rotated.width >= canvas_w or rotated.height >= canvas_h:
        rotated = ImageOps.contain(
            rotated,
            (int(canvas_w * 0.7), int(canvas_h * 0.7)),
            method=Image.Resampling.LANCZOS,
        )
        mask = ImageOps.contain(
            mask,
            rotated.size,
            method=Image.Resampling.LANCZOS,
        )

    max_x = max(0, canvas_w - rotated.width)
    max_y = max(0, canvas_h - rotated.height)
    x = rng.randint(0, max_x) if max_x else 0
    y = rng.randint(0, max_y) if max_y else 0

    background.paste(rotated, (x, y), mask)
    return background


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.background_dir.mkdir(parents=True, exist_ok=True)

    backgrounds = CIFAR100(
        root=args.background_dir,
        train=True,
        download=True,
    )

    class_to_images = discover_country_images(
        raw_dir=args.raw_dir,
        allowed_extensions={".jpg", ".jpeg", ".png", ".webp"},
    )

    total = 0

    for class_index, (code, paths) in enumerate(
        sorted(class_to_images.items()),
        start=1,
    ):
        source_path = choose_flag([Path(path) for path in paths])
        class_dir = args.output_dir / code
        class_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(source_path) as source:
            flag = source.convert("RGB")

            for image_index in range(args.images_per_class):
                background_index = rng.randrange(len(backgrounds))
                background, _ = backgrounds[background_index]

                scene = make_scene(
                    flag=flag,
                    background=background,
                    rng=rng,
                )

                output = (
                    class_dir
                    / f"scene_{image_index:02d}_{background_index:05d}.jpg"
                )
                scene.save(output, quality=92)
                total += 1

        print(
            f"[{class_index:03d}/{len(class_to_images):03d}] "
            f"{code}: {args.images_per_class}"
        )

    print()
    print(f"Classes             : {len(class_to_images)}")
    print(f"Scene images        : {total}")
    print(f"Output directory    : {args.output_dir}")
    print("Background source   : CIFAR-100 train split")
    print("Challenge set usage : none")


if __name__ == "__main__":
    main()
