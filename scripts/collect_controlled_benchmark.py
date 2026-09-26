"""Collect a controlled external benchmark from an independent flag source.

This benchmark is intentionally controlled: each class uses a canonical flag from
hampusborgos/country-flags and derives presentation variants locally. It tests
cross-source visual generalization and presentation robustness. It is not a
replacement for an independent real-world photo benchmark.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import random
import urllib.request

import cairosvg
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from flag_recognition.inference import load_inference_bundle


SOURCE_TEMPLATE = (
    "https://raw.githubusercontent.com/"
    "hampusborgos/country-flags/main/svg/{code}.svg"
)
BACKGROUND = (246, 247, 249)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/models/worldwide_mobilenet_v3_small.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/external_benchmark/known"),
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=3,
    )
    return parser.parse_args()


def fetch_svg(code: str) -> bytes:
    request = urllib.request.Request(
        SOURCE_TEMPLATE.format(code=code.lower()),
        headers={
            "User-Agent": "country-flag-recognition-benchmark/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read()


def render_svg(svg: bytes) -> Image.Image:
    png = cairosvg.svg2png(
        bytestring=svg,
        output_width=640,
    )
    return Image.open(io.BytesIO(png)).convert("RGB")


def fit_canvas(
    image: Image.Image,
    size: tuple[int, int],
    margin: float = 0.08,
) -> Image.Image:
    canvas = Image.new("RGB", size, BACKGROUND)
    max_size = (
        max(1, round(size[0] * (1 - 2 * margin))),
        max(1, round(size[1] * (1 - 2 * margin))),
    )
    fitted = ImageOps.contain(
        image,
        max_size,
        method=Image.Resampling.LANCZOS,
    )
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def build_variants(
    image: Image.Image,
    code: str,
    count: int,
) -> list[tuple[str, Image.Image]]:
    rng = random.Random(10_000 + sum(ord(ch) for ch in code))
    landscape = fit_canvas(image, (420, 280))
    portrait = fit_canvas(image, (280, 420))
    angled = landscape.rotate(
        rng.choice([-16, -12, 12, 16]),
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=BACKGROUND,
    )
    angled = ImageEnhance.Brightness(angled).enhance(
        rng.uniform(0.86, 1.12)
    )
    degraded = landscape.resize(
        (168, 112),
        Image.Resampling.BILINEAR,
    ).resize(
        landscape.size,
        Image.Resampling.BILINEAR,
    )
    degraded = degraded.filter(ImageFilter.GaussianBlur(radius=0.8))

    variants = [
        ("external_clean", landscape),
        ("external_portrait", portrait),
        ("external_angled", angled),
        ("external_degraded", degraded),
    ]
    return variants[:count]


def main():
    args = parse_args()
    if not 1 <= args.variants <= 4:
        raise ValueError("--variants must be between 1 and 4.")

    bundle = load_inference_bundle(args.checkpoint, device="cpu")
    classes = [
        bundle.index_to_class[index]
        for index in sorted(bundle.index_to_class)
    ]

    completed = 0
    failed: list[tuple[str, str]] = []

    for position, code in enumerate(classes, start=1):
        class_dir = args.output_dir / code
        class_dir.mkdir(parents=True, exist_ok=True)

        try:
            svg = fetch_svg(code)
            image = render_svg(svg)
            variants = build_variants(
                image=image,
                code=code,
                count=args.variants,
            )

            for name, variant in variants:
                variant.save(
                    class_dir / f"{name}.png",
                    format="PNG",
                    optimize=True,
                )

            completed += 1
            print(
                f"[{position:03d}/{len(classes):03d}] "
                f"{code}: {len(variants)} image(s)"
            )
        except Exception as exc:
            failed.append((code, str(exc)))
            print(
                f"[{position:03d}/{len(classes):03d}] "
                f"{code}: FAILED — {exc}"
            )

    print()
    print(f"Classes completed : {completed}/{len(classes)}")
    print(f"Classes failed    : {len(failed)}")
    print(f"Output directory  : {args.output_dir}")
    print()
    print(
        "IMPORTANT: this is a controlled cross-source benchmark, "
        "not a real-world photo benchmark."
    )

    if failed:
        print("\nFailed classes:")
        for code, error in failed:
            print(f"  {code}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
