"""Expand the dataset to worldwide coverage with presentation diversity."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import tempfile
from urllib.request import urlretrieve
import zipfile

from PIL import (
    Image,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageOps,
)
import pycountry


ISO_FLAGS_ARCHIVE = (
    "https://github.com/emcrisostomo/flags/"
    "archive/refs/heads/master.zip"
)
KOSOVO_SVG_URL = (
    "https://raw.githubusercontent.com/lipis/flag-icons/"
    "main/flags/4x3/xk.svg"
)

NEUTRAL = (245, 245, 245)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
    )
    parser.add_argument(
        "--taxonomy-path",
        type=Path,
        default=Path("data/taxonomy.csv"),
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path(
            "data/source_metadata/worldwide_coverage.csv"
        ),
    )
    parser.add_argument(
        "--variants-per-class",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    return parser.parse_args()


def to_rgb_preserving_alpha(
    image: Image.Image,
    background: tuple[int, int, int] = NEUTRAL,
) -> Image.Image:
    """Composite transparent/non-rectangular flags onto a neutral background."""
    rgba = image.convert("RGBA")
    canvas = Image.new(
        "RGBA",
        rgba.size,
        (*background, 255),
    )
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")


def fit_on_canvas(
    image: Image.Image,
    size: tuple[int, int],
    background: tuple[int, int, int] = NEUTRAL,
    margin_fraction: float = 0.08,
) -> Image.Image:
    """Fit a full flag onto a canvas without stretching its proportions."""
    image = to_rgb_preserving_alpha(
        image,
        background,
    )

    canvas = Image.new(
        "RGB",
        size,
        background,
    )

    margin_x = round(
        size[0] * margin_fraction
    )
    margin_y = round(
        size[1] * margin_fraction
    )

    fitted = ImageOps.contain(
        image,
        (
            max(1, size[0] - 2 * margin_x),
            max(1, size[1] - 2 * margin_y),
        ),
        method=Image.Resampling.LANCZOS,
    )

    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2

    canvas.paste(
        fitted,
        (x, y),
    )

    return canvas


def apply_shape_mask(
    image: Image.Image,
    shape: str,
    background: tuple[int, int, int] = NEUTRAL,
) -> Image.Image:
    """Simulate flags shown as circular or rounded visual assets."""
    image = image.convert("RGB")
    mask = Image.new(
        "L",
        image.size,
        0,
    )
    draw = ImageDraw.Draw(mask)

    if shape == "circle":
        draw.ellipse(
            (
                0,
                0,
                image.width - 1,
                image.height - 1,
            ),
            fill=255,
        )
    elif shape == "rounded":
        radius = max(
            4,
            round(
                min(image.size) * 0.12
            ),
        )
        draw.rounded_rectangle(
            (
                0,
                0,
                image.width - 1,
                image.height - 1,
            ),
            radius=radius,
            fill=255,
        )
    else:
        raise ValueError(
            f"Unsupported shape: {shape}"
        )

    output = Image.new(
        "RGB",
        image.size,
        background,
    )
    output.paste(
        image,
        (0, 0),
        mask,
    )

    return output


def make_variants(
    canonical: Image.Image,
    count: int,
    seed: int,
) -> list[tuple[str, Image.Image]]:
    """Create deterministic identity-preserving presentation variants."""
    if count < 6:
        raise ValueError(
            "count must be at least 6."
        )

    rng = random.Random(seed)

    base_landscape = fit_on_canvas(
        canonical,
        (360, 240),
    )

    variants: list[
        tuple[str, Image.Image]
    ] = [
        (
            "canonical_landscape",
            base_landscape,
        ),
        (
            "portrait_frame",
            fit_on_canvas(
                canonical,
                (240, 360),
            ),
        ),
        (
            "square_frame",
            fit_on_canvas(
                canonical,
                (320, 320),
            ),
        ),
        (
            "rotated_90",
            base_landscape.rotate(
                90,
                resample=Image.Resampling.BICUBIC,
                expand=True,
                fillcolor=NEUTRAL,
            ),
        ),
        (
            "circle_mask",
            apply_shape_mask(
                fit_on_canvas(
                    canonical,
                    (320, 320),
                ),
                "circle",
            ),
        ),
        (
            "rounded_mask",
            apply_shape_mask(
                fit_on_canvas(
                    canonical,
                    (360, 240),
                ),
                "rounded",
            ),
        ),
        (
            "rotated_small",
            base_landscape.rotate(
                rng.choice(
                    [-18, -12, 12, 18]
                ),
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor=NEUTRAL,
            ),
        ),
        (
            "low_resolution",
            base_landscape.resize(
                (144, 96),
                Image.Resampling.BILINEAR,
            ).resize(
                base_landscape.size,
                Image.Resampling.BILINEAR,
            ),
        ),
        (
            "brightness_low",
            ImageEnhance.Brightness(
                base_landscape
            ).enhance(0.78),
        ),
        (
            "contrast_high",
            ImageEnhance.Contrast(
                base_landscape
            ).enhance(1.22),
        ),
        (
            "blur",
            base_landscape.filter(
                ImageFilter.GaussianBlur(
                    radius=1.1
                )
            ),
        ),
    ]

    # Partial visibility: remove a small edge region, then restore output size.
    crop = base_landscape.crop(
        (
            round(
                base_landscape.width
                * 0.08
            ),
            0,
            base_landscape.width,
            base_landscape.height,
        )
    )
    variants.append(
        (
            "partial_visibility",
            ImageOps.pad(
                crop,
                base_landscape.size,
                method=Image.Resampling.LANCZOS,
                color=NEUTRAL,
            ),
        )
    )

    if count <= len(variants):
        return variants[:count]

    # Additional requested samples combine mild geometry/photometric changes.
    while len(variants) < count:
        index = len(variants)
        image = base_landscape.rotate(
            rng.uniform(-20, 20),
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=NEUTRAL,
        )
        image = ImageEnhance.Brightness(
            image
        ).enhance(
            rng.uniform(0.82, 1.18)
        )

        variants.append(
            (
                f"combined_{index:02d}",
                image,
            )
        )

    return variants


def write_taxonomy(
    path: Path,
    entries: list[dict[str, str]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class_code",
                "name",
                "taxonomy",
                "iso_alpha3",
                "iso_numeric",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(entries)


def save_variants(
    image: Image.Image,
    class_dir: Path,
    class_code: str,
    class_name: str,
    count: int,
    seed: int,
    source_dataset: str,
    source_license: str,
    source_url: str,
    metadata: list[dict[str, str]],
) -> None:
    class_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    variants = make_variants(
        image,
        count=count,
        seed=seed,
    )

    for index, (
        presentation_variant,
        variant,
    ) in enumerate(variants):
        output_path = (
            class_dir
            / f"canonical_{index:02d}.jpg"
        )

        variant.save(
            output_path,
            format="JPEG",
            quality=94,
        )

        metadata.append(
            {
                "derived_path": (
                    output_path.as_posix()
                ),
                "class_code": (
                    class_code
                ),
                "name": class_name,
                "source_type": (
                    "canonical_augmented"
                ),
                "presentation_variant": (
                    presentation_variant
                ),
                "source_dataset": (
                    source_dataset
                ),
                "source_license": (
                    source_license
                ),
                "source_url": (
                    source_url
                ),
            }
        )


def main() -> None:
    args = parse_args()

    if args.variants_per_class < 6:
        raise ValueError(
            "--variants-per-class must be at least 6."
        )

    args.raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    taxonomy: list[
        dict[str, str]
    ] = []
    metadata: list[
        dict[str, str]
    ] = []

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary = Path(
            temporary_directory
        )

        iso_archive = (
            temporary
            / "iso_flags.zip"
        )

        print(
            "Downloading public-domain ISO flag archive..."
        )

        urlretrieve(
            ISO_FLAGS_ARCHIVE,
            iso_archive,
        )

        with zipfile.ZipFile(
            iso_archive
        ) as archive:
            archive.extractall(
                temporary
            )

        iso_root = (
            temporary
            / "flags-master"
            / "png"
            / "512"
        )

        if not iso_root.is_dir():
            raise RuntimeError(
                "ISO canonical flag directory was not found."
            )

        for country in sorted(
            pycountry.countries,
            key=lambda item: item.alpha_2,
        ):
            code_upper = (
                country.alpha_2
            )
            code = (
                code_upper.lower()
            )
            source_path = (
                iso_root
                / f"{code_upper}.png"
            )

            if not source_path.is_file():
                raise FileNotFoundError(
                    "Canonical flag missing for "
                    f"ISO code {code_upper}."
                )

            with Image.open(
                source_path
            ) as image:
                save_variants(
                    image=image.copy(),
                    class_dir=(
                        args.raw_dir
                        / code
                    ),
                    class_code=code,
                    class_name=country.name,
                    count=(
                        args.variants_per_class
                    ),
                    seed=(
                        args.seed
                        + int(
                            country.numeric
                        )
                    ),
                    source_dataset=(
                        "emcrisostomo/flags"
                    ),
                    source_license=(
                        "Public Domain"
                    ),
                    source_url=(
                        "https://github.com/"
                        "emcrisostomo/flags"
                    ),
                    metadata=metadata,
                )

            taxonomy.append(
                {
                    "class_code": code,
                    "name": country.name,
                    "taxonomy": (
                        "ISO 3166-1"
                    ),
                    "iso_alpha3": (
                        country.alpha_3
                    ),
                    "iso_numeric": (
                        country.numeric
                    ),
                    "notes": "",
                }
            )

        # Kosovo is separate because XK is not an official ISO 3166-1 code.
        kosovo_svg = (
            temporary
            / "xk.svg"
        )

        urlretrieve(
            KOSOVO_SVG_URL,
            kosovo_svg,
        )

        try:
            import cairosvg
        except ImportError as exc:
            raise RuntimeError(
                "CairoSVG is required to rasterize Kosovo."
            ) from exc

        kosovo_png = (
            temporary
            / "xk.png"
        )

        cairosvg.svg2png(
            url=str(
                kosovo_svg
            ),
            write_to=str(
                kosovo_png
            ),
            output_width=512,
        )

        with Image.open(
            kosovo_png
        ) as image:
            save_variants(
                image=image.copy(),
                class_dir=(
                    args.raw_dir
                    / "xk"
                ),
                class_code="xk",
                class_name="Kosovo",
                count=(
                    args.variants_per_class
                ),
                seed=(
                    args.seed
                    + 999
                ),
                source_dataset=(
                    "lipis/flag-icons"
                ),
                source_license="MIT",
                source_url=(
                    "https://github.com/"
                    "lipis/flag-icons"
                ),
                metadata=metadata,
            )

        taxonomy.append(
            {
                "class_code": "xk",
                "name": "Kosovo",
                "taxonomy": (
                    "Extra non-ISO class"
                ),
                "iso_alpha3": "",
                "iso_numeric": "",
                "notes": (
                    "XK is an unofficial code; "
                    "it is not an official ISO "
                    "3166-1 assignment."
                ),
            }
        )

    write_taxonomy(
        args.taxonomy_path,
        sorted(
            taxonomy,
            key=lambda row: (
                row["class_code"]
            ),
        ),
    )

    with args.metadata_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "derived_path",
                "class_code",
                "name",
                "source_type",
                "presentation_variant",
                "source_dataset",
                "source_license",
                "source_url",
            ],
        )
        writer.writeheader()
        writer.writerows(
            metadata
        )

    print(
        f"Worldwide classes   : {len(taxonomy)}"
    )
    print(
        "ISO classes         : "
        f"{sum(row['taxonomy'] == 'ISO 3166-1' for row in taxonomy)}"
    )
    print(
        "Extra classes       : 1 (Kosovo / XK)"
    )
    print(
        "Coverage images     : "
        f"{len(metadata)}"
    )
    print(
        "Presentation forms  : "
        "landscape, portrait, square, "
        "90-degree rotation, circle, "
        "rounded, small rotation, "
        "low-resolution, photometric, "
        "blur, partial visibility"
    )
    print(
        f"Taxonomy saved      : {args.taxonomy_path}"
    )
    print(
        f"Metadata saved      : {args.metadata_path}"
    )


if __name__ == "__main__":
    main()
