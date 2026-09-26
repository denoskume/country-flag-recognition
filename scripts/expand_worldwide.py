"""Expand the flag dataset to worldwide ISO 3166-1 coverage plus Kosovo.

This script supplements the real-world Flagnet crops with canonical public-domain
flags for all ISO 3166-1 entities. It also adds Kosovo as a separately documented
non-ISO class using the commonly used XK code.

Canonical images are used to create deterministic coverage variants so every
class has enough samples to enter the training pipeline. Real-world and canonical
sources are kept distinguishable in metadata for honest evaluation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import tempfile
from urllib.request import urlretrieve
import zipfile

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pycountry


ISO_FLAGS_ARCHIVE = (
    "https://github.com/emcrisostomo/flags/"
    "archive/refs/heads/master.zip"
)
KOSOVO_SVG_URL = (
    "https://raw.githubusercontent.com/lipis/flag-icons/"
    "main/flags/4x3/xk.svg"
)


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
        default=6,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    return parser.parse_args()


def fit_on_canvas(
    image: Image.Image,
    size: tuple[int, int] = (320, 240),
    background: tuple[int, int, int] = (245, 245, 245),
) -> Image.Image:
    """Fit a flag on a neutral canvas without stretching its aspect ratio."""
    canvas = Image.new("RGB", size, background)
    fitted = ImageOps.contain(
        image.convert("RGB"),
        (size[0] - 32, size[1] - 32),
        method=Image.Resampling.LANCZOS,
    )
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def make_variants(
    canonical: Image.Image,
    count: int,
    seed: int,
) -> list[Image.Image]:
    """Create deterministic, moderate visual variants from one canonical flag."""
    if count < 1:
        raise ValueError("count must be at least 1.")

    rng = random.Random(seed)
    base = fit_on_canvas(canonical)
    variants: list[Image.Image] = [base]

    operations = [
        "rotate",
        "brightness",
        "contrast",
        "blur",
        "resolution",
        "combined",
    ]

    for index in range(1, count):
        operation = operations[(index - 1) % len(operations)]
        image = base.copy()

        if operation == "rotate":
            angle = rng.choice([-8, -5, 5, 8])
            image = image.rotate(
                angle,
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor=(245, 245, 245),
            )
        elif operation == "brightness":
            image = ImageEnhance.Brightness(image).enhance(
                rng.choice([0.78, 0.88, 1.12, 1.22])
            )
        elif operation == "contrast":
            image = ImageEnhance.Contrast(image).enhance(
                rng.choice([0.82, 0.90, 1.12, 1.20])
            )
        elif operation == "blur":
            image = image.filter(
                ImageFilter.GaussianBlur(
                    radius=rng.choice([0.6, 1.0, 1.4])
                )
            )
        elif operation == "resolution":
            reduced = image.resize(
                (160, 120),
                Image.Resampling.BILINEAR,
            )
            image = reduced.resize(
                image.size,
                Image.Resampling.BILINEAR,
            )
        else:
            image = ImageEnhance.Brightness(image).enhance(0.9)
            image = image.rotate(
                rng.choice([-4, 4]),
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor=(245, 245, 245),
            )

        variants.append(image)

    return variants


def write_taxonomy(
    path: Path,
    entries: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
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


def main() -> None:
    args = parse_args()

    if args.variants_per_class < 5:
        raise ValueError(
            "--variants-per-class must be at least 5 "
            "to satisfy the current training split contract."
        )

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    taxonomy: list[dict[str, str]] = []
    metadata: list[dict[str, str]] = []

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)

        iso_archive = temporary / "iso_flags.zip"
        print("Downloading public-domain ISO flag archive...")
        urlretrieve(ISO_FLAGS_ARCHIVE, iso_archive)

        with zipfile.ZipFile(iso_archive) as archive:
            archive.extractall(temporary)

        iso_root = (
            temporary
            / "flags-master"
            / "png"
            / "512"
        )

        if not iso_root.is_dir():
            raise RuntimeError(
                "ISO canonical flag directory was not found in the archive."
            )

        for country in sorted(
            pycountry.countries,
            key=lambda item: item.alpha_2,
        ):
            code_upper = country.alpha_2
            code = code_upper.lower()
            source_path = iso_root / f"{code_upper}.png"

            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Canonical flag missing for ISO code {code_upper}."
                )

            class_dir = args.raw_dir / code
            class_dir.mkdir(parents=True, exist_ok=True)

            with Image.open(source_path) as image:
                variants = make_variants(
                    image,
                    count=args.variants_per_class,
                    seed=args.seed + int(country.numeric),
                )

            for index, variant in enumerate(variants):
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
                        "derived_path": output_path.as_posix(),
                        "class_code": code,
                        "name": country.name,
                        "source_type": "canonical_augmented",
                        "source_dataset": "emcrisostomo/flags",
                        "source_license": "Public Domain",
                        "source_url": (
                            "https://github.com/emcrisostomo/flags"
                        ),
                    }
                )

            taxonomy.append(
                {
                    "class_code": code,
                    "name": country.name,
                    "taxonomy": "ISO 3166-1",
                    "iso_alpha3": country.alpha_3,
                    "iso_numeric": country.numeric,
                    "notes": "",
                }
            )

        # Kosovo is included separately because XK is not an official ISO 3166-1 code.
        kosovo_svg = temporary / "xk.svg"
        urlretrieve(KOSOVO_SVG_URL, kosovo_svg)

        try:
            import cairosvg
        except ImportError as exc:
            raise RuntimeError(
                "CairoSVG is required to rasterize the Kosovo source flag."
            ) from exc

        kosovo_png = temporary / "xk.png"
        cairosvg.svg2png(
            url=str(kosovo_svg),
            write_to=str(kosovo_png),
            output_width=512,
        )

        with Image.open(kosovo_png) as image:
            variants = make_variants(
                image,
                count=args.variants_per_class,
                seed=args.seed + 999,
            )

        kosovo_dir = args.raw_dir / "xk"
        kosovo_dir.mkdir(parents=True, exist_ok=True)

        for index, variant in enumerate(variants):
            output_path = (
                kosovo_dir
                / f"canonical_{index:02d}.jpg"
            )
            variant.save(
                output_path,
                format="JPEG",
                quality=94,
            )

            metadata.append(
                {
                    "derived_path": output_path.as_posix(),
                    "class_code": "xk",
                    "name": "Kosovo",
                    "source_type": "canonical_augmented",
                    "source_dataset": "lipis/flag-icons",
                    "source_license": "MIT",
                    "source_url": (
                        "https://github.com/lipis/flag-icons"
                    ),
                }
            )

        taxonomy.append(
            {
                "class_code": "xk",
                "name": "Kosovo",
                "taxonomy": "Extra non-ISO class",
                "iso_alpha3": "",
                "iso_numeric": "",
                "notes": (
                    "XK is an unofficial code used in some international "
                    "and European data systems; it is not an official "
                    "ISO 3166-1 assignment."
                ),
            }
        )

    write_taxonomy(
        args.taxonomy_path,
        sorted(
            taxonomy,
            key=lambda row: row["class_code"],
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
                "source_dataset",
                "source_license",
                "source_url",
            ],
        )
        writer.writeheader()
        writer.writerows(metadata)

    print(f"Worldwide classes : {len(taxonomy)}")
    print(
        "ISO classes       : "
        f"{sum(row['taxonomy'] == 'ISO 3166-1' for row in taxonomy)}"
    )
    print("Extra classes     : 1 (Kosovo / XK)")
    print(
        f"Canonical variants: {len(metadata)}"
    )
    print(
        f"Taxonomy saved    : {args.taxonomy_path}"
    )
    print(
        f"Metadata saved    : {args.metadata_path}"
    )


if __name__ == "__main__":
    main()
