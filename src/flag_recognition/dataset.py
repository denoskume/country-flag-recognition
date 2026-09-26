"""Dataset discovery and manifest-backed dataset access."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image


DEFAULT_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def discover_country_images(
    raw_dir: Path,
    allowed_extensions: set[str] | None = None,
) -> dict[str, list[Path]]:
    """Discover one country class per direct child directory."""
    raw_dir = Path(raw_dir)
    extensions = {
        extension.lower()
        for extension in (allowed_extensions or DEFAULT_EXTENSIONS)
    }

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    class_to_images: dict[str, list[Path]] = {}

    for country_dir in sorted(
        path for path in raw_dir.iterdir() if path.is_dir()
    ):
        images = sorted(
            path
            for path in country_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in extensions
        )

        if images:
            class_to_images[country_dir.name] = images

    if not class_to_images:
        raise FileNotFoundError(
            f"No country image classes were discovered under {raw_dir}."
        )

    return class_to_images


class FlagManifestDataset:
    """Map-style image dataset compatible with PyTorch DataLoader."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        class_to_index: dict[str, int],
        transform=None,
    ) -> None:
        required_columns = {
            "path",
            "country",
            "regime",
            "partition",
        }

        missing_columns = required_columns - set(manifest.columns)

        if missing_columns:
            raise ValueError(
                f"Manifest is missing columns: {sorted(missing_columns)}"
            )

        self.manifest = manifest.reset_index(drop=True).copy()
        self.class_to_index = dict(class_to_index)
        self.transform = transform

        unknown_labels = (
            set(self.manifest["country"])
            - set(self.class_to_index)
        )

        if unknown_labels:
            raise ValueError(
                "Manifest contains classes absent from class_to_index: "
                f"{sorted(unknown_labels)}"
            )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]
        path = Path(row["path"])

        with Image.open(path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        target = self.class_to_index[row["country"]]

        return {
            "image": image,
            "target": target,
            "country": row["country"],
            "path": str(path),
        }


def load_manifest(
    manifest_path: Path,
    regime: str | None = None,
    partition: str | None = None,
) -> pd.DataFrame:
    """Load and optionally filter the split manifest."""
    manifest_path = Path(manifest_path)

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_path}"
        )

    manifest = pd.read_csv(manifest_path)

    if regime is not None:
        manifest = manifest[
            manifest["regime"] == regime
        ]

    if partition is not None:
        manifest = manifest[
            manifest["partition"] == partition
        ]

    return manifest.reset_index(drop=True)
