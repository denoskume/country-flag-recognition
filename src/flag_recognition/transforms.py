"""Image transforms shared by training, evaluation and deployment."""

from __future__ import annotations

import random
from io import BytesIO

import numpy as np

from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms


class FitToSquare:
    """Preserve the full flag while fitting it onto a square canvas."""

    def __init__(
        self,
        size: int,
        fill: tuple[int, int, int] = (245, 245, 245),
    ) -> None:
        self.size = int(size)
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.pad(
            image.convert("RGB"),
            (self.size, self.size),
            method=Image.Resampling.LANCZOS,
            color=self.fill,
            centering=(0.5, 0.5),
        )


class RandomPresentationGeometry:
    """Simulate common ways the same flag can be framed or displayed.

    The transform changes presentation, not class identity:
    landscape, portrait, square, circular crop, rounded crop, and scale.
    Horizontal/vertical flips are intentionally never used because they can
    change the semantics of asymmetric flags.
    """

    def __init__(
        self,
        probability: float = 0.65,
        shape_probability: float = 0.30,
    ) -> None:
        self.probability = float(probability)
        self.shape_probability = float(shape_probability)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")

        if random.random() > self.probability:
            return image

        layouts = {
            "landscape": (360, 240),
            "portrait": (240, 360),
            "square": (320, 320),
        }
        _, canvas_size = random.choice(list(layouts.items()))
        canvas_width, canvas_height = canvas_size

        background = random.choice(
            [
                (245, 245, 245),
                (224, 224, 224),
                (255, 255, 255),
                (35, 35, 35),
            ]
        )

        max_width = int(canvas_width * random.uniform(0.68, 0.94))
        max_height = int(canvas_height * random.uniform(0.68, 0.94))

        fitted = ImageOps.contain(
            image,
            (max_width, max_height),
            method=Image.Resampling.LANCZOS,
        )

        if random.random() < self.shape_probability:
            mask = Image.new("L", fitted.size, 0)
            draw = ImageDraw.Draw(mask)

            if random.random() < 0.5:
                draw.ellipse(
                    (0, 0, fitted.width - 1, fitted.height - 1),
                    fill=255,
                )
            else:
                radius = max(
                    4,
                    round(min(fitted.size) * 0.12),
                )
                draw.rounded_rectangle(
                    (0, 0, fitted.width - 1, fitted.height - 1),
                    radius=radius,
                    fill=255,
                )

            shaped = Image.new("RGB", fitted.size, background)
            shaped.paste(fitted, (0, 0), mask)
            fitted = shaped

        canvas = Image.new(
            "RGB",
            (canvas_width, canvas_height),
            background,
        )

        free_x = canvas_width - fitted.width
        free_y = canvas_height - fitted.height

        x = (
            random.randint(0, free_x)
            if free_x > 0
            else 0
        )
        y = (
            random.randint(0, free_y)
            if free_y > 0
            else 0
        )

        canvas.paste(
            fitted,
            (x, y),
        )

        return canvas


class RandomRightAngleRotation:
    """Occasionally simulate an image captured in the wrong orientation."""

    def __init__(self, probability: float = 0.12) -> None:
        self.probability = float(probability)

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image

        return image.rotate(
            random.choice([-90, 90]),
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(245, 245, 245),
        )


class RandomJPEGCompression:
    """Simulate messaging/social-media recompression artifacts."""

    def __init__(
        self,
        probability: float = 0.45,
        quality_range: tuple[int, int] = (35, 92),
    ) -> None:
        self.probability = float(probability)
        self.quality_range = quality_range

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if random.random() > self.probability:
            return image

        buffer = BytesIO()
        quality = random.randint(*self.quality_range)
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        with Image.open(buffer) as compressed:
            compressed.load()
            return compressed.convert("RGB").copy()


class RandomSensorNoise:
    """Add mild camera-like RGB noise."""

    def __init__(
        self,
        probability: float = 0.30,
        sigma_range: tuple[float, float] = (2.0, 10.0),
    ) -> None:
        self.probability = float(probability)
        self.sigma_range = sigma_range

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if random.random() > self.probability:
            return image

        array = np.asarray(image).astype(np.float32)
        sigma = random.uniform(*self.sigma_range)
        noise = np.random.normal(0.0, sigma, size=array.shape)
        array = np.clip(array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(array, mode="RGB")


class RandomPartialOcclusion:
    """Simulate folds, poles, objects, and partially hidden flags."""

    def __init__(
        self,
        probability: float = 0.28,
        max_fraction: float = 0.16,
    ) -> None:
        self.probability = float(probability)
        self.max_fraction = float(max_fraction)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB").copy()
        if random.random() > self.probability:
            return image

        draw = ImageDraw.Draw(image)
        width, height = image.size
        occ_w = max(2, int(width * random.uniform(0.03, self.max_fraction)))
        occ_h = max(2, int(height * random.uniform(0.03, self.max_fraction)))
        x0 = random.randint(0, max(0, width - occ_w))
        y0 = random.randint(0, max(0, height - occ_h))
        fill = random.choice([
            (245, 245, 245),
            (40, 40, 40),
            (120, 120, 120),
        ])
        draw.rectangle(
            (x0, y0, x0 + occ_w, y0 + occ_h),
            fill=fill,
        )
        return image


def build_generalization_train_transform(image_size: int = 224):
    """Aggressive identity-preserving augmentation for real-world robustness."""
    return transforms.Compose(
        [
            RandomPresentationGeometry(
                probability=0.90,
                shape_probability=0.18,
            ),
            RandomRightAngleRotation(
                probability=0.05,
            ),
            transforms.RandomRotation(
                degrees=18,
                fill=(245, 245, 245),
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.12, 0.12),
                scale=(0.62, 1.28),
                shear=(-12, 12, -8, 8),
                fill=(245, 245, 245),
            ),
            transforms.RandomPerspective(
                distortion_scale=0.42,
                p=0.62,
                fill=(245, 245, 245),
            ),
            RandomPartialOcclusion(
                probability=0.28,
                max_fraction=0.16,
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.42,
                        contrast=0.42,
                        saturation=0.35,
                        hue=0.04,
                    )
                ],
                p=0.82,
            ),
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=3,
                        sigma=(0.3, 2.2),
                    )
                ],
                p=0.28,
            ),
            RandomJPEGCompression(
                probability=0.45,
                quality_range=(35, 92),
            ),
            RandomSensorNoise(
                probability=0.30,
                sigma_range=(2.0, 10.0),
            ),
            FitToSquare(image_size),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=0.22,
                scale=(0.015, 0.15),
                ratio=(0.3, 3.0),
                value="random",
            ),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def build_train_transform(image_size: int = 224):
    """Build identity-preserving augmentation for realistic flag presentation."""
    return transforms.Compose(
        [
            RandomPresentationGeometry(
                probability=0.65,
                shape_probability=0.30,
            ),
            RandomRightAngleRotation(
                probability=0.12,
            ),
            transforms.RandomRotation(
                degrees=25,
                fill=(245, 245, 245),
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
                scale=(0.78, 1.18),
                shear=(-8, 8, -5, 5),
                fill=(245, 245, 245),
            ),
            transforms.RandomPerspective(
                distortion_scale=0.30,
                p=0.40,
                fill=(245, 245, 245),
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.30,
                        contrast=0.30,
                        saturation=0.22,
                    )
                ],
                p=0.70,
            ),
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=3,
                        sigma=(0.4, 1.5),
                    )
                ],
                p=0.18,
            ),
            FitToSquare(
                image_size,
            ),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=0.15,
                scale=(0.02, 0.12),
                ratio=(0.4, 2.5),
                value="random",
            ),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def build_eval_transform(image_size: int = 224):
    """Preserve the entire input flag for validation, test and deployment."""
    return transforms.Compose(
        [
            FitToSquare(
                image_size,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
