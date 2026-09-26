"""Image transforms shared by training, evaluation and deployment."""

from __future__ import annotations

import random

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
