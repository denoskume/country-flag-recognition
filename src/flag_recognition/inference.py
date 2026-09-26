"""Checkpoint loading and single-image inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image

from .model import build_classifier
from .transforms import build_eval_transform


@dataclass(frozen=True)
class Prediction:
    """One image-level prediction result."""

    top1_country: str
    top1_confidence: float
    top5: tuple[tuple[str, float], ...]
    is_known: bool
    unknown_threshold: float
    inference_ms: float


@dataclass
class InferenceBundle:
    """Model and metadata required for deployment inference."""

    model: torch.nn.Module
    index_to_class: dict[int, str]
    image_size: int
    unknown_threshold: float
    device: torch.device


def load_inference_bundle(
    checkpoint_path: Path,
    device: str | None = None,
) -> InferenceBundle:
    """Load a supervised checkpoint and its label metadata."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    resolved_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=resolved_device,
    )

    class_to_index = {
        str(country): int(index)
        for country, index in checkpoint["class_to_index"].items()
    }
    index_to_class = {
        index: country
        for country, index in class_to_index.items()
    }

    image_size = int(
        checkpoint.get("image_size", 224)
    )
    unknown_threshold = float(
        checkpoint.get("unknown_threshold", 0.0)
    )

    model = build_classifier(
        num_classes=len(class_to_index),
        pretrained=False,
        dropout=float(
            checkpoint.get("dropout", 0.2)
        ),
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.to(resolved_device)
    model.eval()

    return InferenceBundle(
        model=model,
        index_to_class=index_to_class,
        image_size=image_size,
        unknown_threshold=unknown_threshold,
        device=resolved_device,
    )


@torch.inference_mode()
def predict_image(
    image: Image.Image,
    bundle: InferenceBundle,
    top_k: int = 5,
) -> Prediction:
    """Predict a country and apply the learned open-set threshold."""
    transform = build_eval_transform(
        bundle.image_size
    )

    tensor = transform(
        image.convert("RGB")
    ).unsqueeze(0).to(bundle.device)

    start = perf_counter()

    logits = bundle.model(tensor)
    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]

    if bundle.device.type == "cuda":
        torch.cuda.synchronize()

    inference_ms = (
        perf_counter() - start
    ) * 1000.0

    k = min(
        int(top_k),
        probabilities.numel(),
    )

    values, indices = torch.topk(
        probabilities,
        k=k,
    )

    top5 = tuple(
        (
            bundle.index_to_class[int(index)],
            float(value),
        )
        for value, index in zip(
            values.cpu(),
            indices.cpu(),
        )
    )

    top1_country, top1_confidence = top5[0]

    return Prediction(
        top1_country=top1_country,
        top1_confidence=top1_confidence,
        top5=top5,
        is_known=(
            top1_confidence
            >= bundle.unknown_threshold
        ),
        unknown_threshold=bundle.unknown_threshold,
        inference_ms=inference_ms,
    )



def _scene_crops(image: Image.Image) -> list[Image.Image]:
    """Generate overlapping scene regions while preserving the full image.

    The deployment classifier was trained primarily on flag crops. Full-scene
    photos can contain a small flag, so deployment inference also examines
    coarse overlapping regions without requiring a separate detector.
    """
    image = image.convert("RGB")
    width, height = image.size

    crops: list[Image.Image] = [image]

    # Two-by-two overlapping windows at 65% of each dimension.
    crop_w = max(1, round(width * 0.65))
    crop_h = max(1, round(height * 0.65))
    x_positions = sorted({0, max(0, width - crop_w), max(0, (width - crop_w) // 2)})
    y_positions = sorted({0, max(0, height - crop_h), max(0, (height - crop_h) // 2)})

    for y in y_positions:
        for x in x_positions:
            region = image.crop((x, y, x + crop_w, y + crop_h))
            if region.size != image.size:
                crops.append(region)

    # Wide horizontal bands help with flags mounted high in a scene.
    band_h = max(1, round(height * 0.50))
    for y in sorted({0, max(0, height - band_h)}):
        crops.append(image.crop((0, y, width, y + band_h)))

    return crops


@torch.inference_mode()
def predict_scene(
    image: Image.Image,
    bundle: InferenceBundle,
    top_k: int = 5,
) -> Prediction:
    """Predict from a full scene using multi-region score aggregation."""
    transform = build_eval_transform(bundle.image_size)
    crops = _scene_crops(image)

    batch = torch.stack(
        [transform(crop) for crop in crops],
        dim=0,
    ).to(bundle.device)

    start = perf_counter()
    logits = bundle.model(batch)
    probabilities = torch.softmax(logits, dim=1)

    if bundle.device.type == "cuda":
        torch.cuda.synchronize()

    inference_ms = (perf_counter() - start) * 1000.0

    # Max pooling over regions lets a confident local flag region dominate.
    aggregated = probabilities.max(dim=0).values

    k = min(int(top_k), aggregated.numel())
    values, indices = torch.topk(aggregated, k=k)

    top5 = tuple(
        (
            bundle.index_to_class[int(index)],
            float(value),
        )
        for value, index in zip(
            values.cpu(),
            indices.cpu(),
        )
    )

    top1_country, top1_confidence = top5[0]

    return Prediction(
        top1_country=top1_country,
        top1_confidence=top1_confidence,
        top5=top5,
        is_known=(
            top1_confidence >= bundle.unknown_threshold
        ),
        unknown_threshold=bundle.unknown_threshold,
        inference_ms=inference_ms,
    )
