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
