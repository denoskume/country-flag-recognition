"""Detect a flag region first, then classify the crop with the V1 classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

from .inference import InferenceBundle, Prediction, predict_image


@dataclass(frozen=True)
class DetectionResult:
    prediction: Prediction
    detected: bool
    detector_score: float
    box: tuple[int, int, int, int] | None


def load_flag_detector(
    checkpoint_path: Path,
    device: torch.device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=2,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def detect_and_classify(
    image: Image.Image,
    classifier: InferenceBundle,
    detector,
    detector_threshold: float = 0.50,
    top_k: int = 5,
) -> DetectionResult:
    rgb = image.convert("RGB")
    tensor = torch.from_numpy(
        __import__("numpy").asarray(rgb).copy()
    ).permute(2, 0, 1).float() / 255.0
    tensor = tensor.to(classifier.device)

    output = detector([tensor])[0]
    scores = output["scores"].detach().cpu()
    boxes = output["boxes"].detach().cpu()

    if len(scores) == 0 or float(scores[0]) < detector_threshold:
        prediction = predict_image(rgb, classifier, top_k=top_k)
        return DetectionResult(
            prediction=prediction,
            detected=False,
            detector_score=(
                float(scores[0]) if len(scores) else 0.0
            ),
            box=None,
        )

    x1, y1, x2, y2 = boxes[0].tolist()
    width, height = rgb.size

    pad_x = 0.08 * (x2 - x1)
    pad_y = 0.08 * (y2 - y1)

    crop_box = (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(width, int(x2 + pad_x)),
        min(height, int(y2 + pad_y)),
    )

    crop = rgb.crop(crop_box)
    prediction = predict_image(crop, classifier, top_k=top_k)

    return DetectionResult(
        prediction=prediction,
        detected=True,
        detector_score=float(scores[0]),
        box=crop_box,
    )
