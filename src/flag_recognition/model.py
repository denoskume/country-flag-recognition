"""Model construction for supervised country-flag recognition."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)


def build_classifier(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Build a MobileNetV3-Small classifier for seen-country classes."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")

    weights = (
        MobileNet_V3_Small_Weights.DEFAULT
        if pretrained
        else None
    )

    model = mobilenet_v3_small(weights=weights)

    in_features = model.classifier[-1].in_features

    model.classifier[-2] = nn.Dropout(
        p=dropout,
        inplace=True,
    )
    model.classifier[-1] = nn.Linear(
        in_features,
        num_classes,
    )

    return model
