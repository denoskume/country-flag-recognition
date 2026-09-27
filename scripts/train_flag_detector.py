"""Train a generic flag detector using torchvision Faster R-CNN."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class DetectionDataset(Dataset):
    def __init__(self, manifest: Path):
        rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
        grouped = {}
        for row in rows:
            grouped.setdefault(row["image_path"], []).append(row)

        self.items = sorted(grouped.items())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path_str, rows = self.items[index]
        path = Path(path_str)

        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = F.to_tensor(image)

        boxes = torch.tensor(
            [
                [
                    float(row["xmin"]),
                    float(row["ymin"]),
                    float(row["xmax"]),
                    float(row["ymax"]),
                ]
                for row in rows
            ],
            dtype=torch.float32,
        )
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
        }
        return tensor, target


def collate(batch):
    return tuple(zip(*batch))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/flag_detection/manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/models/flag_detector_fasterrcnn.pt"),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = DetectionDataset(args.manifest)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes=2,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    print(f"Device           : {device}")
    print(f"Detection images : {len(dataset)}")

    model.train()

    for epoch in range(1, args.epochs + 1):
        losses = []

        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [
                {
                    key: value.to(device)
                    for key, value in target.items()
                }
                for target in targets
            ]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

        mean_loss = sum(losses) / max(len(losses), 1)
        print(f"Epoch {epoch:02d} | loss={mean_loss:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "num_classes": 2,
            "seed": args.seed,
        },
        args.output,
    )

    print(f"Saved detector : {args.output}")


if __name__ == "__main__":
    main()
