"""Train the first supervised seen-country baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
import yaml

from flag_recognition.dataset import FlagManifestDataset
from flag_recognition.model import build_classifier
from flag_recognition.transforms import (
    build_eval_transform,
    build_train_transform,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()

    losses: list[float] = []
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["target"].to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        losses.append(float(loss.item()))
        probabilities.append(
            torch.softmax(logits, dim=1)
            .cpu()
            .numpy()
        )
        targets.append(
            labels.cpu().numpy()
        )

    return (
        float(np.mean(losses)),
        np.concatenate(probabilities),
        np.concatenate(targets),
    )


def main() -> None:
    args = parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed = int(config["project"]["seed"])
    image_size = int(
        config["project"]["image_size"]
    )
    set_seed(seed)

    manifest_path = Path(
        config["data"]["split_manifest"]
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_path}. "
            "Run scripts/prepare_splits.py first."
        )

    manifest = pd.read_csv(
        manifest_path
    )

    seen_manifest = manifest[
        manifest["regime"] == "seen"
    ].copy()

    train_manifest = seen_manifest[
        seen_manifest["partition"] == "train"
    ].copy()
    validation_manifest = seen_manifest[
        seen_manifest["partition"] == "validation"
    ].copy()

    seen_classes = sorted(
        seen_manifest["country"].unique()
    )

    if len(seen_classes) < 2:
        raise RuntimeError(
            "At least two seen classes are required for training."
        )

    class_to_index = {
        country: index
        for index, country in enumerate(seen_classes)
    }

    train_dataset = FlagManifestDataset(
        train_manifest,
        class_to_index=class_to_index,
        transform=build_train_transform(
            image_size
        ),
    )
    validation_dataset = FlagManifestDataset(
        validation_manifest,
        class_to_index=class_to_index,
        transform=build_eval_transform(
            image_size
        ),
    )

    training_cfg = config["training"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(
            training_cfg["batch_size"]
        ),
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(
            training_cfg["batch_size"]
        ),
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model_cfg = config["model"]

    model = build_classifier(
        num_classes=len(class_to_index),
        pretrained=bool(
            model_cfg["pretrained"]
        ),
        dropout=float(
            model_cfg["dropout"]
        ),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            training_cfg["learning_rate"]
        ),
        weight_decay=float(
            training_cfg["weight_decay"]
        ),
    )

    epochs = int(training_cfg["epochs"])
    patience = int(
        training_cfg["early_stopping_patience"]
    )

    best_f1 = -1.0
    epochs_without_improvement = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_probabilities: np.ndarray | None = None

    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["target"].to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)
            loss = criterion(
                logits,
                labels,
            )
            loss.backward()
            optimizer.step()

            train_losses.append(
                float(loss.item())
            )

        (
            validation_loss,
            validation_probabilities,
            validation_targets,
        ) = evaluate(
            model,
            validation_loader,
            device,
        )

        validation_predictions = (
            validation_probabilities.argmax(
                axis=1
            )
        )
        validation_f1 = float(
            f1_score(
                validation_targets,
                validation_predictions,
                average="macro",
                zero_division=0,
            )
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(
                np.mean(train_losses)
            ),
            "validation_loss": (
                validation_loss
            ),
            "validation_macro_f1": (
                validation_f1
            ),
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={epoch_record['train_loss']:.4f} | "
            f"val_loss={validation_loss:.4f} | "
            f"val_macro_f1={validation_f1:.4f}"
        )

        if validation_f1 > best_f1:
            best_f1 = validation_f1
            epochs_without_improvement = 0

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }
            best_validation_probabilities = (
                validation_probabilities.copy()
            )
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):
            print(
                "Early stopping: "
                "validation macro-F1 did not improve."
            )
            break

    if (
        best_state is None
        or best_validation_probabilities
        is None
    ):
        raise RuntimeError(
            "Training completed without a valid checkpoint."
        )

    validation_confidences = (
        best_validation_probabilities.max(
            axis=1
        )
    )

    acceptance_quantile = float(
        config["open_set"][
            "known_acceptance_quantile"
        ]
    )
    unknown_threshold = float(
        np.quantile(
            validation_confidences,
            acceptance_quantile,
        )
    )

    model_path = Path(
        config["artifacts"]["model_path"]
    )
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": best_state,
            "class_to_index": class_to_index,
            "image_size": image_size,
            "dropout": float(
                model_cfg["dropout"]
            ),
            "unknown_threshold": (
                unknown_threshold
            ),
            "validation_macro_f1": best_f1,
            "seed": seed,
        },
        model_path,
    )

    metrics_dir = Path(
        config["artifacts"]["metrics_dir"]
    )
    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        metrics_dir
        / "training_history.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            history,
            handle,
            indent=2,
        )

    print(f"\nBest validation macro-F1 : {best_f1:.4f}")
    print(f"Unknown threshold        : {unknown_threshold:.4f}")
    print(f"Saved checkpoint         : {model_path}")


if __name__ == "__main__":
    main()
