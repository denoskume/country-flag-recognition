"""Train the final worldwide deployment classifier on all flag classes."""

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

from flag_recognition.dataset import (
    FlagManifestDataset,
    discover_country_images,
)
from flag_recognition.model import build_classifier
from flag_recognition.splits import is_canonical_variant
from flag_recognition.transforms import (
    build_eval_transform,
    build_generalization_train_transform,
    build_train_transform,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/deployment.yaml"),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_deployment_manifest(
    class_to_images: dict[str, list[Path]],
    seed: int,
    scene_to_images: dict[str, list[Path]] | None = None,
) -> pd.DataFrame:
    """Use every class while keeping one validation source per class."""
    rows: list[dict[str, str]] = []

    for class_index, country in enumerate(sorted(class_to_images)):
        paths = sorted(
            Path(path)
            for path in class_to_images[country]
        )

        canonical = [
            path
            for path in paths
            if is_canonical_variant(path)
        ]
        real_world = [
            path
            for path in paths
            if not is_canonical_variant(path)
        ]

        rng = random.Random(
            seed + class_index
        )

        if real_world:
            validation_path = rng.choice(
                real_world
            )
        elif canonical:
            validation_path = canonical[0]
        else:
            raise RuntimeError(
                f"No usable images for class '{country}'."
            )

        for path in paths:
            partition = (
                "validation"
                if path == validation_path
                else "train"
            )

            rows.append(
                {
                    "path": path.as_posix(),
                    "country": country,
                    "regime": "deployment",
                    "partition": partition,
                }
            )

    if scene_to_images:
        for country, paths in sorted(scene_to_images.items()):
            if country not in class_to_images:
                continue

            for path in sorted(Path(path) for path in paths):
                rows.append(
                    {
                        "path": path.as_posix(),
                        "country": country,
                        "regime": "scene_aware_synthetic",
                        "partition": "train",
                    }
                )

    manifest = pd.DataFrame(rows)

    if manifest.empty:
        raise RuntimeError(
            "Deployment manifest is empty."
        )

    return manifest.sort_values(
        ["country", "partition", "path"]
    ).reset_index(drop=True)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()

    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(
            training_cfg.get(
                "label_smoothing",
                0.0,
            )
        )
    )

    losses: list[float] = []
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["target"].to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        losses.append(float(loss.item()))
        probabilities.append(
            torch.softmax(
                logits,
                dim=1,
            ).cpu().numpy()
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

    with args.config.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    seed = int(
        config["project"]["seed"]
    )
    image_size = int(
        config["project"]["image_size"]
    )
    set_seed(seed)

    class_to_images = discover_country_images(
        raw_dir=Path(
            config["data"]["raw_dir"]
        ),
        allowed_extensions=set(
            config["data"]["allowed_extensions"]
        ),
    )

    scene_to_images = None
    scene_train_dir = config["data"].get("scene_train_dir")

    if scene_train_dir:
        scene_path = Path(scene_train_dir)
        if scene_path.exists():
            scene_to_images = discover_country_images(
                raw_dir=scene_path,
                allowed_extensions=set(
                    config["data"]["allowed_extensions"]
                ),
            )

    manifest = build_deployment_manifest(
        class_to_images,
        seed=seed,
        scene_to_images=scene_to_images,
    )

    train_manifest = manifest[
        manifest["partition"] == "train"
    ].copy()
    validation_manifest = manifest[
        manifest["partition"] == "validation"
    ].copy()

    classes = sorted(
        manifest["country"].unique()
    )
    class_to_index = {
        country: index
        for index, country in enumerate(classes)
    }

    augmentation_profile = str(
        config.get("training", {}).get(
            "augmentation_profile",
            "standard",
        )
    ).lower()

    if augmentation_profile == "generalization":
        train_transform = build_generalization_train_transform(
            image_size
        )
    else:
        train_transform = build_train_transform(
            image_size
        )

    train_dataset = FlagManifestDataset(
        train_manifest,
        class_to_index=class_to_index,
        transform=train_transform,
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

    print(
        f"Device              : {device}"
    )
    print(
        f"Worldwide classes   : {len(classes)}"
    )
    print(
        f"Training images     : {len(train_manifest)}"
    )
    print(
        f"Validation images   : {len(validation_manifest)}"
    )
    scene_train_count = int(
        np.sum(train_manifest["regime"] == "scene_aware_synthetic")
    )
    print(
        f"Scene-aware images  : {scene_train_count}"
    )

    model_cfg = config["model"]

    model = build_classifier(
        num_classes=len(classes),
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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(
            1,
            int(training_cfg["epochs"]),
        ),
        eta_min=float(
            training_cfg.get(
                "minimum_learning_rate",
                1e-6,
            )
        ),
    )

    epochs = int(
        training_cfg["epochs"]
    )
    patience = int(
        training_cfg["early_stopping_patience"]
    )

    best_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_probabilities: np.ndarray | None = None
    epochs_without_improvement = 0

    history: list[
        dict[str, float | int]
    ] = []

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()
        training_losses: list[float] = []

        for batch in train_loader:
            images = batch["image"].to(
                device
            )
            labels = batch["target"].to(
                device
            )

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

            training_losses.append(
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

        predictions = (
            validation_probabilities.argmax(
                axis=1
            )
        )

        validation_f1 = float(
            f1_score(
                validation_targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        )

        record = {
            "epoch": epoch,
            "train_loss": float(
                np.mean(
                    training_losses
                )
            ),
            "validation_loss": (
                validation_loss
            ),
            "validation_macro_f1": (
                validation_f1
            ),
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={record['train_loss']:.4f} | "
            f"val_loss={validation_loss:.4f} | "
            f"val_macro_f1={validation_f1:.4f}"
        )

        scheduler.step()

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
                "Early stopping: validation macro-F1 did not improve."
            )
            break

    if (
        best_state is None
        or best_validation_probabilities is None
    ):
        raise RuntimeError(
            "Training completed without a valid checkpoint."
        )

    confidence_quantile = float(
        config["open_set"][
            "known_acceptance_quantile"
        ]
    )

    unknown_threshold = float(
        np.quantile(
            best_validation_probabilities.max(
                axis=1
            ),
            confidence_quantile,
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
            "augmentation_profile": augmentation_profile,
            "training_scope": (
                "worldwide_250_class_scene_aware"
                if scene_to_images
                else "worldwide_250_class_deployment"
            ),
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

    manifest.to_csv(
        metrics_dir
        / "deployment_manifest.csv",
        index=False,
    )

    with (
        metrics_dir
        / "deployment_training_history.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            history,
            handle,
            indent=2,
        )

    summary = {
        "classes": len(classes),
        "training_images": int(
            len(train_manifest)
        ),
        "validation_images": int(
            len(validation_manifest)
        ),
        "scene_training_images": int(
            np.sum(train_manifest["regime"] == "scene_aware_synthetic")
        ),
        "best_validation_macro_f1": best_f1,
        "unknown_threshold": (
            unknown_threshold
        ),
        "checkpoint": str(
            model_path
        ),
    }

    with (
        metrics_dir
        / "deployment_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print()
    print(
        f"Best validation macro-F1 : {best_f1:.4f}"
    )
    print(
        f"Unknown threshold        : {unknown_threshold:.4f}"
    )
    print(
        f"Saved checkpoint         : {model_path}"
    )


if __name__ == "__main__":
    main()
