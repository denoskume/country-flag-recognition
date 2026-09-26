"""Evaluate closed-set classification and open-set detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
import yaml

from flag_recognition.dataset import FlagManifestDataset
from flag_recognition.inference import load_inference_bundle
from flag_recognition.metrics import (
    classification_summary,
    open_set_summary,
)
from flag_recognition.transforms import build_eval_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
    )
    return parser.parse_args()


@torch.inference_mode()
def predict_loader(
    model,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    model.eval()

    for batch in loader:
        images = batch["image"].to(
            device
        )
        logits = model(
            images
        )

        probabilities.append(
            torch.softmax(
                logits,
                dim=1,
            )
            .cpu()
            .numpy()
        )
        targets.append(
            batch["target"].numpy()
        )

    return (
        np.concatenate(
            probabilities
        ),
        np.concatenate(
            targets
        ),
    )


@torch.inference_mode()
def predict_unknown_scores(
    model,
    image_paths: list[Path],
    transform,
    device: torch.device,
) -> np.ndarray:
    scores: list[float] = []

    model.eval()

    for path in image_paths:
        with Image.open(
            path
        ) as image:
            tensor = transform(
                image.convert("RGB")
            ).unsqueeze(0).to(
                device
            )

        probabilities = torch.softmax(
            model(
                tensor
            ),
            dim=1,
        )[0]

        scores.append(
            float(
                probabilities.max().item()
            )
        )

    return np.asarray(
        scores,
        dtype=float,
    )


def evaluate_open_set_subset(
    known_scores: np.ndarray,
    subset: pd.DataFrame,
    model,
    transform,
    device: torch.device,
    threshold: float,
) -> dict[str, float | int] | None:
    """Evaluate one explicitly labeled unseen-data subset."""
    if subset.empty:
        return None

    unknown_scores = (
        predict_unknown_scores(
            model,
            [
                Path(path)
                for path in subset[
                    "path"
                ].tolist()
            ],
            transform,
            device,
        )
    )

    return {
        "images": int(
            len(subset)
        ),
        **open_set_summary(
            known_scores=(
                known_scores
            ),
            unknown_scores=(
                unknown_scores
            ),
        ),
        "unknown_rejection_rate": float(
            np.mean(
                unknown_scores
                < threshold
            )
        ),
    }


def main() -> None:
    args = parse_args()

    with args.config.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(
            handle
        )

    manifest_path = Path(
        config["data"][
            "split_manifest"
        ]
    )
    manifest = pd.read_csv(
        manifest_path
    )

    checkpoint_path = Path(
        config["artifacts"][
            "model_path"
        ]
    )
    bundle = (
        load_inference_bundle(
            checkpoint_path
        )
    )

    class_to_index = {
        country: index
        for index, country
        in bundle.index_to_class.items()
    }

    seen_test = manifest[
        (
            manifest["regime"]
            == "seen"
        )
        & (
            manifest["partition"]
            == "test"
        )
    ].copy()

    unseen_test = manifest[
        (
            manifest["regime"]
            == "unseen"
        )
        & (
            manifest["partition"]
            == "test"
        )
    ].copy()

    if seen_test.empty:
        raise RuntimeError(
            "The manifest contains no seen test images."
        )

    if unseen_test.empty:
        raise RuntimeError(
            "The manifest contains no unseen test images."
        )

    transform = (
        build_eval_transform(
            bundle.image_size
        )
    )

    seen_dataset = (
        FlagManifestDataset(
            seen_test,
            class_to_index=(
                class_to_index
            ),
            transform=transform,
        )
    )
    seen_loader = DataLoader(
        seen_dataset,
        batch_size=int(
            config["training"][
                "batch_size"
            ]
        ),
        shuffle=False,
        num_workers=0,
    )

    (
        seen_probabilities,
        seen_targets,
    ) = predict_loader(
        bundle.model,
        seen_loader,
        bundle.device,
    )

    closed_set_metrics = (
        classification_summary(
            seen_probabilities,
            seen_targets,
        )
    )

    known_scores = (
        seen_probabilities.max(
            axis=1
        )
    )

    threshold = (
        bundle.unknown_threshold
    )

    known_acceptance = float(
        np.mean(
            known_scores
            >= threshold
        )
    )

    # Keep real-world unknown evidence separate from controlled
    # canonical/presentation variants.
    unseen_real_world = unseen_test[
        unseen_test[
            "evaluation_scope"
        ]
        == "real_world_unknown"
    ].copy()

    unseen_controlled = unseen_test[
        unseen_test[
            "evaluation_scope"
        ]
        == "controlled_unknown"
    ].copy()

    overall_open_set = (
        evaluate_open_set_subset(
            known_scores=(
                known_scores
            ),
            subset=unseen_test,
            model=bundle.model,
            transform=transform,
            device=bundle.device,
            threshold=threshold,
        )
    )

    real_world_open_set = (
        evaluate_open_set_subset(
            known_scores=(
                known_scores
            ),
            subset=(
                unseen_real_world
            ),
            model=bundle.model,
            transform=transform,
            device=bundle.device,
            threshold=threshold,
        )
    )

    controlled_open_set = (
        evaluate_open_set_subset(
            known_scores=(
                known_scores
            ),
            subset=(
                unseen_controlled
            ),
            model=bundle.model,
            transform=transform,
            device=bundle.device,
            threshold=threshold,
        )
    )

    results = {
        "checkpoint": str(
            checkpoint_path
        ),
        "seen_test_images": int(
            len(seen_test)
        ),
        "unseen_test_images": int(
            len(unseen_test)
        ),
        "unknown_threshold": float(
            threshold
        ),
        "closed_set_real_world": {
            **closed_set_metrics,
            "known_acceptance_rate": (
                known_acceptance
            ),
        },
        "open_set_overall": (
            overall_open_set
        ),
        "open_set_real_world": (
            real_world_open_set
        ),
        "open_set_controlled_presentation": (
            controlled_open_set
        ),
    }

    metrics_dir = Path(
        config["artifacts"][
            "metrics_dir"
        ]
    )
    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        metrics_dir
        / "baseline_evaluation.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            results,
            handle,
            indent=2,
        )

    print(
        json.dumps(
            results,
            indent=2,
        )
    )
    print(
        f"\nSaved: {output_path}"
    )


if __name__ == "__main__":
    main()
