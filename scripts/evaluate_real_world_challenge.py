"""Evaluate only human-approved real-world challenge-set images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from flag_recognition.inference import load_inference_bundle
from flag_recognition.metrics import classification_summary
from flag_recognition.transforms import build_eval_transform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/models/worldwide_mobilenet_v3_small.pt"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/external_benchmark/real_world_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/real_world_challenge_evaluation.json"),
    )
    parser.add_argument(
        "--deployment-threshold",
        type=float,
        default=0.8804835677146912,
    )
    return parser.parse_args()


@torch.inference_mode()
def probabilities(path: Path, bundle, transform) -> np.ndarray:
    with Image.open(path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(bundle.device)
    return torch.softmax(bundle.model(tensor), dim=1)[0].cpu().numpy()


def main():
    args = parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["review_status"].strip().lower() == "approved"
        ]

    if not rows:
        raise RuntimeError(
            "No approved images. Review the manifest and set review_status=approved "
            "only for correctly labelled real-world flag photos."
        )

    bundle = load_inference_bundle(args.checkpoint)
    transform = build_eval_transform(bundle.image_size)
    class_to_index = {
        code: index for index, code in bundle.index_to_class.items()
    }

    probs = []
    targets = []
    details = []

    for row in rows:
        code = row["class_code"]
        if code not in class_to_index:
            continue

        path = Path(row["path"])
        prediction_probs = probabilities(path, bundle, transform)
        prediction = int(prediction_probs.argmax())
        confidence = float(prediction_probs[prediction])
        target = class_to_index[code]

        probs.append(prediction_probs)
        targets.append(target)
        details.append({
            "path": path.as_posix(),
            "target": code,
            "prediction": bundle.index_to_class[prediction],
            "confidence": confidence,
            "accepted": confidence >= args.deployment_threshold,
            "correct_top1": prediction == target,
        })

    matrix = np.stack(probs)
    target_array = np.asarray(targets, dtype=int)
    summary = classification_summary(matrix, target_array)
    scores = matrix.max(axis=1)

    report = {
        "approved_images": len(details),
        "classes_tested": len(set(item["target"] for item in details)),
        "deployment_threshold": args.deployment_threshold,
        "closed_set": summary,
        "known_acceptance_rate": float(
            np.mean(scores >= args.deployment_threshold)
        ),
        "known_rejection_rate": float(
            np.mean(scores < args.deployment_threshold)
        ),
        "top1_errors": [
            item for item in details if not item["correct_top1"]
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
