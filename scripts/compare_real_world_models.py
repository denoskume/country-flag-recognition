"""Compare V1 and scene-aware V2 on the same approved real-world challenge set."""

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
        "--manifest",
        type=Path,
        default=Path("data/external_benchmark/real_world_manifest.csv"),
    )
    parser.add_argument(
        "--v1",
        type=Path,
        default=Path("artifacts/models/worldwide_mobilenet_v3_small.pt"),
    )
    parser.add_argument(
        "--v2",
        type=Path,
        default=Path("artifacts/models/worldwide_scene_aware_mobilenet_v3_small.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/real_world_v1_vs_v2.json"),
    )
    return parser.parse_args()


@torch.inference_mode()
def probabilities(path: Path, bundle, transform) -> np.ndarray:
    with Image.open(path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(bundle.device)
    return torch.softmax(bundle.model(tensor), dim=1)[0].cpu().numpy()


def evaluate_checkpoint(
    checkpoint: Path,
    approved_rows: list[dict[str, str]],
) -> dict:
    bundle = load_inference_bundle(checkpoint)
    transform = build_eval_transform(bundle.image_size)
    class_to_index = {
        code: index for index, code in bundle.index_to_class.items()
    }

    probs = []
    targets = []
    details = []

    for row in approved_rows:
        code = row["class_code"]
        if code not in class_to_index:
            continue

        path = Path(row["path"])
        p = probabilities(path, bundle, transform)
        pred = int(p.argmax())
        conf = float(p[pred])
        target = class_to_index[code]

        probs.append(p)
        targets.append(target)
        details.append({
            "path": path.as_posix(),
            "target": code,
            "prediction": bundle.index_to_class[pred],
            "confidence": conf,
            "correct_top1": pred == target,
        })

    matrix = np.stack(probs)
    target_array = np.asarray(targets, dtype=int)
    summary = classification_summary(matrix, target_array)

    return {
        "checkpoint": str(checkpoint),
        "images": len(details),
        "classes_tested": len(set(item["target"] for item in details)),
        "closed_set": summary,
        "correct_top1": int(sum(item["correct_top1"] for item in details)),
        "top1_errors": [
            item for item in details
            if not item["correct_top1"]
        ],
        "details": details,
    }


def main():
    args = parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        approved = [
            row for row in csv.DictReader(handle)
            if row["review_status"].strip().lower() == "approved"
        ]

    if not approved:
        raise RuntimeError("No approved real-world challenge images found.")

    v1 = evaluate_checkpoint(args.v1, approved)
    v2 = evaluate_checkpoint(args.v2, approved)

    v1_by_path = {item["path"]: item for item in v1["details"]}
    v2_by_path = {item["path"]: item for item in v2["details"]}

    fixed = []
    regressed = []
    unchanged_correct = []
    unchanged_wrong = []

    for path in sorted(v1_by_path):
        a = v1_by_path[path]
        b = v2_by_path[path]

        record = {
            "path": path,
            "target": a["target"],
            "v1_prediction": a["prediction"],
            "v1_confidence": a["confidence"],
            "v2_prediction": b["prediction"],
            "v2_confidence": b["confidence"],
        }

        if (not a["correct_top1"]) and b["correct_top1"]:
            fixed.append(record)
        elif a["correct_top1"] and (not b["correct_top1"]):
            regressed.append(record)
        elif a["correct_top1"] and b["correct_top1"]:
            unchanged_correct.append(record)
        else:
            unchanged_wrong.append(record)

    report = {
        "approved_images": len(approved),
        "v1": {
            key: value
            for key, value in v1.items()
            if key != "details"
        },
        "v2": {
            key: value
            for key, value in v2.items()
            if key != "details"
        },
        "comparison": {
            "top1_accuracy_delta": (
                v2["closed_set"]["top1_accuracy"]
                - v1["closed_set"]["top1_accuracy"]
            ),
            "top5_accuracy_delta": (
                v2["closed_set"]["top5_accuracy"]
                - v1["closed_set"]["top5_accuracy"]
            ),
            "macro_f1_delta": (
                v2["closed_set"]["macro_f1"]
                - v1["closed_set"]["macro_f1"]
            ),
            "fixed_by_v2": fixed,
            "regressed_in_v2": regressed,
            "unchanged_correct": len(unchanged_correct),
            "unchanged_wrong": len(unchanged_wrong),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
