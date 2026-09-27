"""Evaluate detector + V1 classifier on approved real-world challenge images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from flag_recognition.detection import (
    detect_and_classify,
    load_flag_detector,
)
from flag_recognition.inference import load_inference_bundle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/external_benchmark/real_world_manifest.csv"),
    )
    parser.add_argument(
        "--classifier",
        type=Path,
        default=Path("artifacts/models/worldwide_mobilenet_v3_small.pt"),
    )
    parser.add_argument(
        "--detector",
        type=Path,
        default=Path("artifacts/models/flag_detector_fasterrcnn.pt"),
    )
    parser.add_argument(
        "--detector-threshold",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/detect_then_classify_real_world.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    classifier = load_inference_bundle(args.classifier)
    detector = load_flag_detector(
        args.detector,
        classifier.device,
    )

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        approved = [
            row for row in csv.DictReader(handle)
            if row["review_status"].strip().lower() == "approved"
        ]

    if not approved:
        raise RuntimeError("No approved challenge images found.")

    details = []

    for row in approved:
        path = Path(row["path"])

        with Image.open(path) as image:
            result = detect_and_classify(
                image=image.convert("RGB"),
                classifier=classifier,
                detector=detector,
                detector_threshold=args.detector_threshold,
                top_k=5,
            )

        details.append({
            "path": path.as_posix(),
            "target": row["class_code"],
            "prediction": result.prediction.top1_country,
            "confidence": result.prediction.top1_confidence,
            "correct_top1": (
                result.prediction.top1_country == row["class_code"]
            ),
            "correct_top5": (
                row["class_code"]
                in [country for country, _ in result.prediction.top5]
            ),
            "detected": result.detected,
            "detector_score": result.detector_score,
            "box": result.box,
        })

    top1 = float(np.mean([item["correct_top1"] for item in details]))
    top5 = float(np.mean([item["correct_top5"] for item in details]))
    detection_rate = float(np.mean([item["detected"] for item in details]))

    report = {
        "approved_images": len(details),
        "detector_threshold": args.detector_threshold,
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "detection_rate": detection_rate,
        "correct_top1": int(sum(item["correct_top1"] for item in details)),
        "top1_errors": [
            item for item in details
            if not item["correct_top1"]
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
