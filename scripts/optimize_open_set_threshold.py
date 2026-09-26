"""Optimize the open-set confidence threshold from external benchmark predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/metrics/deployment_external_predictions.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/open_set_threshold_optimization.json"),
    )
    return parser.parse_args()


def metrics_at_threshold(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    known_acceptance = float(np.mean(known_scores >= threshold))
    known_rejection = 1.0 - known_acceptance
    unknown_rejection = float(np.mean(unknown_scores < threshold))
    unknown_false_acceptance = 1.0 - unknown_rejection
    balanced_accuracy = 0.5 * (
        known_acceptance + unknown_rejection
    )

    tp = float(np.sum(known_scores >= threshold))
    fn = float(np.sum(known_scores < threshold))
    fp = float(np.sum(unknown_scores >= threshold))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "known_acceptance_rate": known_acceptance,
        "known_rejection_rate": known_rejection,
        "unknown_rejection_rate": unknown_rejection,
        "unknown_false_acceptance_rate": unknown_false_acceptance,
        "balanced_accuracy": balanced_accuracy,
        "known_vs_unknown_precision": precision,
        "known_vs_unknown_recall": recall,
        "known_vs_unknown_f1": f1,
    }


def best_under_far(
    rows: list[dict[str, float]],
    far_limit: float,
) -> dict[str, float] | None:
    eligible = [
        row
        for row in rows
        if row["unknown_false_acceptance_rate"] <= far_limit
    ]
    if not eligible:
        return None

    return max(
        eligible,
        key=lambda row: (
            row["known_acceptance_rate"],
            row["balanced_accuracy"],
            -row["threshold"],
        ),
    )


def main():
    args = parse_args()

    table = pd.read_csv(args.predictions)

    known_scores = table.loc[
        table["scope"] == "known",
        "confidence",
    ].to_numpy(dtype=float)

    unknown_scores = table.loc[
        table["scope"] == "unknown",
        "confidence",
    ].to_numpy(dtype=float)

    if not len(known_scores) or not len(unknown_scores):
        raise RuntimeError(
            "Predictions CSV must contain both known and unknown rows."
        )

    candidates = np.unique(
        np.concatenate(
            [
                np.asarray([0.0, 1.0]),
                known_scores,
                unknown_scores,
            ]
        )
    )

    rows = [
        metrics_at_threshold(
            known_scores=known_scores,
            unknown_scores=unknown_scores,
            threshold=float(threshold),
        )
        for threshold in candidates
    ]

    best_balanced = max(
        rows,
        key=lambda row: (
            row["balanced_accuracy"],
            row["known_vs_unknown_f1"],
        ),
    )

    best_f1 = max(
        rows,
        key=lambda row: (
            row["known_vs_unknown_f1"],
            row["balanced_accuracy"],
        ),
    )

    far_targets = {
        "far_10_percent": 0.10,
        "far_5_percent": 0.05,
        "far_2_percent": 0.02,
        "far_1_percent": 0.01,
    }

    constrained = {
        label: best_under_far(rows, limit)
        for label, limit in far_targets.items()
    }

    checkpoint_report = json.loads(
        Path("artifacts/metrics/deployment_external_evaluation.json")
        .read_text(encoding="utf-8")
    )
    checkpoint_threshold = float(
        checkpoint_report["threshold_audit"]["threshold"]
    )

    current = metrics_at_threshold(
        known_scores=known_scores,
        unknown_scores=unknown_scores,
        threshold=checkpoint_threshold,
    )

    report = {
        "predictions": str(args.predictions),
        "known_images": int(len(known_scores)),
        "unknown_images": int(len(unknown_scores)),
        "current_checkpoint_threshold": current,
        "best_balanced_accuracy": best_balanced,
        "best_known_vs_unknown_f1": best_f1,
        "operating_points": constrained,
        "interpretation": {
            "far": (
                "Unknown false acceptance rate: fraction of non-flag "
                "images incorrectly accepted as known flags."
            ),
            "known_acceptance": (
                "Fraction of known external flag images accepted by "
                "the confidence gate."
            ),
            "selection_rule": (
                "For each FAR target, choose the threshold that maximizes "
                "known acceptance while satisfying the FAR constraint."
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
