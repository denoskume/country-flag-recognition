"""Calibrate and validate an open-set threshold on disjoint benchmark splits."""

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
        default=Path("artifacts/metrics/open_set_calibration_validation.json"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--target-far",
        type=float,
        default=0.05,
        help="Maximum unknown false-acceptance rate used to select the threshold.",
    )
    return parser.parse_args()


def metrics(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    known_acceptance = float(np.mean(known_scores >= threshold))
    unknown_rejection = float(np.mean(unknown_scores < threshold))
    far = 1.0 - unknown_rejection

    return {
        "threshold": float(threshold),
        "known_acceptance_rate": known_acceptance,
        "known_rejection_rate": 1.0 - known_acceptance,
        "unknown_rejection_rate": unknown_rejection,
        "unknown_false_acceptance_rate": far,
        "balanced_accuracy": 0.5 * (
            known_acceptance + unknown_rejection
        ),
    }


def split_scores(
    values: np.ndarray,
    rng: np.random.Generator,
    calibration_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(len(values))
    cut = int(round(len(values) * calibration_fraction))
    cut = min(max(cut, 1), len(values) - 1)
    return values[order[:cut]], values[order[cut:]]


def select_threshold(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    far_limit: float,
) -> dict[str, float]:
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
        metrics(
            known_scores=known_scores,
            unknown_scores=unknown_scores,
            threshold=float(threshold),
        )
        for threshold in candidates
    ]

    eligible = [
        row
        for row in rows
        if row["unknown_false_acceptance_rate"] <= far_limit
    ]

    if not eligible:
        raise RuntimeError(
            "No calibration threshold satisfies the requested FAR limit."
        )

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

    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("--calibration-fraction must be between 0 and 1.")

    table = pd.read_csv(args.predictions)

    known_scores = table.loc[
        table["scope"] == "known",
        "confidence",
    ].to_numpy(dtype=float)
    unknown_scores = table.loc[
        table["scope"] == "unknown",
        "confidence",
    ].to_numpy(dtype=float)

    if len(known_scores) < 2 or len(unknown_scores) < 2:
        raise RuntimeError(
            "Need at least two known and two unknown examples."
        )

    rng = np.random.default_rng(args.seed)

    known_cal, known_test = split_scores(
        known_scores,
        rng,
        args.calibration_fraction,
    )
    unknown_cal, unknown_test = split_scores(
        unknown_scores,
        rng,
        args.calibration_fraction,
    )

    selected = select_threshold(
        known_scores=known_cal,
        unknown_scores=unknown_cal,
        far_limit=args.target_far,
    )

    threshold = selected["threshold"]

    report = {
        "predictions": str(args.predictions),
        "seed": args.seed,
        "target_far": args.target_far,
        "calibration_fraction": args.calibration_fraction,
        "sample_counts": {
            "known_calibration": int(len(known_cal)),
            "known_test": int(len(known_test)),
            "unknown_calibration": int(len(unknown_cal)),
            "unknown_test": int(len(unknown_test)),
        },
        "selected_on_calibration": selected,
        "held_out_test": metrics(
            known_scores=known_test,
            unknown_scores=unknown_test,
            threshold=threshold,
        ),
        "methodology": (
            "Threshold selected only on the calibration split, then frozen "
            "and evaluated on a disjoint held-out test split."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
