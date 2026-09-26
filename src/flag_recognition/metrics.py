"""Classification, calibration and open-set evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def top_k_accuracy(
    probabilities: np.ndarray,
    targets: np.ndarray,
    k: int,
) -> float:
    """Compute multiclass top-k accuracy from class probabilities."""
    probabilities = np.asarray(probabilities)
    targets = np.asarray(targets)

    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [N, C].")

    if len(targets) != probabilities.shape[0]:
        raise ValueError("targets length must match probability rows.")

    k = min(int(k), probabilities.shape[1])

    if k < 1:
        raise ValueError("k must be at least 1.")

    top_k = np.argpartition(
        probabilities,
        -k,
        axis=1,
    )[:, -k:]

    return float(
        np.mean(
            [
                target in candidates
                for target, candidates in zip(
                    targets,
                    top_k,
                )
            ]
        )
    )


def expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Estimate calibration error from maximum class confidence."""
    probabilities = np.asarray(probabilities, dtype=float)
    targets = np.asarray(targets, dtype=int)

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = predictions == targets

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (
                (confidences >= lower)
                & (confidences <= upper)
            )
        else:
            mask = (
                (confidences >= lower)
                & (confidences < upper)
            )

        if not mask.any():
            continue

        bin_accuracy = correctness[mask].mean()
        bin_confidence = confidences[mask].mean()
        bin_weight = mask.mean()

        ece += (
            abs(bin_accuracy - bin_confidence)
            * bin_weight
        )

    return float(ece)


def negative_log_likelihood(
    probabilities: np.ndarray,
    targets: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    """Compute multiclass negative log-likelihood."""
    probabilities = np.asarray(probabilities, dtype=float)
    targets = np.asarray(targets, dtype=int)

    selected = probabilities[
        np.arange(len(targets)),
        targets,
    ]

    return float(
        -np.mean(
            np.log(
                np.clip(
                    selected,
                    epsilon,
                    1.0,
                )
            )
        )
    )


def classification_summary(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """Return the core closed-set classification metrics."""
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )
    predictions = probabilities.argmax(axis=1)
    targets = np.asarray(targets)

    return {
        "top1_accuracy": float(
            accuracy_score(targets, predictions)
        ),
        "top5_accuracy": top_k_accuracy(
            probabilities,
            targets,
            k=5,
        ),
        "macro_precision": float(
            precision_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "negative_log_likelihood": (
            negative_log_likelihood(
                probabilities,
                targets,
            )
        ),
        "expected_calibration_error": (
            expected_calibration_error(
                probabilities,
                targets,
            )
        ),
    }


def fpr_at_tpr(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    target_tpr: float = 0.95,
) -> float:
    """Compute unknown false-positive rate at a target known acceptance rate."""
    known_scores = np.asarray(known_scores, dtype=float)
    unknown_scores = np.asarray(unknown_scores, dtype=float)

    if not 0.0 < target_tpr <= 1.0:
        raise ValueError("target_tpr must be in (0, 1].")

    threshold = np.quantile(
        known_scores,
        1.0 - target_tpr,
    )

    return float(
        np.mean(unknown_scores >= threshold)
    )


def open_set_summary(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
) -> dict[str, float]:
    """Evaluate max-confidence scores as a known-vs-unknown detector."""
    known_scores = np.asarray(known_scores, dtype=float)
    unknown_scores = np.asarray(unknown_scores, dtype=float)

    labels = np.concatenate(
        [
            np.ones_like(known_scores, dtype=int),
            np.zeros_like(unknown_scores, dtype=int),
        ]
    )
    scores = np.concatenate(
        [
            known_scores,
            unknown_scores,
        ]
    )

    return {
        "auroc": float(
            roc_auc_score(labels, scores)
        ),
        "aupr_known": float(
            average_precision_score(labels, scores)
        ),
        "fpr_at_95_tpr": fpr_at_tpr(
            known_scores,
            unknown_scores,
            target_tpr=0.95,
        ),
    }
