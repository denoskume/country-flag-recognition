import numpy as np

from flag_recognition.metrics import (
    classification_summary,
    expected_calibration_error,
    open_set_summary,
    top_k_accuracy,
)


def test_top_k_accuracy():
    probabilities = np.array(
        [
            [0.80, 0.10, 0.10],
            [0.10, 0.30, 0.60],
            [0.35, 0.40, 0.25],
        ]
    )
    targets = np.array([0, 1, 2])

    assert np.isclose(
        top_k_accuracy(
            probabilities,
            targets,
            k=1,
        ),
        1 / 3,
    )
    assert np.isclose(
        top_k_accuracy(
            probabilities,
            targets,
            k=2,
        ),
        2 / 3,
    )


def test_ece_is_zero_for_perfect_certainty():
    probabilities = np.eye(3)
    targets = np.array([0, 1, 2])

    assert np.isclose(
        expected_calibration_error(
            probabilities,
            targets,
            n_bins=5,
        ),
        0.0,
    )


def test_classification_summary_contains_core_metrics():
    probabilities = np.array(
        [
            [0.90, 0.10],
            [0.20, 0.80],
            [0.70, 0.30],
            [0.10, 0.90],
        ]
    )
    targets = np.array([0, 1, 0, 1])

    summary = classification_summary(
        probabilities,
        targets,
    )

    assert summary["top1_accuracy"] == 1.0
    assert summary["macro_f1"] == 1.0
    assert summary[
        "negative_log_likelihood"
    ] > 0.0
    assert 0.0 <= summary[
        "expected_calibration_error"
    ] <= 1.0


def test_open_set_scores_separate_known_and_unknown():
    known_scores = np.array(
        [0.96, 0.92, 0.88, 0.85]
    )
    unknown_scores = np.array(
        [0.25, 0.30, 0.40, 0.35]
    )

    summary = open_set_summary(
        known_scores,
        unknown_scores,
    )

    assert summary["auroc"] == 1.0
    assert summary["aupr_known"] == 1.0
    assert summary["fpr_at_95_tpr"] == 0.0
