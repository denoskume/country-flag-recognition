"""External benchmark for the worldwide deployment checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
import torch

from flag_recognition.inference import load_inference_bundle
from flag_recognition.metrics import classification_summary, open_set_summary
from flag_recognition.transforms import build_eval_transform

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/models/worldwide_mobilenet_v3_small.pt"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/external_benchmark"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("artifacts/metrics"))
    parser.add_argument(
        "--taxonomy-audit",
        type=Path,
        default=Path("artifacts/metrics/taxonomy_ambiguity_audit.json"),
    )
    return parser.parse_args()


def iter_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_identical_groups(path: Path) -> list[list[str]]:
    if not path.is_file():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        sorted(str(code) for code in group)
        for group in payload.get("identical_groups", [])
        if len(group) > 1
    ]


def alias_map(
    classes: list[str],
    identical_groups: list[list[str]],
) -> dict[str, str]:
    mapping = {code: code for code in classes}

    for group in identical_groups:
        canonical = "|".join(sorted(group))
        for code in group:
            if code in mapping:
                mapping[code] = canonical

    return mapping


def taxonomy_aware_summary(
    probabilities_matrix: np.ndarray,
    targets: np.ndarray,
    index_to_class: dict[int, str],
    class_aliases: dict[str, str],
) -> dict[str, float]:
    predictions = probabilities_matrix.argmax(axis=1)
    top5 = np.argsort(probabilities_matrix, axis=1)[:, -5:][:, ::-1]

    target_groups = np.asarray([
        class_aliases[index_to_class[int(target)]]
        for target in targets
    ])
    prediction_groups = np.asarray([
        class_aliases[index_to_class[int(prediction)]]
        for prediction in predictions
    ])

    top1_correct = target_groups == prediction_groups
    top5_correct = np.asarray([
        target_groups[row_index]
        in {
            class_aliases[index_to_class[int(candidate)]]
            for candidate in top5[row_index]
        }
        for row_index in range(len(targets))
    ])

    return {
        "top1_accuracy": float(np.mean(top1_correct)),
        "top5_accuracy": float(np.mean(top5_correct)),
        "macro_precision": float(
            precision_score(
                target_groups,
                prediction_groups,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                target_groups,
                prediction_groups,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                target_groups,
                prediction_groups,
                average="macro",
                zero_division=0,
            )
        ),
    }


@torch.inference_mode()
def probabilities(path: Path, bundle, transform) -> np.ndarray:
    with Image.open(path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(bundle.device)
    return torch.softmax(bundle.model(tensor), dim=1)[0].cpu().numpy()


def main():
    args = parse_args()
    bundle = load_inference_bundle(args.checkpoint)
    transform = build_eval_transform(bundle.image_size)

    raw_hashes = {file_hash(p) for p in iter_images(args.raw_dir)}
    class_to_index = {name: index for index, name in bundle.index_to_class.items()}

    identical_groups = load_identical_groups(args.taxonomy_audit)
    class_aliases = alias_map(
        classes=sorted(class_to_index),
        identical_groups=identical_groups,
    )

    known_probs = []
    known_targets = []
    rows = []

    known_root = args.benchmark_dir / "known"
    for class_dir in sorted(p for p in known_root.iterdir() if p.is_dir()) if known_root.exists() else []:
        target_name = class_dir.name
        if target_name not in class_to_index:
            raise RuntimeError(f"Unknown checkpoint class folder: {target_name}")

        for path in iter_images(class_dir):
            if file_hash(path) in raw_hashes:
                raise RuntimeError(f"Benchmark leakage: {path} is byte-identical to data/raw.")

            probs = probabilities(path, bundle, transform)
            pred = int(probs.argmax())
            conf = float(probs[pred])
            target = class_to_index[target_name]
            top5 = probs.argsort()[-5:][::-1]

            known_probs.append(probs)
            known_targets.append(target)
            prediction_name = bundle.index_to_class[pred]
            taxonomy_correct_top1 = (
                class_aliases[target_name]
                == class_aliases[prediction_name]
            )
            taxonomy_correct_top5 = any(
                class_aliases[target_name]
                == class_aliases[bundle.index_to_class[int(candidate)]]
                for candidate in top5
            )

            rows.append({
                "path": path.as_posix(),
                "scope": "known",
                "target": target_name,
                "prediction": prediction_name,
                "confidence": conf,
                "accepted_as_known": conf >= bundle.unknown_threshold,
                "correct_top1": pred == target,
                "correct_top5": target in top5,
                "taxonomy_correct_top1": taxonomy_correct_top1,
                "taxonomy_correct_top5": taxonomy_correct_top5,
                "target_visual_group": class_aliases[target_name],
                "prediction_visual_group": class_aliases[prediction_name],
            })

    unknown_scores = []
    unknown_root = args.benchmark_dir / "unknown"
    for path in iter_images(unknown_root):
        if file_hash(path) in raw_hashes:
            raise RuntimeError(f"Benchmark leakage: {path} is byte-identical to data/raw.")

        probs = probabilities(path, bundle, transform)
        pred = int(probs.argmax())
        conf = float(probs[pred])
        unknown_scores.append(conf)
        rows.append({
            "path": path.as_posix(),
            "scope": "unknown",
            "target": "unknown",
            "prediction": bundle.index_to_class[pred],
            "confidence": conf,
            "accepted_as_known": conf >= bundle.unknown_threshold,
            "correct_top1": False,
            "correct_top5": False,
            "taxonomy_correct_top1": False,
            "taxonomy_correct_top5": False,
            "target_visual_group": "unknown",
            "prediction_visual_group": class_aliases[bundle.index_to_class[pred]],
        })

    if not known_probs:
        raise RuntimeError(f"No known benchmark images found under {known_root}.")
    if not unknown_scores:
        raise RuntimeError(f"No unknown benchmark images found under {unknown_root}.")

    known_probs = np.stack(known_probs)
    known_targets = np.asarray(known_targets, dtype=int)
    known_predictions = known_probs.argmax(axis=1)
    known_scores = known_probs.max(axis=1)
    unknown_scores = np.asarray(unknown_scores, dtype=float)

    closed_set = classification_summary(known_probs, known_targets)
    taxonomy_aware_closed_set = taxonomy_aware_summary(
        probabilities_matrix=known_probs,
        targets=known_targets,
        index_to_class=bundle.index_to_class,
        class_aliases=class_aliases,
    )
    open_set = open_set_summary(known_scores, unknown_scores)
    threshold = float(bundle.unknown_threshold)

    table = pd.DataFrame(rows)
    labels = sorted(set(known_targets.tolist()))
    recalls = recall_score(known_targets, known_predictions, labels=labels, average=None, zero_division=0)
    per_country = pd.DataFrame({
        "country": [bundle.index_to_class[int(label)] for label in labels],
        "recall": recalls,
        "images": [int(np.sum(known_targets == label)) for label in labels],
    }).sort_values(["recall", "country"])

    strict_errors = table[
        (table["scope"] == "known")
        & (~table["correct_top1"])
    ]
    taxonomy_errors = table[
        (table["scope"] == "known")
        & (~table["taxonomy_correct_top1"])
    ]

    def top_confusions(error_table: pd.DataFrame) -> list[dict[str, object]]:
        if error_table.empty:
            return []

        grouped = (
            error_table
            .groupby(["target", "prediction"], as_index=False)
            .size()
            .sort_values("size", ascending=False)
            .head(20)
        )
        return [
            {
                "target": str(row["target"]),
                "prediction": str(row["prediction"]),
                "count": int(row["size"]),
            }
            for row in grouped.to_dict("records")
        ]

    strict_confusions = top_confusions(strict_errors)
    taxonomy_confusions = top_confusions(taxonomy_errors)

    class_counts = table[table["scope"] == "known"].groupby("target").size()
    under_sampled = sorted(str(name) for name, count in class_counts.items() if int(count) < 3)

    results = {
        "checkpoint": str(args.checkpoint),
        "known_images": int(len(known_targets)),
        "known_classes_tested": int(table.loc[table["scope"] == "known", "target"].nunique()),
        "unknown_images": int(len(unknown_scores)),
        "closed_set_strict": closed_set,
        "closed_set_taxonomy_aware": taxonomy_aware_closed_set,
        "taxonomy_ambiguity": {
            "audit_path": str(args.taxonomy_audit),
            "identical_groups": identical_groups,
            "strict_top1_errors": int(len(strict_errors)),
            "taxonomy_aware_top1_errors": int(len(taxonomy_errors)),
            "errors_reclassified_as_visually_equivalent": int(
                len(strict_errors) - len(taxonomy_errors)
            ),
        },
        "open_set": open_set,
        "threshold_audit": {
            "threshold": threshold,
            "known_acceptance_rate": float(np.mean(known_scores >= threshold)),
            "known_rejection_rate": float(np.mean(known_scores < threshold)),
            "unknown_rejection_rate": float(np.mean(unknown_scores < threshold)),
            "unknown_false_acceptance_rate": float(np.mean(unknown_scores >= threshold)),
        },
        "classes_with_fewer_than_3_external_images": under_sampled,
        "top_confusions_strict": strict_confusions,
        "top_confusions_taxonomy_aware": taxonomy_confusions,
        "leakage_check": "passed",
    }

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    (args.metrics_dir / "deployment_external_evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    table.to_csv(args.metrics_dir / "deployment_external_predictions.csv", index=False)
    per_country.to_csv(args.metrics_dir / "deployment_external_per_country_recall.csv", index=False)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
