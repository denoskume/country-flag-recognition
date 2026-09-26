# Country Flag Recognition

**Seen, Unseen and Zero-Shot Evaluation**

A computer-vision project for recognizing country flags under both standard and open-set conditions.

The project is designed around a harder question than ordinary image classification:

> **Can a vision system recognize familiar country flags reliably, detect flags outside its supervised training classes, and generalize to countries it was never trained to classify?**

## Scope

The project evaluates three complementary capabilities:

1. **Closed-set recognition** — classify new images from countries represented during supervised training.
2. **Open-set detection** — detect when an input flag does not belong to the supervised training classes.
3. **Zero-shot recognition** — evaluate a vision-language model on countries deliberately excluded from supervised training.

## Evaluation protocol

The split is deliberately class-aware.

```text
All country classes
        │
        ├── Seen countries
        │     ├── train images
        │     ├── validation images
        │     └── held-out test images
        │
        └── Unseen countries
              └── test images only
```

No unseen-country image is allowed into supervised training or validation.

Within the seen-country set, source images are split **before augmentation** so transformed versions of one source cannot leak across train/validation/test partitions.

## Core metrics

### Seen-country classification

- Top-1 accuracy
- Top-5 accuracy
- Macro precision
- Macro recall
- Macro F1
- Per-country recall
- Confusion matrix
- Negative log-likelihood
- Expected Calibration Error

### Open-set evaluation

- AUROC
- AUPR
- FPR@95TPR
- confidence / entropy distributions
- known-vs-unknown detection accuracy at a selected threshold

### Efficiency

- model parameter count
- model size
- mean inference latency
- throughput

## Baseline model

The first supervised baseline uses **MobileNetV3-Small** with transfer learning.

It is intentionally lightweight enough for:

- fast iteration;
- CPU inference;
- a public Streamlit demo;
- realistic deployment on a free hosting tier.

More complex models will only be added if they provide measurable value.

## Planned zero-shot baseline

A CLIP-style vision-language model will be evaluated separately using prompts such as:

```text
"a photograph of the flag of Ghana"
"a photograph of the flag of Japan"
"a photograph of the flag of Côte d'Ivoire"
```

“Unseen” always means **excluded from this project's supervised training**. It does not imply the country or flag was absent from the foundation model's original pretraining data.

## Robustness tests

The final benchmark will include controlled degradation and realistic variation:

- blur;
- low resolution;
- rotation;
- perspective distortion;
- illumination changes;
- partial occlusion;
- background clutter;
- folds / waving appearance where data are available.

Special attention will be given to visually similar flags such as:

- Chad / Romania
- Indonesia / Monaco
- Ireland / Côte d'Ivoire
- Netherlands / Luxembourg
- Australia / New Zealand

## Repository structure

```text
country-flag-recognition/
├── app.py
├── configs/
│   └── baseline.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── README.md
├── artifacts/
│   ├── models/
│   ├── metrics/
│   └── figures/
├── scripts/
│   ├── prepare_splits.py
│   ├── train.py
│   └── evaluate.py
├── src/
│   └── flag_recognition/
│       ├── dataset.py
│       ├── splits.py
│       ├── model.py
│       ├── metrics.py
│       └── inference.py
├── tests/
├── pyproject.toml
└── requirements.txt
```

## Demo target

The public app will support:

```text
Upload flag image
       ↓
Image preview
       ↓
Country prediction
       ↓
Top-5 candidates
       ↓
Confidence
       ↓
Known / Possible Unknown
       ↓
Inference time
```

The deployment target is **Streamlit Community Cloud**.

## Project status

**Phase 1 — architecture, reproducible class-aware splitting, baseline model, and evaluation foundation.**

The next milestone is the dataset pipeline: country metadata, image collection, source-level deduplication, seen/unseen class partitioning, and dataset quality checks.
