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

## Deployment interface

The Streamlit interface presents the model decision separately from the raw
top candidate:

```text
Upload image
    ↓
Worldwide classifier
    ↓
Confidence threshold
    ├── accepted → country decision + live country dashboard
    └── rejected → Unknown + top candidate for inspection only
```

The result view keeps four primary signals visible:

- decision;
- top candidate;
- confidence;
- inference latency.

Accepted predictions unlock a structured country dashboard organized into
Overview, Government and Geography tabs. Rejected predictions do not assert
country metadata.

## External deployment benchmark

The worldwide checkpoint is evaluated separately from its training data with:

```bash
PYTHONPATH=src python scripts/evaluate_deployment.py
```

External images live under `data/external_benchmark/`. The evaluator reports:

- Top-1 and Top-5 accuracy;
- macro precision, recall and F1;
- calibration metrics;
- per-country recall;
- strongest confusion pairs;
- known acceptance and rejection rates;
- unknown rejection and false-acceptance rates;
- AUROC, AUPR and FPR@95TPR;
- SHA-256 leakage checks against `data/raw`.

The benchmark does not change the model or threshold. Measurement comes first;
threshold tuning or retraining is performed only after external evidence is
available.

## Project status

**Worldwide deployment model trained — external validation and threshold audit in progress.**

Current deployment checkpoint:

`artifacts/models/worldwide_mobilenet_v3_small.pt`

The public deployment target remains **Streamlit Community Cloud**.


## Product web interface

The market-facing interface is served by FastAPI with a dedicated responsive
HTML/CSS/JavaScript frontend. This is the recommended product UI; the Streamlit
app remains available for internal inspection.

Run locally:

```bash
pip install -r requirements.txt
PYTHONPATH=src uvicorn serve:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

The product interface includes drag-and-drop upload, user-controlled decision
policy, threshold and Top-K controls, ranked candidates, accepted/rejected
decision states, live country intelligence, browser-local history, JSON export,
responsive layouts and service health information.


## Real-world robustness status

The current production reference remains the original worldwide classifier (V1).

Three approaches were evaluated on the same 102 human-approved real-world
challenge images:

- **V1 classifier:** Top-1 19.61%, Top-5 27.45%
- **Scene-aware V2:** Top-1 17.65%, Top-5 30.39%
- **Flag detector + V1 classifier:** Top-1 19.61%, Top-5 30.39%
- Detector localization rate: 73.53%

The scene-aware V2 is not promoted because its Top-1 accuracy is lower than V1.
The detect-then-classify pipeline is also not promoted because it does not
improve Top-1 accuracy over V1.

The evaluation shows that the dominant limitation is real-world recognition
generalization, not only flag localization. Further progress requires a larger,
cleaner, current real-world flag dataset with verified labels and appropriate
train/validation/test separation.

The Wikimedia challenge set remains evaluation-only and is not used for
training.
