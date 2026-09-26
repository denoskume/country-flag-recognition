# External Deployment Benchmark

This folder is reserved for images independent from supervised training.

```text
data/external_benchmark/
├── known/
│   ├── fr/
│   ├── ci/
│   ├── jp/
│   └── ...
└── unknown/
    ├── non_flags/
    ├── objects/
    ├── logos/
    └── scenes/
```

For known flags, folder names must match checkpoint class labels. Use at least
three independent images per tested class whenever possible, including clean
and real-world presentations.

Do not copy images from `data/raw`. The evaluator checks SHA-256 hashes and
stops if it finds a byte-identical training image.

Run:

```bash
PYTHONPATH=src python scripts/evaluate_deployment.py
```

Outputs are written to `artifacts/metrics/`:

- `deployment_external_evaluation.json`
- `deployment_external_predictions.csv`
- `deployment_external_per_country_recall.csv`

The benchmark measures the current model and threshold; it does not alter them.
