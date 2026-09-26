# Dataset contract

The project uses one directory per flag class.

```text
data/raw/
├── ci/
├── fr/
├── jp/
├── tw/
├── xk/
└── ...
```

The worldwide taxonomy contains all ISO 3166-1 coded entities plus Kosovo as a separately documented non-ISO `XK` class.

## Data sources

### Real-world source

The real-world baseline uses annotated photographs from **Flagnet**.

```bash
python scripts/import_flagnet.py
```

The importer:

1. downloads the source archive;
2. reads Pascal-VOC flag bounding boxes;
3. crops the annotated flag region;
4. stores crops under `data/raw/<class-code>/`;
5. preserves per-image author, license and source metadata.

### Worldwide canonical coverage

```bash
python scripts/expand_worldwide.py
```

This supplements the real photographs with canonical flags for worldwide class coverage and deterministic presentation variants.

Generated presentation variants include:

- native/canonical landscape;
- portrait framing;
- square framing;
- 90° image rotation;
- moderate free rotation;
- circular presentation;
- rounded-rectangle presentation;
- low-resolution rendering;
- blur;
- brightness/contrast variation;
- partial visibility.

The generator preserves non-rectangular source geometry where available before compositing onto a neutral canvas.

## Identity vs presentation

The classifier must learn **flag identity**, not one fixed display geometry.

The same flag may appear:

- rectangular;
- circular as an icon/badge;
- inside a rounded container;
- landscape;
- portrait;
- rotated because of camera orientation;
- partially visible;
- under perspective distortion;
- at different scales and resolutions.

Horizontal and vertical flips are deliberately **not** part of the standard augmentation pipeline because mirroring an asymmetric flag can alter its visual semantics.

## Leakage rules

Generated variants from one canonical source must never inflate independent evaluation.

For **seen classes**:

- canonical/generated presentation variants are **training-only**;
- validation and test use independent real-world photographs whenever available.

For **unseen classes**:

- the class is completely absent from supervised training;
- real-world unknown images are evaluated separately from controlled canonical/presentation variants.

The manifest records:

- `source_type`;
- `evaluation_scope`;
- `regime`;
- `partition`.

This allows the final report to distinguish genuine real-world evidence from controlled robustness tests.

## Generate seen/unseen splits

```bash
python scripts/prepare_splits.py
```

The generated file is:

```text
data/splits/split_manifest.csv
```

No unseen class enters supervised training or validation.

## Why split by class?

A random image split cannot evaluate unseen-country behavior because every class remains represented during supervised training. This project holds out complete flag classes before training so open-set evaluation measures a genuinely different recognition regime.
