# Dataset contract

The project uses one directory per country class.

```text
data/raw/
├── ci/
│   ├── ci_00000_obj00.jpg
│   └── ...
├── fr/
│   ├── fr_00000_obj00.jpg
│   └── ...
└── ...
```

Country folders use ISO-style lowercase country codes where the source dataset provides them.

## Initial real-world source

The first baseline importer uses the public **Flagnet** dataset as an external source of annotated flag photographs.

Run:

```bash
python scripts/import_flagnet.py
```

The importer:

1. downloads the source archive;
2. reads its Pascal-VOC bounding boxes;
3. crops the annotated flag region with small configurable context padding;
4. stores crops under `data/raw/<country-code>/`;
5. preserves author, license, source URL and original download URL in:

```text
data/source_metadata/flagnet_credits.csv
```

The raw/derived dataset remains local and is ignored by Git.

## Important rules

1. Raw source images are never augmented in place.
2. Train/validation/test assignment happens **before augmentation**.
3. Countries selected as unseen are excluded completely from supervised training and validation.
4. Duplicate or near-duplicate source images must not cross partitions.
5. Generated transformations belong in `data/processed/`, never in `data/raw/`.
6. Source attribution and licensing metadata must be preserved for every imported image.

## Generate seen/unseen splits

After importing data:

```bash
python scripts/prepare_splits.py
```

This writes:

```text
data/splits/split_manifest.csv
```

Each row contains:

- image path;
- country label;
- class regime: `seen` or `unseen`;
- partition: `train`, `validation`, or `test`.

For unseen countries, the partition is always `test`.

## Why split by country?

A random image split cannot test unseen-country behavior because every class remains represented during supervised training. This project deliberately holds out complete country classes before training so open-set evaluation measures a genuinely different recognition regime.
