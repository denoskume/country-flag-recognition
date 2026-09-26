# Run locally — VS Code + Ubuntu/WSL

This is the reference workflow for training, evaluation and dashboard execution.

## 1. Open the repository in Ubuntu

If the repository is not cloned yet:

```bash
cd ~
git clone https://github.com/denoskume/country-flag-recognition.git
cd country-flag-recognition
code .
```

If it already exists:

```bash
cd ~/country-flag-recognition
git pull
code .
```

Run all following commands from the VS Code Ubuntu terminal at the repository root.

## 2. Create the Python environment

```bash
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
```

Check PyTorch and the available compute device:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Device: CPU")
PY
```

The training scripts automatically use CUDA when available and otherwise use CPU.

## 3. Extract the worldwide dataset

Download:

```text
country-flag-recognition-worldwide-dataset-v2.zip
```

Then extract it directly into the repository root.

Example from Windows Downloads under WSL:

```bash
unzip -o /mnt/c/Users/denos/Downloads/country-flag-recognition-worldwide-dataset-v2.zip -d .
```

After extraction, this must exist:

```text
data/raw/
data/splits/split_manifest.csv
data/source_metadata/
data/taxonomy.csv
```

## 4. Verify the dataset

```bash
PYTHONPATH=src python scripts/verify_dataset.py
```

Expected project contract:

```text
250 classes
4,158 images
200 research seen classes
50 research unseen classes
```

The verifier also checks that generated canonical variants do not leak into seen-class validation/test.

## 5. Run the full ML pipeline

```bash
bash scripts/run_local_pipeline.sh
```

This executes, in order:

1. research MobileNetV3 baseline on 200 seen classes;
2. open-set evaluation against 50 unseen classes;
3. final MobileNetV3 deployment model using all 250 worldwide classes.

Generated files:

```text
artifacts/models/baseline_mobilenet_v3_small.pt
artifacts/models/worldwide_mobilenet_v3_small.pt

artifacts/metrics/training_history.json
artifacts/metrics/baseline_evaluation.json
artifacts/metrics/deployment_training_history.json
artifacts/metrics/deployment_summary.json
```

## 6. Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, normally:

```text
http://localhost:8501
```

The dashboard uses the final 250-class worldwide checkpoint:

```text
artifacts/models/worldwide_mobilenet_v3_small.pt
```

## 7. Stop the application

Use:

```text
Ctrl+C
```

in the terminal running Streamlit.

## Research vs deployment

The project deliberately keeps two model roles separate.

**Research benchmark**

```text
200 seen classes
50 completely unseen classes
        ↓
closed-set + open-set evaluation
```

**Final dashboard model**

```text
all 250 worldwide classes
        ↓
production recognition
        ↓
live country intelligence dashboard
```

This keeps the open-set experiment scientifically meaningful while allowing the final application to recognize the full worldwide taxonomy.
