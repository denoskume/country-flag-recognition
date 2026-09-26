#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Run:"
  echo "  bash scripts/setup_ubuntu.sh"
  exit 1
fi

source .venv/bin/activate

if [ ! -d "data/raw" ]; then
  echo "Missing data/raw."
  echo "Extract the worldwide dataset ZIP into the repository root first."
  exit 1
fi

if [ ! -f "data/splits/split_manifest.csv" ]; then
  echo "Generating research split manifest..."
  PYTHONPATH=src python scripts/prepare_splits.py
fi

echo
echo "=== Dataset verification ==="
PYTHONPATH=src python scripts/verify_dataset.py

echo
echo "=== 1/3 Research baseline: 200 seen / 50 unseen ==="
PYTHONPATH=src python scripts/train.py

echo
echo "=== 2/3 Research evaluation ==="
PYTHONPATH=src python scripts/evaluate.py

echo
echo "=== 3/3 Worldwide deployment model: all 250 classes ==="
PYTHONPATH=src python scripts/train_deployment.py

echo
echo "Local ML pipeline completed successfully."
echo
echo "Research checkpoint:"
echo "  artifacts/models/baseline_mobilenet_v3_small.pt"
echo
echo "Worldwide dashboard checkpoint:"
echo "  artifacts/models/worldwide_mobilenet_v3_small.pt"
echo
echo "Launch the dashboard with:"
echo "  streamlit run app.py"
