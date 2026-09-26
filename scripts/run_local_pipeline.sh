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
  echo "Generating split manifest..."
  PYTHONPATH=src python scripts/prepare_splits.py
fi

echo
echo "=== Training baseline ==="
PYTHONPATH=src python scripts/train.py

echo
echo "=== Evaluating baseline ==="
PYTHONPATH=src python scripts/evaluate.py

echo
echo "Pipeline completed successfully."
echo "Launch the dashboard with:"
echo "  streamlit run app.py"
