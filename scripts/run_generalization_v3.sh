#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Run: bash scripts/setup_ubuntu.sh"
  exit 1
fi

source .venv/bin/activate
export PYTHONPATH=src

echo "=== Generalization V3 training ==="
python scripts/train_deployment.py --config configs/generalization_v3.yaml

echo
echo "=== Real-world V3 evaluation ==="
python scripts/evaluate_real_world_challenge.py \
  --checkpoint artifacts/models/worldwide_generalization_v3_mobilenet_v3_small.pt

echo
echo "=== V1 vs V3 real-world comparison ==="
python scripts/compare_real_world_models.py \
  --candidate artifacts/models/worldwide_generalization_v3_mobilenet_v3_small.pt

echo
echo "V3 training/evaluation complete."
echo "Do not promote V3 unless real-world Top-1 improves over V1."
