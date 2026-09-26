#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is required."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating .venv..."
  python3 -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"

echo
echo "Environment ready."
echo "Activate it with:"
echo "  source .venv/bin/activate"
