#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-qwen3:8b}"
PORT="${2:-8501}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Install it from https://ollama.com/download"
  exit 1
fi

export FLAG_INTELLIGENCE_LLM_BACKEND=ollama
export FLAG_INTELLIGENCE_OLLAMA_MODEL="$MODEL"
export OLLAMA_BASE_URL=http://127.0.0.1:11434

echo "Ensuring model '$MODEL' is available..."
ollama pull "$MODEL"

echo "Starting Flag Intelligence locally on http://localhost:$PORT"
echo "To share it temporarily for free, run in another terminal:"
echo "  cloudflared tunnel --url http://localhost:$PORT"

python -m streamlit run app.py --server.port "$PORT"
