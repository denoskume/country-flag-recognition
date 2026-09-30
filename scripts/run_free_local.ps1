param(
    [string]$Model = "qwen3:8b",
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"

Write-Host "Flag Intelligence — zero-cost local mode" -ForegroundColor Cyan

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama is not installed. Install it from https://ollama.com/download"
}

$env:FLAG_INTELLIGENCE_LLM_BACKEND = "ollama"
$env:FLAG_INTELLIGENCE_OLLAMA_MODEL = $Model
$env:OLLAMA_BASE_URL = "http://127.0.0.1:11434"

Write-Host "Ensuring model '$Model' is available..."
ollama pull $Model

Write-Host "Starting Flag Intelligence locally on port $Port..."
Write-Host "Local URL: http://localhost:$Port"
Write-Host ""
Write-Host "To share it temporarily for free, open another terminal and run:"
Write-Host "  cloudflared tunnel --url http://localhost:$Port" -ForegroundColor Yellow

python -m streamlit run app.py --server.port $Port
