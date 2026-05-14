#  Start the RAG Q&A Assistant backend server 
$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".env")) {
    Write-Host "No .env file found. Please create one with at least:" -ForegroundColor Yellow
    Write-Host "ANTHROPIC_API_KEY=your_key_here"
    exit 1
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
    Write-Host "Installing dependencies (this may take a few minutes)..." -ForegroundColor Cyan
    & .\.venv\Scripts\python.exe -m pip install --upgrade pip -q
    & .\.venv\Scripts\python.exe -m pip install -r requirements.txt -q
    Write-Host "Dependencies installed." -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting RAG Q&A Assistant..." -ForegroundColor Cyan
Write-Host "  API: http://localhost:8000"
Write-Host "  UI: http://localhost:8000/app"
Write-Host "  API Docs: http://localhost:8000/docs"
Write-Host ""

& .\.venv\Scripts\uvicorn.exe backend.main:app `
    --host 0.0.0.0 `
    --port 8000 `
    --reload `
    --reload-dir backend
