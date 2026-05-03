# Rebuild images and start the full stack. Requires Docker Desktop (Linux engine) running.
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "Checking Docker daemon..."
docker version *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker is not reachable. Start Docker Desktop and wait until it is idle, then retry." -ForegroundColor Red
    exit 1
}

Write-Host "Building images..."
docker compose build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Starting stack (detached)..."
docker compose up -d
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`nService status:"
docker compose ps

Write-Host "`nURLs:"
Write-Host "  Neo4j Browser: http://localhost:7474"
Write-Host "  API health:    http://localhost:8000/health"
Write-Host "  Frontend:      http://localhost:8501"
Write-Host "`nTip: Ollama must be running on the host (port 11434) for LLM calls inside the API container."
