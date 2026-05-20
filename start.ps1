# Daily-driver launcher: backend + Open WebUI + opens browser.
# Run:  .\start.ps1            (defaults to qwen)
#       .\start.ps1 llama
#       .\start.ps1 gemma

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$Model = if ($args[0]) { $args[0] } else { "qwen" }

$VenvPython  = Join-Path $ProjectRoot "prompt_injection_env\Scripts\python.exe"
$WebuiScript = Join-Path $ProjectRoot "webui2.py"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Backend venv missing. Run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}

# Docker daemon up?
try { docker ps | Out-Null } catch {
    Write-Host "Docker isn't running. Launch Docker Desktop, wait for 'Engine running', then retry." -ForegroundColor Red
    exit 1
}

# 1. Backend in a new window
Write-Host "Starting backend (model=$Model) in a new window..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "& '$VenvPython' '$WebuiScript' --model $Model --port 7860"
)

# 2. Wait for /v1/models to respond (model load can take 30-90s)
Write-Host "Waiting for the model to load..." -ForegroundColor Cyan
$ready = $false
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 2
    try {
        Invoke-RestMethod -Uri "http://127.0.0.1:7860/v1/models" -TimeoutSec 3 | Out-Null
        $ready = $true; break
    } catch {}
}
if (-not $ready) {
    Write-Host "Backend didn't come up in 3 min. Check the backend window for errors." -ForegroundColor Red
    exit 1
}
Write-Host "Backend ready" -ForegroundColor Green

# 3. Open WebUI container (create / start / no-op)
$status = docker ps -a --filter "name=^open-webui$" --format "{{.Status}}"
if (-not $status) {
    Write-Host "Creating Open WebUI container..." -ForegroundColor Cyan
    docker run -d --name open-webui -p 3000:8080 `
        -e OPENAI_API_BASE_URL=http://host.docker.internal:7860/v1 `
        -e OPENAI_API_KEY=sk-local `
        -e WEBUI_AUTH=False `
        -v open-webui:/app/backend/data `
        --restart unless-stopped `
        ghcr.io/open-webui/open-webui:main | Out-Null
} elseif ($status -notmatch "^Up") {
    Write-Host "Starting existing Open WebUI container..." -ForegroundColor Cyan
    docker start open-webui | Out-Null
} else {
    Write-Host "Open WebUI already running" -ForegroundColor Green
}

# 4. Browser
Start-Sleep -Seconds 3
Write-Host "Opening http://localhost:3000" -ForegroundColor Cyan
Start-Process "http://localhost:3000"

Write-Host ""
Write-Host "Open WebUI keeps running in Docker; stop it with:  docker stop open-webui"
