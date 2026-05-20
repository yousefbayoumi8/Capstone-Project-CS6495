# SecureBank + Open WebUI — One-time Windows setup
# Run from PowerShell in the project folder: .\setup.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

Write-Host ""
Write-Host "=== SecureBank + Open WebUI setup ===" -ForegroundColor Cyan
Write-Host ""

# Sanity: are we in the right folder?
if (-not (Test-Path (Join-Path $ProjectRoot "webui2.py"))) {
    Write-Host "webui2.py not found next to this script." -ForegroundColor Red
    Write-Host "Put setup.ps1 in the Capstone-Project-CS6495 folder and try again."
    exit 1
}

# [1/5] Python
Write-Host "[1/5] Checking Python..." -ForegroundColor Yellow
$pyVersion = ""
try { $pyVersion = (python --version 2>&1).ToString() } catch {}
if ($pyVersion -notmatch "Python 3\.(1[0-9]|[2-9][0-9])") {
    Write-Host "    Need Python 3.10+. Got: '$pyVersion'" -ForegroundColor Red
    Write-Host "    Install from https://www.python.org/downloads/  (check 'Add python.exe to PATH')."
    exit 1
}
Write-Host "    OK ($pyVersion)" -ForegroundColor Green

# [2/5] Docker
Write-Host "[2/5] Checking Docker..." -ForegroundColor Yellow
try {
    docker --version | Out-Null
    docker ps     | Out-Null
} catch {
    Write-Host "    Docker isn't running." -ForegroundColor Red
    Write-Host "    1. Install Docker Desktop: https://www.docker.com/products/docker-desktop/"
    Write-Host "    2. Launch it. Wait for 'Engine running' in the tray."
    Write-Host "    3. Re-run .\setup.ps1"
    exit 1
}
Write-Host "    OK" -ForegroundColor Green

# [3/5] Backend venv
$VenvPath = Join-Path $ProjectRoot "prompt_injection_env"
$VenvPip  = Join-Path $VenvPath "Scripts\pip.exe"
Write-Host "[3/5] Backend venv..." -ForegroundColor Yellow
if (-not (Test-Path $VenvPath)) {
    python -m venv $VenvPath
    Write-Host "    Created $VenvPath" -ForegroundColor Green
} else {
    Write-Host "    Reusing existing venv" -ForegroundColor Green
}

# [4/5] Python deps (this is the slow part)
Write-Host "[4/5] Installing Python dependencies (5-15 min on first run)..." -ForegroundColor Yellow
& $VenvPip install --upgrade pip
& $VenvPip install -r (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    Write-Host "    pip install failed. See output above." -ForegroundColor Red
    Write-Host "    Common fix: enable long paths (run PowerShell as Admin):"
    Write-Host "      New-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force"
    Write-Host "    Then reboot and re-run .\setup.ps1"
    exit 1
}
Write-Host "    Dependencies installed" -ForegroundColor Green

# [5/5] Open WebUI image
Write-Host "[5/5] Pulling Open WebUI image (~1.5 GB)..." -ForegroundColor Yellow
docker pull ghcr.io/open-webui/open-webui:main
if ($LASTEXITCODE -ne 0) {
    Write-Host "    docker pull failed." -ForegroundColor Red
    exit 1
}
Write-Host "    Image ready" -ForegroundColor Green

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Cyan
Write-Host "Next: run  .\start.ps1   (or  .\start.ps1 llama  for a different model)"
Write-Host ""
