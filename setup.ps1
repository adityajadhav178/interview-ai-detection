# setup.ps1
# Run this script to set up or verify the Python virtual environment.
# Usage: .\setup.ps1

Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "  Interview AI Detection - Setup" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan

# Step 1: Create virtual environment if it doesn't exist
if (Test-Path ".venv") {
    Write-Host "`n[1/4] .venv already exists. Skipping creation." -ForegroundColor Gray
} else {
    Write-Host "`n[1/4] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
    if (-not $?) { Write-Host "ERROR: Failed to create venv. Is Python installed?" -ForegroundColor Red; exit 1 }
    Write-Host "      Done." -ForegroundColor Green
}

# Step 2: Activate venv
Write-Host "[2/4] Activating virtual environment..." -ForegroundColor Yellow
& ".venv\Scripts\Activate.ps1"
if (-not $?) { Write-Host "ERROR: Could not activate venv." -ForegroundColor Red; exit 1 }
Write-Host "      Done." -ForegroundColor Green

# Step 3: Upgrade pip (with user prompt)
$upgradePip = Read-Host "`n[3/4] Do you want to upgrade pip? (y/n)"
if ($upgradePip -eq 'y' -or $upgradePip -eq 'Y') {
    Write-Host "      Upgrading pip..." -ForegroundColor Yellow
    python -m pip install --upgrade pip --quiet
    Write-Host "      Done." -ForegroundColor Green
} else {
    Write-Host "      Skipped." -ForegroundColor Gray
}

# Step 4: Check all requirements and optionally reinstall
Write-Host "`n[4/4] Checking requirements from requirements.txt..." -ForegroundColor Yellow

$requirements = Get-Content "requirements.txt" | Where-Object { $_ -notmatch '^\s*#' -and $_.Trim() -ne '' }
$missing = @()
$installed = @()

foreach ($pkg in $requirements) {
    # Strip version specifiers to get importable name
    $pkgName = ($pkg -replace '[>=<!].*', '').Trim()
    # Map pip names to import names that differ
    $importMap = @{
        "scikit-learn"      = "sklearn"
        "opencv-python"     = "cv2"
        "praat-parselmouth" = "parselmouth"
        "pillow"            = "PIL"
        "pyyaml"            = "yaml"
        "python-dotenv"     = "dotenv"
        "imageio-ffmpeg"    = "imageio_ffmpeg"
        "vadersentiment"    = "vaderSentiment"
        "openai-whisper"    = "whisper"
        "webrtcvad-wheels"  = "webrtcvad"
        "umap-learn"        = "umap"
        "imbalanced-learn"  = "imblearn"
    }

    $importName = if ($importMap.ContainsKey($pkgName.ToLower())) { $importMap[$pkgName.ToLower()] } else { $pkgName }

    $check = python -c "import $importName" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] $pkgName" -ForegroundColor Green
        $installed += $pkgName
    } else {
        Write-Host "  [MISSING] $pkgName" -ForegroundColor Red
        $missing += $pkgName
    }
}

Write-Host "`n  Installed : $($installed.Count)" -ForegroundColor Green
Write-Host "  Missing   : $($missing.Count)" -ForegroundColor $(if ($missing.Count -gt 0) { 'Red' } else { 'Green' })

# Ask to reinstall
$reinstall = Read-Host "`nDo you want to (re)install all requirements from requirements.txt? (y/n)"
if ($reinstall -eq 'y' -or $reinstall -eq 'Y') {
    Write-Host "      Installing..." -ForegroundColor Yellow
    pip install -r requirements.txt
    Write-Host "      Done." -ForegroundColor Green
} else {
    Write-Host "      Skipped." -ForegroundColor Gray
}

# # Verify key libraries
# $verify = Read-Host "`nDo you want to verify key libraries? (y/n)"
# if ($verify -eq 'y' -or $verify -eq 'Y') {
#     Write-Host "`n--- Verifying Key Libraries ---" -ForegroundColor Cyan
#     $verify_libs = @("librosa", "parselmouth", "webrtcvad", "lightgbm", "xgboost", "sklearn", "imblearn", "noisereduce")
#     foreach ($lib in $verify_libs) {
#         $result = python -c "import $lib; print('  [OK] $lib')" 2>&1
#         if ($LASTEXITCODE -eq 0) {
#             Write-Host $result -ForegroundColor Green
#         } else {
#             Write-Host "  [FAIL] $lib" -ForegroundColor Red
#         }
#     }
# }

Write-Host "`n=======================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "  Activate anytime with:" -ForegroundColor White
Write-Host "  .venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "=======================================" -ForegroundColor Cyan
