#!/usr/bin/env pwsh
# ============================================================================
#  DEEPFAKE DETECTION - Windows PowerShell Installation Script
#  Python 3.11 - One-click setup
# ============================================================================

Write-Host "`n" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "  DEEPFAKE DETECTION - Windows Installation (PowerShell)" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "`n"

Write-Host "This script will:`n" -ForegroundColor Yellow
Write-Host "  1. Create virtual environment"
Write-Host "  2. Upgrade pip"
Write-Host "  3. Install all dependencies"
Write-Host "  4. Verify installation"
Write-Host "`n"

Write-Host "Requirements: Python 3.11, Internet, 2GB free space`n" -ForegroundColor Cyan
Read-Host "Press Enter to continue"

# Function for error handling
function Check-Error {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nERROR: $args" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Check Python version
Write-Host "`n[1/5] Checking Python version..." -ForegroundColor Cyan
python --version
Check-Error "Python not found"

# Create venv
Write-Host "`n[2/5] Creating virtual environment..." -ForegroundColor Cyan
if (Test-Path "venv") {
    Write-Host "Removing old venv..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force venv
}
python -m venv venv
Check-Error "Failed to create virtual environment"
Write-Host "Virtual environment created: venv\" -ForegroundColor Green

# Activate venv
Write-Host "`nActivating virtual environment..." -ForegroundColor Cyan
& "venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "`n[3/5] Upgrading pip (this may take a minute)..." -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel --quiet
Check-Error "Failed to upgrade pip"
Write-Host "Pip upgraded successfully" -ForegroundColor Green

# Install dependencies
Write-Host "`n[4/5] Installing dependencies (this takes 5-10 minutes)..." -ForegroundColor Cyan
Write-Host "Please wait, do not interrupt...`n" -ForegroundColor Yellow
pip install -r requirements.txt
Check-Error "Installation failed"

# Verify installation
Write-Host "`n[5/5] Verifying installation..." -ForegroundColor Cyan
python -c "import torch; import mediapipe; import cv2; print('✓ All packages loaded successfully')"
Check-Error "Verification failed"

# Success
Write-Host "`n" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "  ✓ INSTALLATION COMPLETE!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "`n"

Write-Host "Virtual environment activated. Next steps:`n" -ForegroundColor Yellow
Write-Host "  1. Download models:"
Write-Host "     python scripts/download_models.py`n"
Write-Host "  2. Test detection:"
Write-Host "     python simple_test.py`n"
Write-Host "  3. Start API:"
Write-Host "     python backend/app.py`n"
Write-Host "For detailed info, read: WINDOWS_SETUP.txt" -ForegroundColor Cyan
Write-Host "`n"

Read-Host "Press Enter to exit"
