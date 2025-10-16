# DocuForge API Server Startup Script (PowerShell)
# This script starts the FastAPI backend server with proper error checking

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  DocuForge - Document Forgery Detection API Server" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if main.py exists
if (-not (Test-Path "api\main.py")) {
    Write-Host "[ERROR] api\main.py not found!" -ForegroundColor Red
    Write-Host "Please ensure you're running this script from the DocuForge\server directory." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if model exists
if (-not (Test-Path "models\saved_models\best_model.pth")) {
    Write-Host "[WARNING] Model file not found at models\saved_models\best_model.pth" -ForegroundColor Yellow
    Write-Host "The server will start but predictions will fail!" -ForegroundColor Yellow
    Write-Host ""
}

# Check if FastAPI is installed
Write-Host "[INFO] Checking dependencies..." -ForegroundColor Green
$fastapi_check = python -c "import fastapi" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] FastAPI not found. Installing dependencies..." -ForegroundColor Yellow
    Write-Host ""
    python -m pip install -r requirements\requirements_api.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install dependencies!" -ForegroundColor Red
        Write-Host "Please manually run: pip install -r requirements\requirements_api.txt" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host ""
Write-Host "[INFO] Starting FastAPI server..." -ForegroundColor Green
Write-Host "[INFO] Server will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "[INFO] API Documentation: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "[INFO] Press CTRL+C to stop the server" -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Start the server
try {
    python api\main.py
}
catch {
    Write-Host ""
    Write-Host "[ERROR] Failed to start server!" -ForegroundColor Red
    Write-Host "Error details: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please ensure:" -ForegroundColor Yellow
    Write-Host "  1. Python is installed" -ForegroundColor Yellow
    Write-Host "  2. Required packages are installed: pip install -r requirements\requirements_api.txt" -ForegroundColor Yellow
    Write-Host "  3. api\main.py exists in the current directory" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}
