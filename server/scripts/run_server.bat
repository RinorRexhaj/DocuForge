@echo off
REM DocuForge API Server Startup Script
REM This script starts the FastAPI backend server

echo.
echo ============================================================
echo   DocuForge - Document Forgery Detection API Server
echo ============================================================
echo.

REM Check if main.py exists
if not exist "api\main.py" (
    echo [ERROR] api\main.py not found!
    echo Please ensure you're running this script from the DocuForge\server directory.
    pause
    exit /b 1
)

REM Check if model exists
if not exist "models\saved_models\best_model.pth" (
    echo [WARNING] Model file not found at models\saved_models\best_model.pth
    echo The server will start but predictions will fail!
    echo.
)

echo [INFO] Starting FastAPI server...
echo [INFO] Server will be available at: http://localhost:8000
echo [INFO] API Documentation: http://localhost:8000/docs
echo [INFO] Press CTRL+C to stop the server
echo.
echo ============================================================
echo.

REM Start the server
python api\main.py

REM If python fails, show error
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start server!
    echo Please ensure:
    echo   1. Python is installed
    echo   2. Required packages are installed: pip install -r requirements\requirements_api.txt
    echo   3. api\main.py exists in the current directory
    pause
    exit /b 1
)
