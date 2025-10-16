# Quick Start Guide - Running the DocuForge API Server

## 🚀 Quick Start (Choose One Method)

### Method 1: Simple Command (Recommended for PowerShell)

```powershell
python main.py
```

### Method 2: Using Uvicorn Directly

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Method 3: Batch Script (Windows)

```powershell
.\run_server.bat
```

### Method 4: PowerShell Script (Windows)

```powershell
.\run_server.ps1
```

### Method 5: Python Startup Script (Cross-platform with auto-checks)

```powershell
python start_server.py
```

## 📋 Prerequisites

### Install Required Packages:

```powershell
pip install fastapi uvicorn[standard] python-multipart
```

Or install all at once:

```powershell
pip install -r requirements_api.txt
```

### Verify Installation:

```powershell
python -c "import fastapi, uvicorn; print('✅ Ready to go!')"
```

## 🧪 Testing the Server

### 1. Check if server is running:

Open your browser and go to: **http://localhost:8000**

You should see:

```json
{
  "message": "DocuForge API is running",
  "status": "healthy",
  "model_loaded": true,
  ...
}
```

### 2. View API Documentation:

Go to: **http://localhost:8000/docs**

This opens the interactive Swagger UI where you can test the API directly.

### 3. Test with a file using PowerShell:

```powershell
$file = Get-Item "path\to\your\image.jpg"
Invoke-WebRequest -Uri "http://localhost:8000/predict" -Method Post -Form @{file=$file}
```

### 4. Test with Python script:

```powershell
python test_client.py path\to\your\image.jpg
```

## 🛑 Stopping the Server

Press **CTRL+C** in the terminal where the server is running.

## ⚠️ Troubleshooting

### Port 8000 already in use:

```powershell
# Find what's using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with the actual process ID)
taskkill /PID <PID> /F
```

### FastAPI not found:

```powershell
pip install fastapi uvicorn[standard] python-multipart
```

### Model not loading:

Ensure the model file exists at:

```
saved_models\best_model.pth
```

## 📁 Files Created

- **main.py** - Main FastAPI server application
- **run_server.bat** - Windows batch script to start server
- **run_server.ps1** - PowerShell script with dependency checks
- **start_server.py** - Python script with auto-checks and installation
- **test_client.py** - Test client to send requests to the API
- **requirements_api.txt** - Python dependencies
- **API_README.md** - Comprehensive API documentation

## 🎯 What Each Script Does

| Script            | Features                                   |
| ----------------- | ------------------------------------------ |
| `python main.py`  | Simple, direct start                       |
| `run_server.bat`  | Windows batch with basic checks            |
| `run_server.ps1`  | PowerShell with color output and checks    |
| `start_server.py` | Full validation, auto-install dependencies |

Choose the one that works best for your workflow!

## 📚 Next Steps

1. Start the server using any method above
2. Open http://localhost:8000/docs to explore the API
3. Test predictions using the test client or browser interface
4. Integrate with your frontend application

For detailed API documentation, see **API_README.md**
