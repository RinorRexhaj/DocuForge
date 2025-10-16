# Quick Start Guide - DocuForge (Restructured)

## 🚀 Quick Start

Your codebase has been restructured! Here's how to get started:

### 1. Start the Server

**Windows (PowerShell):**

```powershell
.\scripts\run_server.ps1
```

**Windows (Command Prompt):**

```cmd
.\scripts\run_server.bat
```

**Alternative (Direct Python):**

```bash
python api\main.py
```

### 2. Test the API

Open your browser and go to:

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Main Endpoint**: http://localhost:8000

### 3. Make a Prediction

**Using curl:**

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@path/to/image.jpg"
```

**Using Python:**

```python
import requests

with open('document.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)
    print(response.json())
```

**Using the test client:**

```bash
python tests\test_client.py path\to\image.jpg
```

## 📁 New Folder Structure

```
server/
├── api/              # FastAPI server
├── models/           # ML models
├── detection/        # Detection algorithms
├── tests/            # Tests
├── examples/         # Usage examples
├── notebooks/        # Jupyter notebooks
├── docs/             # Documentation
├── scripts/          # Startup scripts
└── requirements/     # Dependencies
```

## 📖 Key Files

- **`README.md`** - Full documentation
- **`RESTRUCTURING_SUMMARY.md`** - Details of changes made
- **`api/main.py`** - FastAPI application
- **`models/predict.py`** - Prediction functions
- **`detection/tampering_localization.py`** - Tampering detection

## 🧪 Running Tests

```bash
# Test the API client
python tests\test_client.py path\to\image.jpg

# Test tampering detection
python tests\test_tampering_module.py

# Test blur detection
python tests\test_blur_detection_forgery.py
```

## 💡 Running Examples

```bash
# Basic usage
python examples\example_usage.py

# Tampering detection
python examples\example_tampering_usage.py

# Full integration
python examples\integration_example.py

# Blur detection demo
python examples\demo_blur_detection.py
```

## ⚙️ Installation

If you haven't installed dependencies yet:

```bash
# For API functionality
pip install -r requirements\requirements_api.txt

# For tampering detection features
pip install -r requirements\requirements_tampering.txt
```

## 🔍 What Changed?

All files have been reorganized into logical folders. The imports have been updated to work with the new structure. Your data, datasets, and saved models are all preserved in their original locations.

## ❓ Need Help?

- Check `README.md` for comprehensive documentation
- See `docs/` folder for detailed guides
- Look at `examples/` for usage examples
- Review `RESTRUCTURING_SUMMARY.md` for what changed

## 🎯 Common Tasks

### Train a Model

```bash
# Open the training notebook
jupyter notebook notebooks\Model.ipynb
```

### Generate Forgeries

```bash
python models\Forgery.py
```

### Analyze Regions

```bash
jupyter notebook notebooks\Regions.ipynb
```

### Check Model Info

```bash
python -c "from models.predict import load_model; model, device = load_model('models/saved_models/best_model.pth'); print(f'Model loaded on: {device}')"
```

---

**Happy coding! 🎉**
