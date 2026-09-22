# DocuForge - Document Forgery Detection System

## � Authentication

The API now supports **Auth0 JWT authentication** to protect endpoints. When enabled, clients must include a valid Bearer token in the `Authorization` header.

**Quick Start:**

1. See `docs/AUTH0_QUICK_REFERENCE.md` for 5-minute setup
2. Full guide: `docs/AUTH0_SETUP_GUIDE.md`
3. Implementation details: `docs/AUTH0_IMPLEMENTATION_SUMMARY.md`

**Development Mode:** Run without authentication by not setting Auth0 environment variables.

## �📁 Project Structure

The codebase has been reorganized into logical folders for better maintainability:

```
server/
├── api/                    # API-related files
│   ├── main.py            # FastAPI application and endpoints
│   ├── auth.py            # Auth0 JWT authentication module (NEW)
│   └── start_server.py    # Server startup script with dependency checks
│
├── models/                # Machine learning models
│   ├── Forgery.py        # Forgery generation module
│   ├── predict.py        # Model inference and prediction functions
│   └── saved_models/     # Trained model weights
│       └── best_model.pth
│
├── detection/            # Detection modules
│   ├── tampering_localization.py  # Tampering detection and localization
│   ├── enhanced_blur_detection.py # Enhanced blur detection
│   └── regions.py                 # Region analysis
│
├── tests/               # Test files
│   ├── test_blur_detection_forgery.py
│   ├── test_blur_method.py
│   ├── test_client.py
│   ├── test_opencv_fix.py
│   └── test_tampering_module.py
│
├── examples/            # Example usage scripts
│   ├── example_usage.py
│   ├── example_tampering_usage.py
│   ├── integration_example.py
│   └── demo_blur_detection.py
│
├── notebooks/           # Jupyter notebooks
│   ├── Model.ipynb
│   └── Regions.ipynb
│
├── docs/               # Documentation
│   ├── API_README.md
│   ├── ARCHITECTURE_DIAGRAM.txt
│   ├── BLUR_DETECTION_GUIDE.md
│   ├── BLUR_ENHANCEMENT_SUMMARY.txt
│   ├── IMPLEMENTATION_SUMMARY.txt
│   ├── RUN_SERVER.md
│   ├── SETUP_GUIDE.md
│   ├── TAMPERING_DETECTION_README.md
│   ├── AUTH0_SETUP_GUIDE.md         # Auth0 complete setup guide (NEW)
│   ├── AUTH0_IMPLEMENTATION_SUMMARY.md  # Implementation details (NEW)
│   └── AUTH0_QUICK_REFERENCE.md     # Quick reference card (NEW)
│
├── scripts/            # Shell and batch scripts
│   ├── run              # Unix/Linux startup script
│   ├── run_server.bat   # Windows batch script
│   └── run_server.ps1   # PowerShell script
│
├── requirements/       # Python dependencies
│   ├── requirements_api.txt
│   └── requirements_tampering.txt
│
├── data/              # Data files
├── dataset/           # Training/validation/test datasets
├── images/            # Image assets
├── invoices/          # Invoice-specific datasets
├── letters/           # Letter-specific datasets
├── regions/           # Region analysis output
├── runs/              # Training runs
├── evaluation_results/  # Model evaluation results
├── tampering_results/   # Tampering detection results
└── test_results/        # Test output files
```

## 🚀 Getting Started

### Installation

1. Install dependencies:

```bash
pip install -r requirements/requirements_api.txt
```

For tampering detection features:

```bash
pip install -r requirements/requirements_tampering.txt
```

### Running the Server

#### Windows (PowerShell)

```powershell
.\scripts\run_server.ps1
```

#### Windows (Command Prompt)

```cmd
.\scripts\run_server.bat
```

#### Unix/Linux/Mac

```bash
./scripts/run
```

#### Direct Python

```bash
cd server
python api/main.py
```

Or use the startup script:

```bash
python api/start_server.py
```

The server will be available at:

- Main API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📖 API Usage

### Predict Document Authenticity

```python
import requests

# Upload an image for prediction
with open('document.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)
    result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
```

### Health Check

```bash
curl http://localhost:8000/health
```

## 🧪 Running Tests

Run the test client:

```bash
python tests/test_client.py path/to/image.jpg
```

Run tampering detection tests:

```bash
python tests/test_tampering_module.py
```

Run blur detection tests:

```bash
python tests/test_blur_detection_forgery.py
```

## 📚 Examples

Check the `examples/` folder for sample scripts:

- `example_usage.py` - Basic prediction usage
- `example_tampering_usage.py` - Tampering localization examples
- `integration_example.py` - Full integration example
- `demo_blur_detection.py` - Blur detection demonstration

## 📝 Documentation

All documentation is located in the `docs/` folder:

- `API_README.md` - API documentation
- `SETUP_GUIDE.md` - Setup instructions
- `RUN_SERVER.md` - Server running guide
- `TAMPERING_DETECTION_README.md` - Tampering detection guide
- `BLUR_DETECTION_GUIDE.md` - Blur detection guide

## 🔧 Development

### Project Modules

- **API Module** (`api/`): FastAPI server and endpoints
- **Models Module** (`models/`): ML models and inference
- **Detection Module** (`detection/`): Various detection algorithms
- **Tests Module** (`tests/`): Test suites
- **Examples Module** (`examples/`): Usage examples

### Import Pattern

All modules use relative imports from the server root. Example:

```python
from models.predict import predict, load_model
from detection.tampering_localization import detect_tampering_hybrid
```

## 📋 Requirements

- Python 3.8+
- PyTorch
- FastAPI
- OpenCV
- NumPy
- PIL/Pillow
- scikit-image
- scipy

See `requirements/` folder for complete dependency lists.

## 🎯 Features

- **Document Forgery Detection**: Deep learning-based binary classification
- **Tampering Localization**: Hybrid approach combining DL and classical methods
- **Blur Detection**: Specialized detection for blur-based forgeries
- **Region Analysis**: Document region segmentation and analysis
- **REST API**: Easy integration with web and mobile applications
- **Batch Processing**: Process multiple documents efficiently

---

For more information, see the documentation in the `docs/` folder.
