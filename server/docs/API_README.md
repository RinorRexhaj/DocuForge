# DocuForge FastAPI Backend

A FastAPI backend server for document forgery detection using deep learning.

## Features

- 🚀 Single endpoint for document prediction
- 📤 File upload support (JPG, PNG, BMP, TIFF)
- 🎯 Real-time forgery detection
- 📊 Returns prediction with probability and confidence scores
- 🔄 CORS enabled for frontend integration
- 📚 Auto-generated API documentation (Swagger UI)

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements_api.txt
```

The required packages are:

- `fastapi` - Modern web framework
- `uvicorn` - ASGI server
- `python-multipart` - For file upload support
- `torch` - PyTorch (already installed for your model)
- `torchvision` - Computer vision library
- `Pillow` - Image processing

### 2. Ensure Model is Available

Make sure your trained model exists at:

```
saved_models/best_model.pth
```

## Running the Server

### Start the server:

```bash
python main.py
```

Or alternatively:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`

### Server Output:

```
🚀 Starting DocuForge API Server...
📦 Loading model...
✅ Model loaded successfully!
📍 Using device: cuda

📚 Starting server...
📖 API Documentation will be available at: http://localhost:8000/docs
```

## API Endpoints

### 1. Root Endpoint - GET `/`

Health check and API information

**Response:**

```json
{
  "message": "DocuForge API is running",
  "status": "healthy",
  "model_loaded": true,
  "endpoints": {
    "predict": "/predict (POST)",
    "health": "/health (GET)",
    "docs": "/docs (GET)"
  }
}
```

### 2. Health Check - GET `/health`

Check API and model status

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### 3. Predict - POST `/predict`

Upload an image and get forgery prediction

**Request:**

- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Form data with file field named `file`

**Supported formats:** JPG, JPEG, PNG, BMP, TIFF

**Response:**

```json
{
  "prediction": "authentic",
  "probability": 0.2345,
  "confidence": 0.7655,
  "filename": "document.jpg"
}
```

**Fields:**

- `prediction`: "authentic" or "forged"
- `probability`: Probability of being forged (0-1)
- `confidence`: Confidence in the prediction (0-1)
- `filename`: Name of the uploaded file

## Testing the API

### Method 1: Using the Test Client Script

```bash
python test_client.py path/to/your/image.jpg
```

### Method 2: Using curl (Command Line)

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@path/to/image.jpg"
```

### Method 3: Using PowerShell

```powershell
$file = Get-Item "path\to\image.jpg"
$response = Invoke-WebRequest -Uri "http://localhost:8000/predict" -Method Post -Form @{file=$file}
$response.Content | ConvertFrom-Json
```

### Method 4: Using Python requests

```python
import requests

# Open and send the image
with open('document.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)
    result = response.json()
    print(result)
```

### Method 5: Using the Interactive API Documentation

1. Open your browser and navigate to `http://localhost:8000/docs`
2. Click on the `/predict` endpoint
3. Click "Try it out"
4. Upload a file using the file picker
5. Click "Execute"
6. View the response

## Example Response

### Authentic Document:

```json
{
  "prediction": "authentic",
  "probability": 0.1234,
  "confidence": 0.8766,
  "filename": "letter.jpg"
}
```

### Forged Document:

```json
{
  "prediction": "forged",
  "probability": 0.8923,
  "confidence": 0.8923,
  "filename": "fake_invoice.jpg"
}
```

## Error Handling

### Invalid File Type (400):

```json
{
  "detail": "Invalid file type. Allowed types: .jpg, .jpeg, .png, .bmp, .tiff, .tif"
}
```

### Model Not Loaded (503):

```json
{
  "detail": "Model not loaded. Please restart the server."
}
```

### Prediction Error (500):

```json
{
  "detail": "Prediction error: [error message]"
}
```

## Production Considerations

### 1. CORS Configuration

In production, modify the CORS settings in `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specify your frontend domain
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### 2. Security

- Add authentication/authorization
- Implement rate limiting
- Add file size limits
- Validate file content (not just extension)
- Use HTTPS in production

### 3. Performance

- Consider using async file operations
- Add caching for repeated predictions
- Use a production ASGI server like Gunicorn with Uvicorn workers
- Deploy behind a reverse proxy (Nginx)

### 4. Monitoring

- Add logging
- Implement health checks
- Track prediction metrics
- Monitor server resources

## Troubleshooting

### Server won't start

- Check if port 8000 is already in use
- Verify all dependencies are installed
- Ensure the model file exists

### Model not loading

- Check the model path: `saved_models/best_model.pth`
- Verify the model file is not corrupted
- Check CUDA/GPU availability if using GPU

### Predictions failing

- Verify uploaded image is valid
- Check image format is supported
- Review server logs for errors

## Architecture

```
Client (Browser/App)
    ↓ (HTTP POST with file)
FastAPI Server (main.py)
    ↓ (saves temp file)
Predict Module (predict.py)
    ↓ (loads and processes)
Deep Learning Model (best_model.pth)
    ↓ (returns prediction)
JSON Response to Client
```

## License

[Your License Here]

## Contact

[Your Contact Information]
