from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.predict import predict, load_model
import os
import shutil
import tempfile

# Initialize FastAPI app
app = FastAPI(
    title="DocuForge Document Forgery Detection API",
    description="API for detecting forged documents using deep learning",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable - loaded once at startup
model = None
device = None

# Allowed image extensions
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


@app.on_event("startup")
async def startup_event():
    """
    Load the model when the server starts up.
    This ensures the model is loaded only once and reused for all predictions.
    """
    global model, device
    print("\n🚀 Starting DocuForge API Server...")
    print("📦 Loading model...")
    
    try:
        # Update path to reflect new models folder structure
        model_path = Path(__file__).parent.parent / 'models' / 'saved_models' / 'best_model.pth'
        model, device = load_model(str(model_path))
        print("✅ Model loaded successfully!")
        print(f"📍 Using device: {device}\n")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print("⚠️  Server will start but predictions will fail!\n")


@app.get("/")
async def root():
    """
    Root endpoint - health check and API information.
    """
    return {
        "message": "DocuForge API is running",
        "status": "healthy",
        "model_loaded": model is not None,
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API and model status.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not loaded"
    }


@app.post("/predict")
async def predict_document(file: UploadFile = File(...)):
    """
    Upload an image file and get a prediction on whether it's authentic or forged.
    
    Args:
        file: Image file (jpg, jpeg, png, bmp, tiff)
    
    Returns:
        JSON response with prediction results:
        - prediction: 'authentic' or 'forged'
        - probability: probability of being forged (0-1)
        - confidence: confidence in the prediction (0-1)
        - filename: name of the uploaded file
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please restart the server."
        )
    
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Create a temporary file to save the upload
    try:
        # Create temporary file with the same extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            # Save uploaded file to temporary location
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        # Make prediction
        result = predict(
            image_path=temp_file_path,
            model=model,
            threshold=0.5,
            return_probability=True
        )
        
        # Add filename to result
        result['filename'] = file.filename
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        try:
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)
        except Exception as e:
            print(f"Warning: Could not delete temporary file: {e}")


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("DocuForge Document Forgery Detection API")
    print("=" * 60)
    print("\n📚 Starting server...")
    print("📖 API Documentation will be available at: http://localhost:8000/docs")
    print("🔗 Root endpoint: http://localhost:8000")
    print("🎯 Prediction endpoint: http://localhost:8000/predict (POST)")
    print("\n" + "=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
