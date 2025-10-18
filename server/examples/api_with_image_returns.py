"""
FastAPI Endpoint Example - Tampering Detection with Image Returns
=================================================================
This example shows how to create a complete API endpoint that:
1. Accepts an uploaded image
2. Runs tampering detection
3. Returns all images as base64 strings in JSON
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path
import tempfile
import os
import shutil

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.predict import load_model
from detection.tampering_localization import detect_tampering_hybrid

# Initialize FastAPI app
app = FastAPI(
    title="DocuForge Tampering Detection API",
    description="API for detecting and localizing document tampering",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
device = None

# Allowed image extensions
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


@app.on_event("startup")
async def startup_event():
    """Load the model when the server starts up."""
    global model, device
    print("\n🚀 Starting Tampering Detection API Server...")
    print("📦 Loading model...")
    
    try:
        model_path = Path(__file__).parent.parent / 'models' / 'saved_models' / 'best_model.pth'
        model, device = load_model(str(model_path))
        print("✅ Model loaded successfully!")
        print(f"📍 Using device: {device}\n")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print("⚠️  Server will start but predictions will fail!\n")


@app.get("/")
async def root():
    """Root endpoint - health check and API information."""
    return {
        "message": "DocuForge Tampering Detection API is running",
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "2.0.0",
        "endpoints": {
            "detect_tampering": "/detect-tampering (POST)",
            "detect_tampering_detailed": "/detect-tampering-detailed (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API and model status."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not loaded"
    }


@app.post("/detect-tampering")
async def detect_tampering(
    file: UploadFile = File(...),
    sensitivity: float = 0.5
):
    """
    Upload an image and get tampering detection results with images.
    
    Args:
        file: Image file (jpg, jpeg, png, bmp, tiff)
        sensitivity: Detection sensitivity (0.0 - 1.0, default 0.5)
    
    Returns:
        JSON response with:
        - heatmap: Base64-encoded heatmap image
        - mask: Base64-encoded binary mask image
        - tampered_regions: Base64-encoded image with bounding boxes
        - probability: Tampering confidence score (0-1)
        - bboxes: List of detected region coordinates
        - num_regions: Number of suspicious regions found
        - filename: Original filename
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
    
    # Validate sensitivity
    if not 0 <= sensitivity <= 1:
        raise HTTPException(
            status_code=400,
            detail="Sensitivity must be between 0 and 1"
        )
    
    # Create a temporary file to save the upload
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        # Run tampering detection with base64 output
        result = detect_tampering_hybrid(
            image_path=temp_file_path,
            model=model,
            device=device,
            save_results=False,        # Don't save to disk
            sensitivity=sensitivity,
            return_base64=True,        # Return images as base64
            return_intermediate_maps=False
        )
        
        # Enhance response with additional metadata
        response = {
            'filename': file.filename,
            'num_regions': len(result['bboxes']),
            'is_tampered': result['probability'] > 0.5,
            'heatmap': result['heatmap'],
            'mask': result['mask'],
            'tampered_regions': result['tampered_regions'],
            'probability': result['probability'],
            'bboxes': result['bboxes']
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Detection error: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        try:
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)
        except Exception as e:
            print(f"Warning: Could not delete temporary file: {e}")


@app.post("/detect-tampering-detailed")
async def detect_tampering_detailed(
    file: UploadFile = File(...),
    sensitivity: float = 0.5
):
    """
    Upload an image and get detailed tampering detection results.
    Includes all intermediate forensic maps.
    
    Args:
        file: Image file (jpg, jpeg, png, bmp, tiff)
        sensitivity: Detection sensitivity (0.0 - 1.0, default 0.5)
    
    Returns:
        JSON response with:
        - All basic detection results
        - intermediate_maps: Individual forensic detection maps
        - gradcam: GradCAM visualization
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        # Run tampering detection with ALL outputs
        result = detect_tampering_hybrid(
            image_path=temp_file_path,
            model=model,
            device=device,
            save_results=False,
            sensitivity=sensitivity,
            return_base64=True,
            return_intermediate_maps=True  # Include all forensic maps
        )
        
        # Enhance response
        response = {
            'filename': file.filename,
            'num_regions': len(result['bboxes']),
            'is_tampered': result['probability'] > 0.5,
            'heatmap': result['heatmap'],
            'mask': result['mask'],
            'tampered_regions': result['tampered_regions'],
            'probability': result['probability'],
            'bboxes': result['bboxes'],
            'intermediate_maps': result.get('intermediate_maps', {}),
            'gradcam': result.get('gradcam', None)
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Detection error: {str(e)}"
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
    print("DocuForge Tampering Detection API v2.0")
    print("=" * 60)
    print("\n📚 Starting server...")
    print("📖 API Documentation: http://localhost:8001/docs")
    print("🔗 Root endpoint: http://localhost:8001")
    print("🎯 Basic detection: http://localhost:8001/detect-tampering (POST)")
    print("🔬 Detailed detection: http://localhost:8001/detect-tampering-detailed (POST)")
    print("\n💡 All images returned as base64 strings in JSON!")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
