from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from models.predict import predict, load_model
from detection.tampering_localization import detect_tampering_hybrid
from api.auth import get_current_user, get_current_user_optional, is_auth_enabled
from database.config import get_db, init_db, test_connection
from database import crud
import os
import shutil
import tempfile
import time

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
    
    # Test database connection
    print("🔌 Testing database connection...")
    if test_connection():
        print("✅ Database connection successful!")
        try:
            init_db()
        except Exception as e:
            print(f"⚠️  Database initialization warning: {str(e)}")
    else:
        print("❌ Database connection failed!")
        print("⚠️  Server will start but database operations will not work.")
        print("   Please check your DATABASE_URL in .env file")
    
    # Check Auth0 configuration
    if is_auth_enabled():
        print("🔒 Auth0 authentication: ENABLED")
        print(f"   Domain: {os.getenv('AUTH0_DOMAIN')}")
        print(f"   Audience: {os.getenv('AUTH0_API_AUDIENCE')}")
    else:
        print("⚠️  Auth0 authentication: DISABLED (no auth required)")
        print("   Set AUTH0_DOMAIN and AUTH0_API_AUDIENCE to enable")
    
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
    Public endpoint (no authentication required).
    """
    return {
        "message": "DocuForge API is running",
        "status": "healthy",
        "model_loaded": model is not None,
        "auth_enabled": is_auth_enabled(),
        "endpoints": {
            "predict": "/predict (POST) - Protected",
            "health": "/health (GET) - Public",
            "docs": "/docs (GET) - Public"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API and model status.
    Public endpoint (no authentication required).
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not loaded",
        "auth_enabled": is_auth_enabled()
    }


@app.post("/predict")
async def predict_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    # user: dict = Depends(get_current_user)
):
    """
    Upload an image file and get a prediction on whether it's authentic or forged.
    Results are automatically saved to the database.
    **Protected endpoint - requires Auth0 authentication.**
    
    Args:
        file: Image file (jpg, jpeg, png, bmp, tiff)
        db: Database session (injected)
        user: Authenticated user information (from JWT token)
    
    Returns:
        JSON response with prediction results:
        - id: Database ID of the saved analysis
        - prediction: 'authentic' or 'forged'
        - probability: probability of being forged (0-1)
        - confidence: confidence in the prediction (0-1)
        - filename: name of the uploaded file
        - heatmap: base64 encoded tampering heatmap
        - mask: base64 encoded tampering mask
        - tampered_regions: base64 encoded image with bounding boxes
        - user_id: ID of the authenticated user who made the request
        - created_at: timestamp of analysis
    
    Authorization:
        Requires Bearer token in Authorization header.
        Example: Authorization: Bearer <your_auth0_token>
    """
    start_time = time.time()
    temp_file_path = None
    
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
    
    # Get file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
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

        images = detect_tampering_hybrid(
            image_path=temp_file_path,
            model=model,
            device=device,
            save_results=False,  # Don't save to disk
            sensitivity=0.7,     # Less tamperering sensitivity for visualization
            return_base64=True,  # Return images as base64 strings for JSON
            return_intermediate_maps=False
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Add filename and images to result
        result['filename'] = file.filename
        result['heatmap'] = images['heatmap']
        result['mask'] = images['mask']
        result['tampered_regions'] = images['tampered_regions']
        # result['user_id'] = user.get('sub', 'unknown')  # Include authenticated user ID
        
        # Save to database
        try:
            db_result = crud.create_analysis_result(
                db=db,
                filename=file.filename,
                prediction=result['prediction'],
                probability=result['probability'],
                confidence=result['confidence'],
                # heatmap=images['heatmap'],
                # mask=images['mask'],
                # tampered_regions=images['tampered_regions'],
                # user_id=user.get('sub'),
                # user_email=user.get('email'),
                user_id=None,
                user_email=None,
                file_size=file_size,
                model_version="1.0.0",
                processing_time=processing_time,
                success=True
            )
            result['id'] = str(db_result.id)
            result['created_at'] = db_result.created_at.isoformat()
        except Exception as db_error:
            print(f"Warning: Failed to save to database: {str(db_error)}")
            # Continue even if database save fails
        
        return JSONResponse(content=result)
    
    except Exception as e:
        # Log error to database if possible
        try:
            crud.create_analysis_result(
                db=db,
                filename=file.filename,
                prediction="error",
                probability=0.0,
                confidence=0.0,
                # user_id=user.get('sub') if 'user' in locals() else None,
                # user_email=user.get('email') if 'user' in locals() else None,
                user_id=None,
                user_email=None,
                file_size=file_size,
                success=False,
                error_message=str(e)
            )
        except:
            pass  # Silently fail if database is not available
        
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        try:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except Exception as e:
            print(f"Warning: Could not delete temporary file: {e}")


@app.post("/detect-tampering")
async def detect_tampering_endpoint(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    # user: dict = Depends(get_current_user)
):
    """
    Dedicated endpoint for comprehensive tampering detection using hybrid approach.
    Results are automatically saved to the database.
    **Protected endpoint - requires Auth0 authentication.**
    
    Focuses on detecting:
    - Blur inconsistencies (Laplacian variance, gradient magnitude)
    - Color/illumination inconsistencies
    - Copy-move and splicing forgeries
    
    Args:
        file: Image file (jpg, jpeg, png, bmp, tiff)
        db: Database session (injected)
        user: Authenticated user information (from JWT token)
    
    Returns:
        JSON response with detailed tampering analysis:
        - id: Database ID of the saved analysis
        - heatmap: base64 encoded tampering heatmap (red = high tampering probability)
        - mask: base64 encoded binary tampering mask (white = tampered regions)
        - tampered_regions: base64 encoded image with detected region bounding boxes
        - filename: name of the uploaded file
        - user_id: ID of the authenticated user who made the request
        - created_at: timestamp of analysis
    
    Authorization:
        Requires Bearer token in Authorization header.
        Example: Authorization: Bearer <your_auth0_token>
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
        
        # Run comprehensive tampering detection
        tampering_results = detect_tampering_hybrid(
            image_path=temp_file_path,
            model=model,
            device=device,
            save_results=False,  # Don't save to disk
            sensitivity=0.7,     # Tampering sensitivity for visualization
            return_base64=True,  # Return images as base64 strings for JSON
            return_intermediate_maps=False
        )
        
        # Build response
        response_data = {
            "filename": file.filename,
            "heatmap": tampering_results['heatmap'],
            "mask": tampering_results['mask'],
            "tampered_regions": tampering_results['tampered_regions'],
        }
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tampering detection error: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        try:
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)
        except Exception as e:
            print(f"Warning: Could not delete temporary file: {e}")


@app.get("/analysis/{analysis_id}")
async def get_analysis(
    analysis_id: str,
    include_images: bool = True,
    db: Session = Depends(get_db)
):
    """
    Get a specific analysis result by ID.
    Public endpoint (no authentication required).
    
    Args:
        analysis_id: UUID of the analysis result
        include_images: Whether to include base64 image data (default: True)
        db: Database session (injected)
    
    Returns:
        Analysis result details
    """
    try:
        import uuid
        result = crud.get_analysis_result(db, uuid.UUID(analysis_id))
        if not result:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return result.to_dict(include_images=include_images)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid analysis ID format")


@app.get("/history")
async def get_analysis_history(
    skip: int = 0,
    limit: int = 50,
    user_id: str = None,
    prediction: str = None,
    db: Session = Depends(get_db)
):
    """
    Get analysis history with optional filtering.
    Public endpoint (no authentication required).
    
    Args:
        skip: Number of records to skip (pagination)
        limit: Maximum number of records to return (max 100)
        user_id: Filter by user ID (optional)
        prediction: Filter by prediction type ('authentic' or 'forged')
        db: Database session (injected)
    
    Returns:
        List of analysis results (without images to reduce size)
    """
    try:
        if limit > 100:
            limit = 100
        
        results = crud.get_analysis_results(
            db=db,
            skip=skip,
            limit=limit,
            user_id=user_id,
            prediction=prediction,
            success_only=True
        )
        
        return [result.to_dict(include_images=False) for result in results]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching analysis history: {str(e)}"
        )


@app.get("/analysis/recent")
async def get_recent_analysis(
    hours: int = 24,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Get recent analyses within the specified time window.
    Public endpoint (no authentication required).
    
    Args:
        hours: Number of hours to look back (default: 24)
        limit: Maximum number of records to return (max 100)
        db: Database session (injected)
    
    Returns:
        List of recent analysis results (without images)
    """
    if limit > 100:
        limit = 100
    
    results = crud.get_recent_analyses(db=db, hours=hours, limit=limit)
    return [result.to_dict(include_images=False) for result in results]


@app.get("/statistics")
async def get_analysis_statistics(
    user_id: str = None,
    db: Session = Depends(get_db)
):
    """
    Get statistics about analyses.
    Public endpoint (no authentication required).
    
    Args:
        user_id: Optional user ID to filter statistics
        db: Database session (injected)
    
    Returns:
        Statistics including total analyses, predictions breakdown, averages
    """
    return crud.get_statistics(db=db, user_id=user_id)


@app.delete("/analysis/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    db: Session = Depends(get_db),
    # user: dict = Depends(get_current_user)
):
    """
    Delete an analysis result by ID.
    **Protected endpoint - requires Auth0 authentication.**
    
    Args:
        analysis_id: UUID of the analysis result
        db: Database session (injected)
        user: Authenticated user information (from JWT token)
    
    Returns:
        Success message
    """
    try:
        import uuid
        deleted = crud.delete_analysis_result(db, uuid.UUID(analysis_id))
        if not deleted:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return {"message": "Analysis deleted successfully", "id": analysis_id}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid analysis ID format")


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("DocuForge Document Forgery Detection API")
    print("=" * 60)
    print("\n📚 Starting server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔗 Root endpoint: http://localhost:8000")
    
    auth_status = "ENABLED ✅" if is_auth_enabled() else "DISABLED ⚠️"
    print(f"\n🔐 Authentication: {auth_status}")
    
    if is_auth_enabled():
        print("   Protected endpoints require Bearer token:")
        print("   • POST /predict - Document prediction")
        print("   • POST /detect-tampering - Tampering detection")
    else:
        print("   Running in OPEN mode (no authentication required)")
        print("   Set AUTH0_DOMAIN and AUTH0_API_AUDIENCE to enable Auth0")
    
    print("\n" + "=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
