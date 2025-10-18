"""
Database models for storing analysis results.
"""
from sqlalchemy import Column, String, Float, DateTime, Text, Boolean, Integer
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

from database.config import Base


class AnalysisResult(Base):
    """
    Table for storing document forgery analysis results.
    
    Stores comprehensive information about each analysis including:
    - Prediction results (authentic/forged)
    - Probabilities and confidence scores
    - Tampering detection results
    - Base64 encoded images (heatmap, mask, regions)
    - User information (if auth is enabled)
    - Metadata (filename, timestamp)
    """
    __tablename__ = "analysis_results"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # File information
    filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=True)  # File size in bytes
    
    # User information (optional - only if auth is enabled)
    user_id = Column(String(255), nullable=True, index=True)
    user_email = Column(String(255), nullable=True)
    
    # Prediction results
    prediction = Column(String(50), nullable=False)  # 'authentic' or 'forged'
    probability = Column(Float, nullable=False)  # Probability of being forged (0-1)
    confidence = Column(Float, nullable=False)  # Confidence in prediction (0-1)
    
    # Tampering detection results (base64 encoded images)
    heatmap = Column(Text, nullable=True)  # Base64 encoded heatmap
    mask = Column(Text, nullable=True)  # Base64 encoded mask
    tampered_regions = Column(Text, nullable=True)  # Base64 encoded regions image
    
    # Additional metadata
    model_version = Column(String(50), nullable=True)  # Track which model version was used
    processing_time = Column(Float, nullable=True)  # Time taken for analysis in seconds
    error_message = Column(Text, nullable=True)  # Store error if analysis failed
    success = Column(Boolean, default=True, nullable=False)  # Whether analysis succeeded
    
    def __repr__(self):
        return f"<AnalysisResult(id={self.id}, filename={self.filename}, prediction={self.prediction})>"
    
    def to_dict(self, include_images=True):
        """
        Convert model to dictionary.
        
        Args:
            include_images: Whether to include base64 image data (can be large)
        
        Returns:
            Dictionary representation of the analysis result
        """
        data = {
            "id": str(self.id),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "filename": self.filename,
            "file_size": self.file_size,
            "user_id": self.user_id,
            "user_email": self.user_email,
            "prediction": self.prediction,
            "probability": self.probability,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "processing_time": self.processing_time,
            "error_message": self.error_message,
            "success": self.success
        }
        
        if include_images:
            data.update({
                "heatmap": self.heatmap,
                "mask": self.mask,
                "tampered_regions": self.tampered_regions
            })
        
        return data
