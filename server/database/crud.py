"""
CRUD operations for analysis results.
"""
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List, Optional
from datetime import datetime, timedelta
import uuid

from database.models import AnalysisResult


def create_analysis_result(
    db: Session,
    filename: str,
    prediction: str,
    probability: float,
    confidence: float,
    heatmap: Optional[str] = None,
    mask: Optional[str] = None,
    tampered_regions: Optional[str] = None,
    user_id: Optional[str] = None,
    user_email: Optional[str] = None,
    file_size: Optional[int] = None,
    model_version: Optional[str] = None,
    processing_time: Optional[float] = None,
    success: bool = True,
    error_message: Optional[str] = None
) -> AnalysisResult:
    """
    Create a new analysis result record.
    
    Args:
        db: Database session
        filename: Name of the analyzed file
        prediction: 'authentic' or 'forged'
        probability: Probability of being forged (0-1)
        confidence: Confidence in prediction (0-1)
        heatmap: Base64 encoded heatmap image
        mask: Base64 encoded mask image
        tampered_regions: Base64 encoded regions image
        user_id: User ID (if authenticated)
        user_email: User email (if authenticated)
        file_size: File size in bytes
        model_version: Version of the model used
        processing_time: Time taken for analysis
        success: Whether analysis succeeded
        error_message: Error message if analysis failed
    
    Returns:
        Created AnalysisResult object
    """
    result = AnalysisResult(
        filename=filename,
        prediction=prediction,
        probability=probability,
        confidence=confidence,
        heatmap=heatmap,
        mask=mask,
        tampered_regions=tampered_regions,
        user_id=user_id,
        user_email=user_email,
        file_size=file_size,
        model_version=model_version,
        processing_time=processing_time,
        success=success,
        error_message=error_message
    )
    
    db.add(result)
    db.commit()
    db.refresh(result)
    return result


def get_analysis_result(db: Session, result_id: uuid.UUID) -> Optional[AnalysisResult]:
    """
    Get a single analysis result by ID.
    
    Args:
        db: Database session
        result_id: UUID of the analysis result
    
    Returns:
        AnalysisResult object or None if not found
    """
    return db.query(AnalysisResult).filter(AnalysisResult.id == result_id).first()


def get_analysis_results(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    user_id: Optional[str] = None,
    prediction: Optional[str] = None,
    success_only: bool = False
) -> List[AnalysisResult]:
    """
    Get multiple analysis results with optional filtering.
    
    Args:
        db: Database session
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return
        user_id: Filter by user ID (optional)
        prediction: Filter by prediction type ('authentic' or 'forged')
        success_only: Only return successful analyses
    
    Returns:
        List of AnalysisResult objects
    """
    query = db.query(AnalysisResult)
    
    if user_id:
        query = query.filter(AnalysisResult.user_id == user_id)
    
    if prediction:
        query = query.filter(AnalysisResult.prediction == prediction)
    
    if success_only:
        query = query.filter(AnalysisResult.success == True)
    
    return query.order_by(desc(AnalysisResult.created_at)).offset(skip).limit(limit).all()


def get_user_analysis_count(db: Session, user_id: str) -> int:
    """
    Get the total number of analyses for a specific user.
    
    Args:
        db: Database session
        user_id: User ID
    
    Returns:
        Count of analyses
    """
    return db.query(AnalysisResult).filter(AnalysisResult.user_id == user_id).count()


def get_recent_analyses(db: Session, hours: int = 24, limit: int = 50) -> List[AnalysisResult]:
    """
    Get recent analyses within the specified time window.
    
    Args:
        db: Database session
        hours: Number of hours to look back
        limit: Maximum number of records to return
    
    Returns:
        List of recent AnalysisResult objects
    """
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    return (
        db.query(AnalysisResult)
        .filter(AnalysisResult.created_at >= cutoff_time)
        .order_by(desc(AnalysisResult.created_at))
        .limit(limit)
        .all()
    )


def get_statistics(db: Session, user_id: Optional[str] = None) -> dict:
    """
    Get statistics about analyses.
    
    Args:
        db: Database session
        user_id: Optional user ID to filter statistics
    
    Returns:
        Dictionary with statistics
    """
    query = db.query(AnalysisResult)
    
    if user_id:
        query = query.filter(AnalysisResult.user_id == user_id)
    
    total = query.count()
    authentic = query.filter(AnalysisResult.prediction == "authentic").count()
    forged = query.filter(AnalysisResult.prediction == "forged").count()
    successful = query.filter(AnalysisResult.success == True).count()
    failed = query.filter(AnalysisResult.success == False).count()
    
    # Average confidence and probability
    avg_confidence = query.filter(AnalysisResult.success == True).with_entities(
        func.avg(AnalysisResult.confidence)
    ).scalar() or 0
    
    avg_probability = query.filter(AnalysisResult.success == True).with_entities(
        func.avg(AnalysisResult.probability)
    ).scalar() or 0
    
    # Average processing time
    avg_processing_time = query.filter(AnalysisResult.processing_time.isnot(None)).with_entities(
        func.avg(AnalysisResult.processing_time)
    ).scalar() or 0
    
    return {
        "total_analyses": total,
        "authentic_count": authentic,
        "forged_count": forged,
        "successful_count": successful,
        "failed_count": failed,
        "average_confidence": round(float(avg_confidence), 4) if avg_confidence else 0,
        "average_probability": round(float(avg_probability), 4) if avg_probability else 0,
        "average_processing_time": round(float(avg_processing_time), 4) if avg_processing_time else 0
    }


def delete_analysis_result(db: Session, result_id: uuid.UUID) -> bool:
    """
    Delete an analysis result by ID.
    
    Args:
        db: Database session
        result_id: UUID of the analysis result
    
    Returns:
        True if deleted, False if not found
    """
    result = db.query(AnalysisResult).filter(AnalysisResult.id == result_id).first()
    if result:
        db.delete(result)
        db.commit()
        return True
    return False


def delete_old_results(db: Session, days: int = 30) -> int:
    """
    Delete analysis results older than specified days.
    Useful for data cleanup.
    
    Args:
        db: Database session
        days: Number of days to keep (delete older)
    
    Returns:
        Number of deleted records
    """
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    deleted = db.query(AnalysisResult).filter(
        AnalysisResult.created_at < cutoff_date
    ).delete()
    db.commit()
    return deleted
