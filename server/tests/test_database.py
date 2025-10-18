"""
Simple script to test database connection and setup.
Run this before starting the server to verify PostgreSQL is configured correctly.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from database.config import test_connection, init_db, SessionLocal
from database import crud

from dotenv import load_dotenv
load_dotenv()


def main():
    print("\n" + "=" * 60)
    print("DocuForge Database Setup Test")
    print("=" * 60 + "\n")
    
    # Test 1: Connection
    print("Test 1: Testing database connection...")
    if test_connection():
        print("✅ Database connection successful!\n")
    else:
        print("❌ Database connection failed!")
        print("Please check your DATABASE_URL in .env file")
        print("Example: DATABASE_URL=postgresql://postgres:postgres@localhost:5432/docuforge\n")
        return False
    
    # Test 2: Initialize tables
    print("Test 2: Initializing database tables...")
    try:
        init_db()
        print("✅ Database tables created/verified!\n")
    except Exception as e:
        print(f"❌ Error creating tables: {str(e)}\n")
        return False
    
    # Test 3: CRUD operations
    print("Test 3: Testing CRUD operations...")
    db = SessionLocal()
    try:
        # Create a test record
        test_result = crud.create_analysis_result(
            db=db,
            filename="test_image.jpg",
            prediction="authentic",
            probability=0.15,
            confidence=0.85,
            file_size=1024,
            model_version="1.0.0",
            processing_time=1.5,
            success=True
        )
        print(f"✅ Created test record with ID: {test_result.id}")
        
        # Read the record
        retrieved = crud.get_analysis_result(db, test_result.id)
        if retrieved:
            print(f"✅ Retrieved record: {retrieved.filename}")
        
        # Get statistics
        stats = crud.get_statistics(db)
        print(f"✅ Statistics: {stats['total_analyses']} total analyses")
        
        # Delete the test record
        deleted = crud.delete_analysis_result(db, test_result.id)
        if deleted:
            print(f"✅ Deleted test record")
        
        print("✅ All CRUD operations successful!\n")
    
    except Exception as e:
        print(f"❌ Error during CRUD operations: {str(e)}\n")
        return False
    finally:
        db.close()
    
    # Test 4: Display current statistics
    print("Test 4: Current database statistics...")
    db = SessionLocal()
    try:
        stats = crud.get_statistics(db)
        print("Current Statistics:")
        print(f"  - Total analyses: {stats['total_analyses']}")
        print(f"  - Authentic: {stats['authentic_count']}")
        print(f"  - Forged: {stats['forged_count']}")
        print(f"  - Successful: {stats['successful_count']}")
        print(f"  - Failed: {stats['failed_count']}")
        print(f"  - Avg confidence: {stats['average_confidence']:.4f}")
        print(f"  - Avg probability: {stats['average_probability']:.4f}")
        print(f"  - Avg processing time: {stats['average_processing_time']:.4f}s")
    except Exception as e:
        print(f"⚠️  Could not fetch statistics: {str(e)}")
    finally:
        db.close()
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Database is ready to use.")
    print("=" * 60 + "\n")
    print("You can now start the server with:")
    print("  python api/main.py")
    print("\nOr use uvicorn:")
    print("  uvicorn api.main:app --reload")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
