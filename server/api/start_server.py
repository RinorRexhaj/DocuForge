#!/usr/bin/env python3
"""
DocuForge API Server Startup Script
This script checks dependencies and starts the FastAPI server
"""
import sys
import os
import subprocess
from pathlib import Path


def print_header():
    """Print the startup header."""
    print("\n" + "=" * 60)
    print("  DocuForge - Document Forgery Detection API Server")
    print("=" * 60 + "\n")


def check_file_exists(filepath, file_type="file"):
    """Check if a file exists."""
    if not os.path.exists(filepath):
        print(f"❌ [ERROR] {file_type} not found: {filepath}")
        return False
    print(f"✅ [OK] {file_type} found: {filepath}")
    return True


def check_dependencies():
    """Check if required Python packages are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = ['fastapi', 'uvicorn', 'torch', 'torchvision', 'PIL']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  [WARNING] Missing packages: {', '.join(missing_packages)}")
        print("   Installing dependencies from requirements_api.txt...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "-r", "requirements/requirements_api.txt"
            ])
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ [ERROR] Failed to install dependencies!")
            print("   Please manually run: pip install -r requirements/requirements_api.txt")
            return False
    
    return True


def start_server():
    """Start the FastAPI server."""
    print("\n" + "=" * 60)
    print("🚀 Starting FastAPI server...")
    print("=" * 60)
    print("\n📍 Server URLs:")
    print("   • Main API: http://localhost:8000")
    print("   • API Docs: http://localhost:8000/docs")
    print("   • ReDoc: http://localhost:8000/redoc")
    print("\n⚠️  Press CTRL+C to stop the server")
    print("=" * 60 + "\n")
    
    try:
        # Import and run the server
        import uvicorn
        
        # Run the API main module
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ [ERROR] Failed to start server: {str(e)}")
        sys.exit(1)


def main():
    """Main function to run all checks and start the server."""
    print_header()
    
    # Get the server directory (parent of api folder)
    server_dir = Path(__file__).parent.parent
    os.chdir(server_dir)
    
    # Check if we're in the right directory
    if not check_file_exists("api/main.py", "Main server file"):
        print("\n⚠️  Please run this script from the DocuForge/server directory.")
        sys.exit(1)
    
    # Check if model exists
    model_exists = check_file_exists("models/saved_models/best_model.pth", "Model file")
    if not model_exists:
        print("⚠️  [WARNING] Server will start but predictions will fail!")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Start the server
    start_server()


if __name__ == "__main__":
    main()
