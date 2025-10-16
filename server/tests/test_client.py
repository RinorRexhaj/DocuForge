"""
Simple test client for the DocuForge API
This script demonstrates how to send requests to the API endpoint
"""
import requests
import sys


def test_prediction(image_path, api_url="http://localhost:8000/predict"):
    """
    Send an image to the API for prediction.
    
    Args:
        image_path (str): Path to the image file
        api_url (str): URL of the prediction endpoint
    
    Returns:
        dict: Prediction results
    """
    try:
        # Open the image file
        with open(image_path, 'rb') as f:
            files = {'file': (image_path, f, 'image/jpeg')}
            
            # Send POST request
            response = requests.post(api_url, files=files)
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Details: {response.json()}")
                return None
    
    except FileNotFoundError:
        print(f"❌ Error: Image file not found: {image_path}")
        return None
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API. Is the server running?")
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def print_result(result):
    """Pretty print the prediction result."""
    if result:
        print("\n" + "=" * 60)
        print("📊 PREDICTION RESULTS")
        print("=" * 60)
        print(f"📄 Filename: {result['filename']}")
        print(f"🎯 Prediction: {result['prediction'].upper()}")
        print(f"📈 Probability (forged): {result['probability']:.4f} ({result['probability']*100:.2f}%)")
        print(f"✨ Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <path_to_image>")
        print("Example: python test_client.py test_document.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("🚀 Testing DocuForge API...")
    print(f"📤 Sending image: {image_path}")
    
    # Test the prediction endpoint
    result = test_prediction(image_path)
    
    # Print results
    print_result(result)
