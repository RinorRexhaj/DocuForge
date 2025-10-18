"""
Client-Side Example - Using the Tampering Detection API
=======================================================
This example shows how to:
1. Call the tampering detection API
2. Receive base64-encoded images
3. Display them in a web interface
"""

import requests
import base64
from pathlib import Path
import json


def call_tampering_api_python(image_path, api_url="http://localhost:8001", sensitivity=0.5):
    """
    Call the tampering detection API from Python.
    
    Args:
        image_path: Path to image file
        api_url: API base URL
        sensitivity: Detection sensitivity (0-1)
    
    Returns:
        Dictionary with detection results and base64 images
    """
    print(f"\n🔍 Analyzing {image_path}...")
    
    # Prepare the file
    with open(image_path, 'rb') as f:
        files = {'file': (Path(image_path).name, f, 'image/jpeg')}
        params = {'sensitivity': sensitivity}
        
        # Make API request
        response = requests.post(
            f"{api_url}/detect-tampering",
            files=files,
            params=params
        )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Detection completed!")
        print(f"   Tampering probability: {result['probability']:.2%}")
        print(f"   Suspicious regions: {result['num_regions']}")
        print(f"   Is tampered: {result['is_tampered']}")
        
        return result
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None


def save_base64_images(result, output_dir="api_results"):
    """
    Save base64-encoded images to disk.
    
    Args:
        result: API response dictionary
        output_dir: Directory to save images
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = result['filename'].rsplit('.', 1)[0]
    
    # Save each image
    images = {
        'heatmap': result['heatmap'],
        'mask': result['mask'],
        'tampered_regions': result['tampered_regions']
    }
    
    for name, b64_data in images.items():
        # Decode base64 to bytes
        img_bytes = base64.b64decode(b64_data)
        
        # Save to file
        output_path = Path(output_dir) / f"{base_name}_{name}.png"
        with open(output_path, 'wb') as f:
            f.write(img_bytes)
        
        print(f"✓ Saved: {output_path}")


def display_base64_image_cv2(b64_string, window_name="Image"):
    """
    Display a base64-encoded image using OpenCV.
    
    Args:
        b64_string: Base64-encoded image string
        window_name: Window name for display
    """
    import cv2
    import numpy as np
    
    # Decode base64 to bytes
    img_bytes = base64.b64decode(b64_string)
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(img_bytes, np.uint8)
    
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Display
    cv2.imshow(window_name, img)


def display_results_cv2(result):
    """
    Display all detection results using OpenCV.
    
    Args:
        result: API response dictionary
    """
    import cv2
    
    print("\n📺 Displaying results (press any key to close)...")
    
    display_base64_image_cv2(result['heatmap'], "Tampering Heatmap")
    display_base64_image_cv2(result['mask'], "Tampering Mask")
    display_base64_image_cv2(result['tampered_regions'], "Detected Regions")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_html_report(result, output_file="tampering_report.html"):
    """
    Generate an HTML report with embedded base64 images.
    
    Args:
        result: API response dictionary
        output_file: Output HTML file path
    """
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tampering Detection Report - {result['filename']}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .summary {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric {{
                display: inline-block;
                margin: 10px 20px 10px 0;
            }}
            .metric-label {{
                font-size: 14px;
                color: #666;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #333;
            }}
            .tampered {{
                color: #e74c3c;
            }}
            .authentic {{
                color: #27ae60;
            }}
            .images-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 20px;
                margin-top: 30px;
            }}
            .image-card {{
                background: white;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .image-card h3 {{
                margin-top: 0;
                color: #333;
            }}
            .image-card img {{
                width: 100%;
                border-radius: 5px;
                border: 1px solid #ddd;
            }}
            .bboxes {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-top: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 Document Tampering Detection Report</h1>
            <p>Analysis of: <strong>{result['filename']}</strong></p>
        </div>
        
        <div class="summary">
            <h2>Detection Summary</h2>
            <div class="metric">
                <div class="metric-label">Status</div>
                <div class="metric-value {'tampered' if result['is_tampered'] else 'authentic'}">
                    {'⚠️ TAMPERED' if result['is_tampered'] else '✅ AUTHENTIC'}
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">Tampering Probability</div>
                <div class="metric-value">{result['probability']:.1%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Suspicious Regions</div>
                <div class="metric-value">{result['num_regions']}</div>
            </div>
            
            {f'''
            <div class="bboxes">
                <h4>Detected Region Coordinates:</h4>
                <ul>
                    {''.join([f'<li>Region {i+1}: x={x}, y={y}, width={w}, height={h}</li>' 
                             for i, (x, y, w, h) in enumerate(result['bboxes'])])}
                </ul>
            </div>
            ''' if result['bboxes'] else ''}
        </div>
        
        <div class="images-grid">
            <div class="image-card">
                <h3>🌡️ Tampering Heatmap</h3>
                <img src="data:image/png;base64,{result['heatmap']}" alt="Heatmap">
                <p>Red/yellow areas indicate high probability of tampering.</p>
            </div>
            
            <div class="image-card">
                <h3>🎭 Binary Mask</h3>
                <img src="data:image/png;base64,{result['mask']}" alt="Mask">
                <p>White regions show detected tampering areas.</p>
            </div>
            
            <div class="image-card">
                <h3>📍 Detected Regions</h3>
                <img src="data:image/png;base64,{result['tampered_regions']}" alt="Regions">
                <p>Original image with bounding boxes around suspicious areas.</p>
            </div>
        </div>
        
        <div style="margin-top: 30px; text-align: center; color: #666;">
            <p>Generated by DocuForge Tampering Detection API v2.0</p>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_template)
    
    print(f"✓ HTML report saved: {output_file}")
    return output_file


# ============================================================================
# JAVASCRIPT EXAMPLE (for frontend developers)
# ============================================================================

JAVASCRIPT_EXAMPLE = """
// JavaScript/React Example - Calling the API from Frontend

// 1. Upload and detect tampering
async function detectTampering(imageFile, sensitivity = 0.5) {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    const response = await fetch(
        `http://localhost:8001/detect-tampering?sensitivity=${sensitivity}`,
        {
            method: 'POST',
            body: formData
        }
    );
    
    const result = await response.json();
    return result;
}

// 2. Display base64 images in React component
function TamperingResults({ result }) {
    return (
        <div className="tampering-results">
            <h2>Detection Results</h2>
            
            <div className="summary">
                <p>Probability: {(result.probability * 100).toFixed(1)}%</p>
                <p>Status: {result.is_tampered ? '⚠️ Tampered' : '✅ Authentic'}</p>
                <p>Suspicious Regions: {result.num_regions}</p>
            </div>
            
            <div className="images-grid">
                <div className="image-card">
                    <h3>Heatmap</h3>
                    <img 
                        src={`data:image/png;base64,${result.heatmap}`} 
                        alt="Tampering Heatmap" 
                    />
                </div>
                
                <div className="image-card">
                    <h3>Mask</h3>
                    <img 
                        src={`data:image/png;base64,${result.mask}`} 
                        alt="Tampering Mask" 
                    />
                </div>
                
                <div className="image-card">
                    <h3>Detected Regions</h3>
                    <img 
                        src={`data:image/png;base64,${result.tampered_regions}`} 
                        alt="Detected Regions" 
                    />
                </div>
            </div>
        </div>
    );
}

// 3. Complete example with file upload
function TamperingDetectionApp() {
    const [result, setResult] = React.useState(null);
    const [loading, setLoading] = React.useState(false);
    
    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;
        
        setLoading(true);
        try {
            const detectionResult = await detectTampering(file, 0.5);
            setResult(detectionResult);
        } catch (error) {
            console.error('Detection failed:', error);
        }
        setLoading(false);
    };
    
    return (
        <div>
            <input type="file" onChange={handleFileUpload} accept="image/*" />
            {loading && <p>Analyzing...</p>}
            {result && <TamperingResults result={result} />}
        </div>
    );
}
"""


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Tampering Detection API - Client Examples")
    print("="*70)
    
    # Example 1: Call API and display results
    print("\n📋 Example 1: Basic API Call")
    print("-" * 70)
    print("""
result = call_tampering_api_python(
    "path/to/document.jpg",
    api_url="http://localhost:8001",
    sensitivity=0.5
)

# Result contains:
# - heatmap (base64 string)
# - mask (base64 string)
# - tampered_regions (base64 string)
# - probability (float)
# - bboxes (list)
    """)
    
    # Example 2: Save images
    print("\n📋 Example 2: Save Base64 Images to Disk")
    print("-" * 70)
    print("""
save_base64_images(result, output_dir="results")
# Saves: heatmap.png, mask.png, tampered_regions.png
    """)
    
    # Example 3: Display with OpenCV
    print("\n📋 Example 3: Display Results")
    print("-" * 70)
    print("""
display_results_cv2(result)
# Shows all three images in OpenCV windows
    """)
    
    # Example 4: HTML report
    print("\n📋 Example 4: Generate HTML Report")
    print("-" * 70)
    print("""
generate_html_report(result, "report.html")
# Creates interactive HTML report with embedded images
    """)
    
    # JavaScript example
    print("\n📋 Example 5: Frontend JavaScript/React")
    print("-" * 70)
    print(JAVASCRIPT_EXAMPLE)
    
    print("\n" + "="*70)
    print("✅ All examples documented!")
    print("="*70)
    
    # Uncomment to run a real example:
    # result = call_tampering_api_python("path/to/test/image.jpg")
    # if result:
    #     save_base64_images(result)
    #     generate_html_report(result)
    #     display_results_cv2(result)
