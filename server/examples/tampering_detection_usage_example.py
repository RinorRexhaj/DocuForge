"""
Example Usage of Updated detect_tampering_hybrid Function
==========================================================
This script demonstrates how to use the updated tampering detection function
that returns images directly instead of saving them to disk.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import base64

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from detection.tampering_localization import detect_tampering_hybrid
from models.predict import load_model


def example_1_numpy_arrays():
    """
    Example 1: Get images as numpy arrays (for programmatic use)
    """
    print("\n" + "="*70)
    print("Example 1: Returning Images as NumPy Arrays")
    print("="*70)
    
    # Load model
    model_path = Path(__file__).parent.parent / 'models' / 'saved_models' / 'best_model.pth'
    model, device = load_model(str(model_path))
    
    # Path to test image
    image_path = "path/to/your/test/image.jpg"
    
    # Run detection - returns numpy arrays
    result = detect_tampering_hybrid(
        image_path=image_path,
        model=model,
        device=device,
        save_results=False,  # Don't save to disk
        return_base64=False,  # Return as numpy arrays
        sensitivity=0.5
    )
    
    # Access the images (all are numpy arrays)
    heatmap = result['heatmap']  # RGB numpy array
    mask = result['mask']  # Grayscale numpy array (0-255)
    tampered_regions = result['tampered_regions']  # RGB numpy array with bounding boxes
    probability = result['probability']  # Float (0-1)
    bboxes = result['bboxes']  # List of (x, y, w, h) tuples
    
    print(f"\n✅ Detection completed!")
    print(f"   Heatmap shape: {heatmap.shape}")
    print(f"   Mask shape: {mask.shape}")
    print(f"   Tampered regions shape: {tampered_regions.shape}")
    print(f"   Tampering probability: {probability:.2%}")
    print(f"   Number of suspicious regions: {len(bboxes)}")
    
    # You can now use these images directly in your code
    # For example, display them:
    cv2.imshow("Heatmap", cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    cv2.imshow("Mask", mask)
    cv2.imshow("Tampered Regions", cv2.cvtColor(tampered_regions, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return result


def example_2_base64_strings():
    """
    Example 2: Get images as base64 strings (for API/JSON use)
    """
    print("\n" + "="*70)
    print("Example 2: Returning Images as Base64 Strings")
    print("="*70)
    
    # Load model
    model_path = Path(__file__).parent.parent / 'models' / 'saved_models' / 'best_model.pth'
    model, device = load_model(str(model_path))
    
    # Path to test image
    image_path = "path/to/your/test/image.jpg"
    
    # Run detection - returns base64 strings
    result = detect_tampering_hybrid(
        image_path=image_path,
        model=model,
        device=device,
        save_results=False,  # Don't save to disk
        return_base64=True,  # Return as base64 strings
        sensitivity=0.5
    )
    
    # Access the images (all are base64 strings)
    heatmap_b64 = result['heatmap']  # Base64 string
    mask_b64 = result['mask']  # Base64 string
    tampered_regions_b64 = result['tampered_regions']  # Base64 string
    probability = result['probability']  # Float (0-1)
    bboxes = result['bboxes']  # List of (x, y, w, h) tuples
    
    print(f"\n✅ Detection completed!")
    print(f"   Heatmap (base64): {heatmap_b64[:50]}... ({len(heatmap_b64)} chars)")
    print(f"   Mask (base64): {mask_b64[:50]}... ({len(mask_b64)} chars)")
    print(f"   Tampered regions (base64): {tampered_regions_b64[:50]}... ({len(tampered_regions_b64)} chars)")
    print(f"   Tampering probability: {probability:.2%}")
    print(f"   Number of suspicious regions: {len(bboxes)}")
    
    # This format is perfect for sending over HTTP APIs
    # Example: Return in FastAPI response
    """
    @app.post("/detect-tampering")
    async def detect_tampering(file: UploadFile):
        result = detect_tampering_hybrid(
            image_path=saved_file_path,
            model=model,
            device=device,
            return_base64=True
        )
        return JSONResponse(content=result)
    """
    
    # If you need to convert base64 back to image:
    def base64_to_image(b64_string):
        """Convert base64 string back to numpy array"""
        img_bytes = base64.b64decode(b64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Decode and display
    heatmap_img = base64_to_image(heatmap_b64)
    print(f"\n   Decoded heatmap shape: {heatmap_img.shape}")
    
    return result


def example_3_with_intermediate_maps():
    """
    Example 3: Get all intermediate detection maps
    """
    print("\n" + "="*70)
    print("Example 3: Including Intermediate Detection Maps")
    print("="*70)
    
    # Load model
    model_path = Path(__file__).parent.parent / 'models' / 'saved_models' / 'best_model.pth'
    model, device = load_model(str(model_path))
    
    # Path to test image
    image_path = "path/to/your/test/image.jpg"
    
    # Run detection with intermediate maps
    result = detect_tampering_hybrid(
        image_path=image_path,
        model=model,
        device=device,
        save_results=False,
        return_base64=False,
        return_intermediate_maps=True,  # Get all individual detection maps
        sensitivity=0.5
    )
    
    # Access intermediate maps
    print(f"\n✅ Detection completed!")
    print(f"   Main outputs: heatmap, mask, tampered_regions, bboxes")
    print(f"   Tampering probability: {result['probability']:.2%}")
    
    if 'intermediate_maps' in result:
        print(f"\n   Intermediate maps available:")
        for map_name in result['intermediate_maps'].keys():
            print(f"     - {map_name}")
        print(f"   GradCAM map also available")
    
    return result


def example_4_save_and_return():
    """
    Example 4: Both save to disk AND return images
    """
    print("\n" + "="*70)
    print("Example 4: Save to Disk AND Return Images")
    print("="*70)
    
    # Load model
    model_path = Path(__file__).parent.parent / 'models' / 'saved_models' / 'best_model.pth'
    model, device = load_model(str(model_path))
    
    # Path to test image
    image_path = "path/to/your/test/image.jpg"
    
    # Run detection - both save and return
    result = detect_tampering_hybrid(
        image_path=image_path,
        model=model,
        device=device,
        save_results=True,  # Save to disk
        return_base64=False,  # Also return as numpy arrays
        sensitivity=0.5
    )
    
    print(f"\n✅ Detection completed!")
    print(f"   Images saved to: tampering_results/")
    print(f"   Images also returned in result dictionary")
    print(f"   You can use them directly in your code!")
    
    return result


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Tampering Detection - Updated Return Format Examples")
    print("="*70)
    
    print("\n📝 The function now returns images directly instead of just saving them!")
    print("\n   Two modes available:")
    print("   1. return_base64=False → Returns numpy arrays (for Python code)")
    print("   2. return_base64=True  → Returns base64 strings (for APIs/JSON)")
    
    print("\n   Return dictionary now includes:")
    print("   - 'heatmap': Tampering heatmap overlay")
    print("   - 'mask': Binary tampering mask")
    print("   - 'tampered_regions': Original image with bounding boxes")
    print("   - 'bboxes': List of detected region coordinates")
    print("   - 'probability': Overall tampering confidence")
    print("   - 'fused_map': Raw detection scores")
    
    print("\n" + "="*70)
    print("\nUncomment the example you want to run:\n")
    
    # Uncomment to run examples:
    # example_1_numpy_arrays()
    # example_2_base64_strings()
    # example_3_with_intermediate_maps()
    # example_4_save_and_return()
    
    print("\n✅ Check the function signatures for full details!")
