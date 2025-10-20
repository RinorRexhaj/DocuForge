"""
Example Usage: Document Tampering Localization
===============================================
Demonstrates how to use the hybrid tampering detection system.
"""

import os
import sys
import torch
import cv2
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the tampering detection module
from detection.tampering_localization import (
    detect_tampering_hybrid,
    visualize_detection_result,
    batch_detect_tampering
)
from models.predict import load_model, predict

def example_single_image_detection():
    """Example 1: Detect tampering in a single document image."""
    
    print("\n" + "="*70)
    print("Example 1: Single Image Tampering Detection")
    print("="*70 + "\n")
    
    # Load your pre-trained model
    model_path = Path(__file__).parent.parent / "models" / "saved_models" / "best_model.pth"
    
    if os.path.exists(model_path):
        model, device = load_model(str(model_path))
            
        model.eval()
        print(f"✓ Model loaded from: {model_path}\n")
    else:
        print(f"❌ Model not found at: {model_path}")
        print("Please provide a valid model path.")
        return
    
    # Specify document image to analyze
    image_path = "dataset/test/forged/0000975053_forged.png"

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("Please provide a valid image path.")
        return
    
    # Detect tampering
    result = detect_tampering_hybrid(
        image_path=image_path,
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_results=True,
        sensitivity=0.5,
        return_intermediate_maps=True
    )
    
    # Display results
    visualize_detection_result(result, display=False)
    
    # Access individual results
    print(f"\n📋 Detailed Results:")
    print(f"  - Tampering Probability: {result['probability']:.2%}")
    print(f"  - Is Suspicious: {'YES' if result['probability'] > 0.5 else 'NO'}")
    print(f"  - Number of tampered regions: {len(result['bboxes'])}")
    
    if result['bboxes']:
        print(f"\n  📦 Bounding Boxes (tampered regions):")
        for i, (x, y, w, h) in enumerate(result['bboxes'], 1):
            print(f"    Region {i}: Position=({x}, {y}), Size=({w}×{h}) pixels")
    
    # Show intermediate forensic maps
    if 'intermediate_maps' in result:
        print(f"\n  🔬 Forensic Analysis Components:")
        for name in result['intermediate_maps'].keys():
            print(f"    - {name.upper()} analysis completed")
    
    print(f"\n✅ Results saved to: tampering_results/")


def example_batch_processing():
    """Example 2: Process multiple documents in batch."""
    
    print("\n" + "="*70)
    print("Example 2: Batch Processing Multiple Documents")
    print("="*70 + "\n")
    
    # Load model
    model_path = "saved_models/best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        return
    
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    # Get all test images
    test_dirs = [
        "dataset/test/authentic/",
        "dataset/test/forged/"
    ]
    
    image_paths = []
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_paths.extend(Path(test_dir).glob(ext))
    
    if not image_paths:
        print("❌ No test images found")
        return
    
    # Limit to first 10 for demo
    image_paths = [str(p) for p in image_paths[:10]]
    
    print(f"Processing {len(image_paths)} documents...\n")
    
    # Batch process
    results_df = batch_detect_tampering(
        image_paths=image_paths,
        model=model,
        device=device,
        output_csv="tampering_results/batch_results.csv"
    )
    
    # Display summary
    print("\n📊 Batch Processing Summary:")
    print(results_df.to_string(index=False))
    
    # Statistics
    suspicious_count = len(results_df[results_df['status'] == 'suspicious'])
    authentic_count = len(results_df[results_df['status'] == 'authentic'])
    
    print(f"\n📈 Statistics:")
    print(f"  - Suspicious documents: {suspicious_count}")
    print(f"  - Authentic documents: {authentic_count}")
    print(f"  - Average tampering probability: {results_df['tampering_probability'].mean():.2%}")


def example_custom_sensitivity():
    """Example 3: Adjust sensitivity for different use cases."""
    
    print("\n" + "="*70)
    print("Example 3: Custom Sensitivity Adjustment")
    print("="*70 + "\n")
    
    model_path = "saved_models/best_model.pth"
    image_path = "dataset/test/forged/2048868340_forged.png"
    
    if not os.path.exists(model_path) or not os.path.exists(image_path):
        print("❌ Required files not found")
        return

    model, device = load_model(model_path)

    # Test different sensitivity levels
    sensitivities = [0.3, 0.5, 0.7]
    
    for sensitivity in sensitivities:
        print(f"\n🎚️  Testing with sensitivity = {sensitivity}")
        
        result = detect_tampering_hybrid(
            image_path=image_path,
            model=model,
            device=device,
            save_results=True,
            sensitivity=sensitivity
        )
        
        print(f"  - Detected regions: {len(result['bboxes'])}")
        print(f"  - Probability: {result['probability']:.2%}")


def example_with_visualization():
    """Example 4: Generate and display visualizations."""
    
    print("\n" + "="*70)
    print("Example 4: Interactive Visualization")
    print("="*70 + "\n")
    
    model_path = "saved_models/best_model.pth"
    image_path = "dataset/test/forged/2048868340_forged.png"
    
    if not os.path.exists(model_path) or not os.path.exists(image_path):
        print("❌ Required files not found")
        return

    model, device = load_model(model_path)

    # Run detection with all intermediate results
    result = detect_tampering_hybrid(
        image_path=image_path,
        model=model,
        device=device,
        save_results=True,
        return_intermediate_maps=True
    )
    
    # Display heatmap
    cv2.imshow("Tampering Heatmap Overlay", cv2.cvtColor(result['heatmap'], cv2.COLOR_RGB2BGR))
    
    # Display binary mask
    cv2.imshow("Binary Tampering Mask", result['mask'] * 255)
    
    # Display intermediate forensic maps
    if 'intermediate_maps' in result:
        for name, map_data in result['intermediate_maps'].items():
            map_display = (map_data * 255).astype('uint8')
            map_colored = cv2.applyColorMap(map_display, cv2.COLORMAP_JET)
            cv2.imshow(f"Forensic: {name.upper()}", map_colored)
    
    print("\n📺 Displaying visualizations...")
    print("Press any key to close windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def example_api_integration():
    """Example 5: Integration with API/Web service."""
    
    print("\n" + "="*70)
    print("Example 5: API Integration Example")
    print("="*70 + "\n")
    
    import json
    
    model_path = "server/models/saved_models/best_model.pth"
    image_path = "server/examples/2048864472_2048864474_forged.png"
    
    if not os.path.exists(model_path) or not os.path.exists(image_path):
        print("❌ Required files not found")
        return

    model, device = load_model(model_path)
    model.eval()

    prediction = predict(image_path=image_path, model=model, return_probability=True)
    print(f"🔧 Predicted sensitivity: {prediction['probability']:.2f}")

    # Run detection
    result = detect_tampering_hybrid(
        image_path=image_path,
        model=model,
        device=device,
        save_results=True,
        sensitivity=0.7
    )
    
    # Create API response format
    api_response = {
        "status": "success",
        "document": os.path.basename(image_path),
        "analysis": {
            "is_tampered": result['probability'] > 0.5,
            "confidence": float(result['probability']),
            "tampering_score": float(result['probability']),
            "risk_level": "high" if result['probability'] > 0.7 else "medium" if result['probability'] > 0.4 else "low"
        },
        "detected_regions": [
            {
                "region_id": i,
                "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "area_pixels": int(w * h)
            }
            for i, (x, y, w, h) in enumerate(result['bboxes'], 1)
        ],
        "visualization_urls": {
            "heatmap": "tampering_results/sample_document_heatmap.png",
            "mask": "tampering_results/sample_document_mask.png",
            "analysis": "tampering_results/sample_document_tampering_analysis.png"
        }
    }
    
    # Print JSON response
    print("📤 API Response (JSON):")
    print(json.dumps(api_response, indent=2))
    
    # Save response
    with open("server/tampering_results/api_response.json", "w") as f:
        json.dump(api_response, f, indent=2)

    print(f"\n✅ API response saved to: server/tampering_results/api_response.json")


def main():
    """Run all examples."""
    
    print("\n" + "="*70)
    print("🔍 Document Tampering Localization - Examples")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists("models/saved_models/best_model.pth"):
        print("\n⚠️  Warning: Model file not found at 'saved_models/best_model.pth'")
        print("Please train a model first or provide the correct path.")
        print("\nYou can still review the example code structure.\n")
    
    # Run examples
    try:
        # Uncomment the examples you want to run:
        
        # example_single_image_detection()
        
        # example_batch_processing()
        
        # example_custom_sensitivity()
        
        # example_with_visualization()
        
        example_api_integration()
        
    except Exception as e:
        print(f"\n❌ Error running example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
