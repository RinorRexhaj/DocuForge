"""
Quick Demo: Blur Detection on Forgery.py Images
================================================
This is a simple demo showing how to detect blur artifacts from Forgery.py.
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

from detection.tampering_localization import detect_tampering_hybrid


def quick_demo():
    """Quick demonstration of blur detection."""
    
    print("\n" + "="*70)
    print(" 🎯 BLUR DETECTION DEMO FOR FORGERY.PY ARTIFACTS")
    print("="*70 + "\n")
    
    # Setup
    print("📋 Setup:")
    print("  - This demo detects blur/smudge tampering from Forgery.py")
    print("  - Specifically designed for: local_blur_smudge() operations")
    print("  - Detects: Gaussian blur, motion blur, splice boundaries\n")
    
    # Check for model
    model_path = "saved_models/best_model.pth"
    
    if not os.path.exists(model_path):
        print("⚠️  Model not found. Creating a dummy model for demo...")
        # Create simple model for demo
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
                self.layer4 = nn.Sequential(nn.Conv2d(64, 512, 3, padding=1))
                self.avgpool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(512, 2)
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.layer4(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                return self.fc(x)
        
        model = SimpleModel()
        model.eval()
        print("  ✓ Dummy model created\n")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(model_path, map_location=device)
        model.eval()
        print(f"  ✓ Model loaded from {model_path}\n")
    
    # Find test images
    print("🔍 Looking for forged images...")
    
    # Try multiple possible locations
    possible_dirs = [
        "E:\\Thesis\\forgery_dataset_realistic\\train\\forged",
        "E:\\Thesis\\forgery_dataset_realistic\\test\\forged",
        "dataset/test/forged",
        "test/test/forged",
    ]
    
    test_images = []
    for directory in possible_dirs:
        if os.path.exists(directory):
            test_images = [f for f in Path(directory).glob("*.jpg")]
            if test_images:
                print(f"  ✓ Found {len(test_images)} images in {directory}\n")
                break
    
    if not test_images:
        print("  ❌ No forged images found in standard locations.")
        print("  Please provide a path to a forged image:\n")
        
        # Manual input
        img_path = input("  Enter image path (or press Enter to skip): ").strip()
        if img_path and os.path.exists(img_path):
            test_images = [Path(img_path)]
        else:
            print("\n  Skipping demo. Run Forgery.py first to generate test images.")
            return
    
    # Process first image
    img_path = str(test_images[0])
    print(f"📄 Processing: {Path(img_path).name}\n")
    
    print("="*70)
    print(" RUNNING ENHANCED BLUR DETECTION")
    print("="*70 + "\n")
    
    # Run detection
    try:
        result = detect_tampering_hybrid(
            image_path=img_path,
            model=model,
            device="cuda" if torch.cuda.is_available() else "cpu",
            save_results=True,
            sensitivity=0.4,  # More sensitive for blur
            return_intermediate_maps=True
        )
        
        print("\n" + "="*70)
        print(" DETECTION RESULTS")
        print("="*70 + "\n")
        
        # Overall results
        print(f"📊 Overall Analysis:")
        print(f"  - Tampering Probability: {result['probability']:.2%}")
        
        if result['probability'] > 0.7:
            print(f"  - Assessment: 🚫 HIGH RISK - Likely tampered")
        elif result['probability'] > 0.5:
            print(f"  - Assessment: ⚠️  MEDIUM RISK - Suspicious")
        elif result['probability'] > 0.3:
            print(f"  - Assessment: ⚠️  LOW RISK - Minor artifacts")
        else:
            print(f"  - Assessment: ✅ AUTHENTIC - Low suspicion")
        
        print(f"  - Detected Regions: {len(result['bboxes'])}")
        
        # Bounding boxes
        if result['bboxes']:
            print(f"\n  📦 Tampered Regions:")
            for i, (x, y, w, h) in enumerate(result['bboxes'], 1):
                area_pct = (w * h) / (result['mask'].shape[0] * result['mask'].shape[1]) * 100
                print(f"    {i}. Position: ({x:4d}, {y:4d})  Size: {w:4d}×{h:4d} px  Area: {area_pct:.1f}%")
        
        # Blur-specific results
        if 'intermediate_maps' in result:
            blur_maps = {k: v for k, v in result['intermediate_maps'].items() 
                        if k.startswith('blur_')}
            
            if blur_maps:
                print(f"\n🎯 Blur Detection Breakdown:")
                
                for name, blur_map in blur_maps.items():
                    if isinstance(blur_map, np.ndarray) and blur_map.size > 0:
                        mean_score = float(np.mean(blur_map))
                        max_score = float(np.max(blur_map))
                        
                        # Format name
                        display_name = name.replace('blur_', '').replace('_', ' ').title()
                        
                        # Status indicator
                        if max_score > 0.7:
                            status = "🔴 HIGH"
                        elif max_score > 0.4:
                            status = "🟡 MEDIUM"
                        else:
                            status = "🟢 LOW"
                        
                        print(f"  - {display_name:25s}: avg={mean_score:.3f}  max={max_score:.3f}  {status}")
        
        # Visualizations
        print(f"\n💾 Saved Outputs:")
        output_dir = "tampering_results"
        basename = Path(img_path).stem
        
        saved_files = [
            f"{basename}_tampering_analysis.png",
            f"{basename}_heatmap.png",
            f"{basename}_mask.png"
        ]
        
        for filename in saved_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                print(f"  ✓ {filename}")
        
        print(f"\n📁 All results saved to: {output_dir}/")
        
        # Summary
        print(f"\n" + "="*70)
        print(" SUMMARY")
        print("="*70 + "\n")
        
        print("✅ Detection completed successfully!")
        print(f"   The system analyzed the image for blur/smudge tampering")
        print(f"   artifacts generated by Forgery.py.\n")
        
        # Interpretation
        if result['probability'] > 0.5:
            print("🔍 Interpretation:")
            print("   This image shows signs of tampering, likely including:")
            
            if 'blur_blur' in result.get('intermediate_maps', {}):
                if np.max(result['intermediate_maps']['blur_blur']) > 0.6:
                    print("   • Local Gaussian blur (from local_blur_smudge())")
            
            if 'blur_motion_blur' in result.get('intermediate_maps', {}):
                if np.max(result['intermediate_maps']['blur_motion_blur']) > 0.5:
                    print("   • Motion blur artifacts")
            
            if 'blur_text_overlay' in result.get('intermediate_maps', {}):
                if np.max(result['intermediate_maps']['blur_text_overlay']) > 0.6:
                    print("   • Text overlay/editing (from text_overlay())")
            
            if 'blur_splice_boundary' in result.get('intermediate_maps', {}):
                if np.max(result['intermediate_maps']['blur_splice_boundary']) > 0.6:
                    print("   • Copy-paste splicing (from copy_paste_splice())")
        
        print()
        
    except Exception as e:
        print(f"\n❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Offer to process more
    if len(test_images) > 1:
        print("="*70)
        response = input(f"\n🔄 Process more images? ({len(test_images)-1} remaining) [y/N]: ").strip().lower()
        
        if response == 'y':
            print("\n")
            for img_path in test_images[1:3]:  # Process 2 more
                print(f"📄 Processing: {Path(img_path).name}")
                
                try:
                    result = detect_tampering_hybrid(
                        image_path=str(img_path),
                        model=model,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        save_results=True,
                        sensitivity=0.4,
                        return_intermediate_maps=False  # Faster
                    )
                    
                    print(f"  ✓ Probability: {result['probability']:.2%}  Regions: {len(result['bboxes'])}\n")
                    
                except Exception as e:
                    print(f"  ❌ Error: {e}\n")
    
    print("="*70)
    print("✅ Demo completed! Check tampering_results/ for visualizations.")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        quick_demo()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
