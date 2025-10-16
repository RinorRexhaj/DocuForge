"""
Quick test for enhanced_blur_detection method
"""
import numpy as np
import cv2
import sys
from pathlib import Path

# Add server directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.tampering_localization import DocumentTamperingDetector
import torch
import torch.nn as nn

class DummyModel(nn.Module):
    """Dummy model for testing"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)
    
    def forward(self, x):
        return self.conv(x)

# Create test image
print("Creating test image...")
img = (np.random.rand(400, 600, 3) * 255).astype(np.uint8)

# Initialize detector with dummy model
print("Initializing detector...")
dummy_model = DummyModel()
detector = DocumentTamperingDetector(model=dummy_model)

# Test enhanced blur detection
print("Testing enhanced_blur_detection()...")
try:
    result = detector.enhanced_blur_detection(img)
    print(f"✓ enhanced_blur_detection() works!")
    
    # Handle both single array and tuple outputs
    if isinstance(result, tuple):
        combined_map, details = result
        print(f"  Combined map shape: {combined_map.shape}, dtype: {combined_map.dtype}")
        print(f"  Combined map range: [{combined_map.min():.3f}, {combined_map.max():.3f}]")
        print(f"  Details keys: {list(details.keys())}")
    else:
        print(f"  Output shape: {result.shape}, dtype: {result.dtype}")
        print(f"  Output range: [{result.min():.3f}, {result.max():.3f}]")
    
    print("\n✅ Blur detection method verified successfully!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
