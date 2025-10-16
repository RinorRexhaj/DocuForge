"""
Quick Test: Verify OpenCV medianBlur Fix
=========================================
This script tests that the medianBlur fix resolves the error.
"""

import sys
import numpy as np
import cv2

print("Testing OpenCV medianBlur fix...\n")

# Test 1: The old problematic code (should fail)
print("Test 1: Old problematic code")
try:
    test_array = np.random.rand(100, 100).astype(np.float32)
    result = cv2.medianBlur(test_array, 15)
    print("  ❌ Old code should have failed but didn't!")
except cv2.error as e:
    print(f"  ✓ Old code fails as expected: {str(e)[:80]}...")

# Test 2: The new fixed code (should work)
print("\nTest 2: New fixed code")
try:
    test_array = np.random.rand(100, 100).astype(np.float32)
    
    # New approach: scale to uint8, apply filter, scale back
    test_array_max = test_array.max()
    test_array_scaled = (test_array * 255 / (test_array_max + 1e-8)).astype(np.uint8)
    result_scaled = cv2.medianBlur(test_array_scaled, 15)
    result = result_scaled.astype(np.float32) * (test_array_max + 1e-8) / 255.0
    
    print(f"  ✓ New code works!")
    print(f"  Input shape: {test_array.shape}, dtype: {test_array.dtype}")
    print(f"  Output shape: {result.shape}, dtype: {result.dtype}")
    print(f"  Input range: [{test_array.min():.3f}, {test_array.max():.3f}]")
    print(f"  Output range: [{result.min():.3f}, {result.max():.3f}]")
except Exception as e:
    print(f"  ❌ New code failed: {e}")

# Test 3: Test the actual module
print("\nTest 3: Testing tampering_localization module")
try:
    from tampering_localization import DocumentTamperingDetector
    import torch
    import torch.nn as nn
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7)
            self.layer4 = nn.Sequential(nn.Conv2d(64, 512, 3))
            self.fc = nn.Linear(512, 2)
        
        def forward(self, x):
            x = self.conv1(x)
            return self.fc(torch.flatten(x, 1))
    
    model = DummyModel()
    model.eval()
    
    # Create detector
    detector = DocumentTamperingDetector(model, device="cpu", output_dir="test_results")
    
    # Create test image
    test_img = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    
    # Test noise inconsistency map (this uses medianBlur)
    noise_map = detector.noise_inconsistency_map(test_img)
    
    print(f"  ✓ noise_inconsistency_map() works!")
    print(f"  Output shape: {noise_map.shape}, dtype: {noise_map.dtype}")
    print(f"  Output range: [{noise_map.min():.3f}, {noise_map.max():.3f}]")
    
    # Test other methods
    ela_map = detector.error_level_analysis(cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
    print(f"  ✓ error_level_analysis() works!")
    
    edge_map = detector.edge_artifact_map(test_img)
    print(f"  ✓ edge_artifact_map() works!")
    
    jpeg_map = detector.jpeg_block_artifact_analysis(test_img)
    print(f"  ✓ jpeg_block_artifact_analysis() works!")
    
    print("\n✅ ALL TESTS PASSED!")
    print("The medianBlur error has been fixed.")
    
except Exception as e:
    print(f"  ❌ Module test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ OpenCV medianBlur fix verified successfully!")
print("="*70)
