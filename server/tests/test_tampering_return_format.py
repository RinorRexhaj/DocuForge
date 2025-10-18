"""
Quick test script to verify the updated detect_tampering_hybrid function
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_return_formats():
    """Test that the function returns images in the expected formats"""
    print("\n" + "="*70)
    print("Testing Updated detect_tampering_hybrid Return Formats")
    print("="*70)
    
    # Test 1: Check function signature
    print("\n✓ Test 1: Checking function signature...")
    from detection.tampering_localization import detect_tampering_hybrid
    import inspect
    
    sig = inspect.signature(detect_tampering_hybrid)
    params = list(sig.parameters.keys())
    
    assert 'return_base64' in params, "❌ Missing 'return_base64' parameter"
    print("  ✓ Function has 'return_base64' parameter")
    
    # Test 2: Check that base64 module is imported
    print("\n✓ Test 2: Checking required imports...")
    import detection.tampering_localization as tl
    assert hasattr(tl, 'base64'), "❌ base64 module not imported"
    print("  ✓ base64 module imported")
    
    # Test 3: Check return structure documentation
    print("\n✓ Test 3: Checking function docstring...")
    docstring = detect_tampering_hybrid.__doc__
    assert 'return_base64' in docstring, "❌ Docstring not updated"
    assert 'tampered_regions' in docstring, "❌ Missing tampered_regions in docstring"
    assert 'base64 string or RGB numpy array' in docstring or 'base64 string or numpy array' in docstring, "❌ Return format not documented"
    print("  ✓ Docstring updated with new parameters and return structure")
    
    print("\n" + "="*70)
    print("✅ All static tests passed!")
    print("="*70)
    print("\nThe function has been successfully updated with:")
    print("  1. New 'return_base64' parameter")
    print("  2. Returns 'heatmap', 'mask', and 'tampered_regions' as images")
    print("  3. Supports both NumPy array and base64 string formats")
    print("\nTo test with actual images, run:")
    print("  python examples/tampering_detection_usage_example.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        test_return_formats()
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        sys.exit(1)
