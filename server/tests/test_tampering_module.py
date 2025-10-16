"""
Test Suite for Document Tampering Localization Module
======================================================
This script verifies that all components are working correctly.
"""

import os
import sys
import traceback
from pathlib import Path

# Test results tracker
test_results = {
    "passed": [],
    "failed": [],
    "warnings": []
}


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def test_dependencies():
    """Test 1: Check if all required dependencies are installed."""
    print_section("TEST 1: Dependency Check")
    
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'torch': 'torch',
        'PIL': 'Pillow',
        'skimage': 'scikit-image',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib'
    }
    
    missing = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {package:20s} installed")
            test_results["passed"].append(f"Dependency: {package}")
        except ImportError:
            print(f"✗ {package:20s} MISSING")
            missing.append(package)
            test_results["failed"].append(f"Dependency: {package}")
    
    # Optional: pytorch-grad-cam
    try:
        __import__('pytorch_grad_cam')
        print(f"✓ {'pytorch-grad-cam':20s} installed (optional)")
        test_results["passed"].append("Optional: pytorch-grad-cam")
    except ImportError:
        print(f"⚠ {'pytorch-grad-cam':20s} not installed (will use fallback)")
        test_results["warnings"].append("Optional: pytorch-grad-cam not installed")
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    else:
        print("\n✅ All required dependencies installed!")
        return True


def test_module_import():
    """Test 2: Check if the tampering localization module can be imported."""
    print_section("TEST 2: Module Import")
    
    try:
        from tampering_localization import (
            DocumentTamperingDetector,
            detect_tampering_hybrid,
            visualize_detection_result,
            batch_detect_tampering
        )
        print("✓ Main module imported successfully")
        print("✓ DocumentTamperingDetector class available")
        print("✓ detect_tampering_hybrid function available")
        print("✓ visualize_detection_result function available")
        print("✓ batch_detect_tampering function available")
        
        test_results["passed"].append("Module import")
        return True
        
    except Exception as e:
        print(f"✗ Failed to import module: {e}")
        traceback.print_exc()
        test_results["failed"].append("Module import")
        return False


def test_detector_initialization():
    """Test 3: Check if detector can be initialized."""
    print_section("TEST 3: Detector Initialization")
    
    try:
        import torch
        import torch.nn as nn
        from tampering_localization import DocumentTamperingDetector
        
        # Create a dummy model
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.fc = nn.Linear(64 * 224 * 224, 2)
            
            def forward(self, x):
                x = self.conv1(x)
                return x.view(x.size(0), -1)
        
        dummy_model = DummyModel()
        dummy_model.eval()
        
        # Initialize detector
        detector = DocumentTamperingDetector(
            model=dummy_model,
            device="cpu",
            output_dir="test_results"
        )
        
        print("✓ Detector initialized successfully")
        print(f"✓ Using device: {detector.device}")
        print(f"✓ Output directory: {detector.output_dir}")
        
        test_results["passed"].append("Detector initialization")
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize detector: {e}")
        traceback.print_exc()
        test_results["failed"].append("Detector initialization")
        return False


def test_forensic_methods():
    """Test 4: Check if individual forensic methods work."""
    print_section("TEST 4: Forensic Methods")
    
    try:
        import cv2
        import numpy as np
        import torch
        import torch.nn as nn
        from tampering_localization import DocumentTamperingDetector
        
        # Create test image
        test_img = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        test_img_rgb = test_img.copy()
        
        # Create dummy model
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3)
            
            def forward(self, x):
                return self.conv1(x)
        
        detector = DocumentTamperingDetector(
            model=DummyModel(),
            device="cpu"
        )
        
        # Test ELA
        try:
            ela_map = detector.error_level_analysis(test_img)
            assert ela_map.shape == (400, 600)
            assert ela_map.min() >= 0 and ela_map.max() <= 1
            print("✓ Error Level Analysis works")
            test_results["passed"].append("ELA method")
        except Exception as e:
            print(f"✗ ELA failed: {e}")
            test_results["failed"].append("ELA method")
        
        # Test Noise Inconsistency
        try:
            noise_map = detector.noise_inconsistency_map(test_img_rgb)
            assert noise_map.shape == (400, 600)
            assert noise_map.min() >= 0 and noise_map.max() <= 1
            print("✓ Noise Inconsistency Analysis works")
            test_results["passed"].append("Noise method")
        except Exception as e:
            print(f"✗ Noise analysis failed: {e}")
            test_results["failed"].append("Noise method")
        
        # Test Edge Artifacts
        try:
            edge_map = detector.edge_artifact_map(test_img_rgb)
            assert edge_map.shape == (400, 600)
            assert edge_map.min() >= 0 and edge_map.max() <= 1
            print("✓ Edge Artifact Detection works")
            test_results["passed"].append("Edge method")
        except Exception as e:
            print(f"✗ Edge artifact detection failed: {e}")
            test_results["failed"].append("Edge method")
        
        # Test JPEG Block Analysis
        try:
            jpeg_map = detector.jpeg_block_artifact_analysis(test_img_rgb)
            assert jpeg_map.shape == (400, 600)
            assert jpeg_map.min() >= 0 and jpeg_map.max() <= 1
            print("✓ JPEG Block Artifact Analysis works")
            test_results["passed"].append("JPEG method")
        except Exception as e:
            print(f"✗ JPEG block analysis failed: {e}")
            test_results["failed"].append("JPEG method")
        
        # Test Copy-Move Detection
        try:
            copymove_map = detector.copy_move_detection(test_img_rgb)
            assert copymove_map.shape == (400, 600)
            assert copymove_map.min() >= 0 and copymove_map.max() <= 1
            print("✓ Copy-Move Detection works")
            test_results["passed"].append("Copy-Move method")
        except Exception as e:
            print(f"✗ Copy-Move detection failed: {e}")
            test_results["failed"].append("Copy-Move method")
        
        print("\n✅ All forensic methods functional!")
        return True
        
    except Exception as e:
        print(f"\n✗ Forensic methods test failed: {e}")
        traceback.print_exc()
        test_results["failed"].append("Forensic methods")
        return False


def test_fusion_and_visualization():
    """Test 5: Check fusion and visualization functions."""
    print_section("TEST 5: Fusion and Visualization")
    
    try:
        import numpy as np
        import torch.nn as nn
        from tampering_localization import DocumentTamperingDetector
        
        # Create dummy model
        class DummyModel(nn.Module):
            def forward(self, x):
                return x
        
        detector = DocumentTamperingDetector(
            model=DummyModel(),
            device="cpu"
        )
        
        # Create test data
        h, w = 400, 600
        gradcam_map = np.random.rand(224, 224)
        classical_maps = [np.random.rand(h, w) for _ in range(3)]
        test_img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        
        # Test fusion
        try:
            fused = detector.combine_maps_fusion(gradcam_map, classical_maps)
            assert fused.shape == (h, w)
            assert fused.min() >= 0 and fused.max() <= 1
            print("✓ Map fusion works")
            test_results["passed"].append("Fusion method")
        except Exception as e:
            print(f"✗ Fusion failed: {e}")
            test_results["failed"].append("Fusion method")
        
        # Test heatmap overlay
        try:
            heatmap = np.random.rand(h, w)
            overlay = detector.apply_heatmap_overlay(test_img, heatmap)
            assert overlay.shape == (h, w, 3)
            print("✓ Heatmap overlay works")
            test_results["passed"].append("Heatmap overlay")
        except Exception as e:
            print(f"✗ Heatmap overlay failed: {e}")
            test_results["failed"].append("Heatmap overlay")
        
        # Test bounding box extraction
        try:
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[100:200, 150:300] = 1  # Add a region
            bboxes = detector.extract_bounding_boxes_from_mask(mask)
            assert len(bboxes) > 0
            print(f"✓ Bounding box extraction works (found {len(bboxes)} boxes)")
            test_results["passed"].append("BBox extraction")
        except Exception as e:
            print(f"✗ Bounding box extraction failed: {e}")
            test_results["failed"].append("BBox extraction")
        
        print("\n✅ Fusion and visualization methods work!")
        return True
        
    except Exception as e:
        print(f"\n✗ Fusion/visualization test failed: {e}")
        traceback.print_exc()
        test_results["failed"].append("Fusion and visualization")
        return False


def test_end_to_end():
    """Test 6: End-to-end pipeline with synthetic image."""
    print_section("TEST 6: End-to-End Pipeline")
    
    try:
        import cv2
        import numpy as np
        import torch
        import torch.nn as nn
        from tampering_localization import detect_tampering_hybrid
        
        # Create a simple test model
        class SimpleResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
                self.layer4 = nn.Sequential(
                    nn.Conv2d(64, 512, 3, padding=1)
                )
                self.fc = nn.Linear(512, 2)
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.layer4(x)
                x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        model = SimpleResNet()
        model.eval()
        
        # Create test image
        test_img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        test_path = "test_results/test_image.jpg"
        
        # Create directory if needed
        Path("test_results").mkdir(exist_ok=True)
        
        # Save test image
        cv2.imwrite(test_path, test_img)
        print(f"✓ Test image created: {test_path}")
        
        # Run detection
        print("Running full tampering detection pipeline...")
        result = detect_tampering_hybrid(
            image_path=test_path,
            model=model,
            device="cpu",
            save_results=False,  # Don't save for test
            sensitivity=0.5,
            return_intermediate_maps=True
        )
        
        # Verify output structure
        assert 'heatmap' in result, "Missing heatmap in result"
        assert 'mask' in result, "Missing mask in result"
        assert 'bboxes' in result, "Missing bboxes in result"
        assert 'probability' in result, "Missing probability in result"
        assert 'fused_map' in result, "Missing fused_map in result"
        
        print(f"✓ Pipeline completed successfully")
        print(f"  - Probability: {result['probability']:.4f}")
        print(f"  - Detected regions: {len(result['bboxes'])}")
        print(f"  - Heatmap shape: {result['heatmap'].shape}")
        print(f"  - Mask shape: {result['mask'].shape}")
        
        if 'intermediate_maps' in result:
            print(f"  - Intermediate maps: {len(result['intermediate_maps'])}")
        
        test_results["passed"].append("End-to-end pipeline")
        print("\n✅ End-to-end test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ End-to-end test failed: {e}")
        traceback.print_exc()
        test_results["failed"].append("End-to-end pipeline")
        return False


def test_example_scripts():
    """Test 7: Check if example scripts exist and are valid Python."""
    print_section("TEST 7: Example Scripts")
    
    example_files = [
        "example_tampering_usage.py",
        "integration_example.py"
    ]
    
    all_valid = True
    
    for filename in example_files:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    compile(f.read(), filename, 'exec')
                print(f"✓ {filename} is valid Python")
                test_results["passed"].append(f"Example: {filename}")
            except SyntaxError as e:
                print(f"✗ {filename} has syntax errors: {e}")
                test_results["failed"].append(f"Example: {filename}")
                all_valid = False
        else:
            print(f"⚠ {filename} not found")
            test_results["warnings"].append(f"Example: {filename} not found")
            all_valid = False
    
    if all_valid:
        print("\n✅ All example scripts are valid!")
    return all_valid


def generate_test_report():
    """Generate final test report."""
    print_section("TEST SUMMARY")
    
    total_passed = len(test_results["passed"])
    total_failed = len(test_results["failed"])
    total_warnings = len(test_results["warnings"])
    total_tests = total_passed + total_failed
    
    print(f"Passed:   {total_passed:3d} / {total_tests}")
    print(f"Failed:   {total_failed:3d} / {total_tests}")
    print(f"Warnings: {total_warnings:3d}")
    print()
    
    if total_failed == 0:
        print("✅ ALL TESTS PASSED! The module is ready to use.")
        success_rate = 100.0
    else:
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        print(f"⚠️  Some tests failed. Success rate: {success_rate:.1f}%")
    
    print()
    
    if test_results["failed"]:
        print("Failed tests:")
        for test in test_results["failed"]:
            print(f"  ✗ {test}")
        print()
    
    if test_results["warnings"]:
        print("Warnings:")
        for warning in test_results["warnings"]:
            print(f"  ⚠ {warning}")
        print()
    
    print("="*70)
    
    # Save report to file
    report_path = "test_results/test_report.txt"
    Path("test_results").mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("Document Tampering Localization - Test Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Tests:  {total_tests}\n")
        f.write(f"Passed:       {total_passed}\n")
        f.write(f"Failed:       {total_failed}\n")
        f.write(f"Warnings:     {total_warnings}\n")
        f.write(f"Success Rate: {success_rate:.1f}%\n\n")
        
        if test_results["passed"]:
            f.write("Passed Tests:\n")
            for test in test_results["passed"]:
                f.write(f"  ✓ {test}\n")
            f.write("\n")
        
        if test_results["failed"]:
            f.write("Failed Tests:\n")
            for test in test_results["failed"]:
                f.write(f"  ✗ {test}\n")
            f.write("\n")
        
        if test_results["warnings"]:
            f.write("Warnings:\n")
            for warning in test_results["warnings"]:
                f.write(f"  ⚠ {warning}\n")
    
    print(f"Test report saved to: {report_path}")
    
    return total_failed == 0


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("  DOCUMENT TAMPERING LOCALIZATION - TEST SUITE")
    print("="*70)
    print("\nThis will verify that the module is properly installed and functional.\n")
    
    # Run tests in order
    tests = [
        ("Dependencies", test_dependencies),
        ("Module Import", test_module_import),
        ("Detector Initialization", test_detector_initialization),
        ("Forensic Methods", test_forensic_methods),
        ("Fusion & Visualization", test_fusion_and_visualization),
        ("End-to-End Pipeline", test_end_to_end),
        ("Example Scripts", test_example_scripts)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Unexpected error in {test_name}: {e}")
            traceback.print_exc()
            results.append(False)
    
    # Generate report
    all_passed = generate_test_report()
    
    # Return exit code
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
