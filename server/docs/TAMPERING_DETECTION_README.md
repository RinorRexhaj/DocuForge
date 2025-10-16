# 🔍 Document Tampering Localization System

## Overview

This module provides a **hybrid approach** for detecting and localizing tampered regions in document images (passports, IDs, certificates, forms) by combining:

1. **Deep Learning**: Grad-CAM attention from a pre-trained CNN forgery detector
2. **Classical Forensics**: Multiple traditional image forensic techniques
3. **Intelligent Fusion**: Weighted combination for robust detection

---

## 🎯 Features

### Deep Learning Branch

- ✅ **Grad-CAM/Grad-CAM++** visualization from ResNet50 or custom CNN
- ✅ Automatic target layer detection
- ✅ Custom Grad-CAM fallback implementation

### Classical Forensics Branch

- ✅ **Error Level Analysis (ELA)** - JPEG compression inconsistencies
- ✅ **Noise Inconsistency Map** - Local variance analysis
- ✅ **Edge Artifact Detection** - High-frequency Laplacian filtering
- ✅ **JPEG Block Artifacts** - DCT coefficient analysis (8×8 blocks)
- ✅ **Copy-Move Detection** - ORB keypoint matching (optional)

### Fusion & Visualization

- ✅ Weighted fusion (60% classical + 40% Grad-CAM)
- ✅ Heatmap overlay on original image
- ✅ Binary tampering mask
- ✅ Bounding box extraction
- ✅ Comprehensive visualization suite

---

## 📦 Installation

### 1. Install Dependencies

```bash
cd server
pip install -r requirements_tampering.txt
```

### 2. Verify Installation

```python
import cv2
import torch
from tampering_localization import detect_tampering_hybrid
print("✓ All dependencies installed successfully!")
```

---

## 🚀 Quick Start

### Basic Usage

```python
from tampering_localization import detect_tampering_hybrid
import torch

# Load your pre-trained model
model = torch.load("saved_models/best_model.pth")
model.eval()

# Detect tampering
result = detect_tampering_hybrid(
    image_path="sample_passport.jpg",
    model=model,
    device="cuda",
    save_results=True
)

# Access results
print(f"Tampering probability: {result['probability']:.2%}")
print(f"Detected regions: {len(result['bboxes'])}")

# Display visualizations
import cv2
cv2.imshow("Heatmap", result["heatmap"])
cv2.imshow("Mask", result["mask"] * 255)
cv2.waitKey(0)
```

---

## 📖 API Reference

### Main Function

```python
def detect_tampering_hybrid(
    image_path: str,
    model: torch.nn.Module,
    device: str = "cuda",
    save_results: bool = True,
    sensitivity: float = 0.5,
    return_intermediate_maps: bool = False
) -> dict
```

#### Parameters

| Parameter                  | Type              | Default    | Description                                 |
| -------------------------- | ----------------- | ---------- | ------------------------------------------- |
| `image_path`               | `str`             | _required_ | Path to document image (PNG, JPG, JPEG)     |
| `model`                    | `torch.nn.Module` | _required_ | Pre-trained PyTorch forgery detection model |
| `device`                   | `str`             | `"cuda"`   | Device for inference (`"cuda"` or `"cpu"`)  |
| `save_results`             | `bool`            | `True`     | Whether to save visualizations to disk      |
| `sensitivity`              | `float`           | `0.5`      | Threshold for binary mask (0-1)             |
| `return_intermediate_maps` | `bool`            | `False`    | Include individual forensic maps in output  |

#### Returns

Dictionary with the following keys:

```python
{
    "heatmap": np.ndarray,        # RGB overlay image
    "mask": np.ndarray,            # Binary mask (0 or 1)
    "bboxes": List[Tuple],         # [(x, y, w, h), ...]
    "probability": float,          # Overall tampering score [0, 1]
    "fused_map": np.ndarray,       # Raw fused heatmap
    "intermediate_maps": dict,     # (optional) Individual forensic maps
    "gradcam": np.ndarray          # (optional) Grad-CAM heatmap
}
```

---

## 🎨 Example Usage Scenarios

### Example 1: Single Document Analysis

```python
result = detect_tampering_hybrid(
    "documents/passport_001.jpg",
    model=resnet_model,
    device="cuda",
    sensitivity=0.5
)

if result['probability'] > 0.5:
    print("⚠️  ALERT: Document appears tampered!")
    print(f"Confidence: {result['probability']:.1%}")
    print(f"Tampered regions: {len(result['bboxes'])}")
else:
    print("✓ Document appears authentic")
```

### Example 2: Batch Processing

```python
from tampering_localization import batch_detect_tampering

image_list = [
    "docs/id_001.jpg",
    "docs/id_002.jpg",
    "docs/passport_001.jpg"
]

results_df = batch_detect_tampering(
    image_paths=image_list,
    model=model,
    output_csv="results.csv"
)

print(results_df)
```

### Example 3: Custom Sensitivity

```python
# High sensitivity (detect subtle tampering)
result_sensitive = detect_tampering_hybrid(
    "document.jpg",
    model=model,
    sensitivity=0.3  # Lower threshold = more sensitive
)

# Low sensitivity (only obvious tampering)
result_strict = detect_tampering_hybrid(
    "document.jpg",
    model=model,
    sensitivity=0.7  # Higher threshold = less sensitive
)
```

### Example 4: API Integration

```python
import json

result = detect_tampering_hybrid(
    "uploaded_document.jpg",
    model=model,
    save_results=True
)

# Create API response
response = {
    "is_tampered": result['probability'] > 0.5,
    "confidence": float(result['probability']),
    "risk_level": "high" if result['probability'] > 0.7 else "medium",
    "regions": [
        {"x": x, "y": y, "width": w, "height": h}
        for x, y, w, h in result['bboxes']
    ]
}

print(json.dumps(response, indent=2))
```

### Example 5: Forensic Analysis

```python
# Get detailed forensic breakdown
result = detect_tampering_hybrid(
    "suspicious_doc.jpg",
    model=model,
    return_intermediate_maps=True
)

# Access individual forensic maps
ela_map = result['intermediate_maps']['ela']
noise_map = result['intermediate_maps']['noise']
edge_map = result['intermediate_maps']['edge']
jpeg_map = result['intermediate_maps']['jpeg']

# Visualize each separately
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].imshow(ela_map, cmap='hot')
axes[0, 0].set_title('Error Level Analysis')
axes[0, 1].imshow(noise_map, cmap='hot')
axes[0, 1].set_title('Noise Inconsistency')
axes[1, 0].imshow(edge_map, cmap='hot')
axes[1, 0].set_title('Edge Artifacts')
axes[1, 1].imshow(jpeg_map, cmap='hot')
axes[1, 1].set_title('JPEG Block Artifacts')
plt.show()
```

---

## 🔧 Configuration

### Adjusting Fusion Weights

Modify the fusion strategy in `tampering_localization.py`:

```python
# Default: 40% Grad-CAM, 60% classical
fused_map = detector.combine_maps_fusion(
    gradcam_map,
    classical_maps,
    weights={'gradcam': 0.4, 'classical': 0.6}
)

# More weight on deep learning
fused_map = detector.combine_maps_fusion(
    gradcam_map,
    classical_maps,
    weights={'gradcam': 0.7, 'classical': 0.3}
)
```

### Custom Output Directory

```python
from tampering_localization import DocumentTamperingDetector

detector = DocumentTamperingDetector(
    model=model,
    device="cuda",
    output_dir="custom_results/"
)
```

---

## 📊 Output Files

When `save_results=True`, the following files are created in `tampering_results/`:

```
tampering_results/
├── {filename}_tampering_analysis.png    # 4-panel visualization
├── {filename}_heatmap.png               # Heatmap overlay
├── {filename}_mask.png                  # Binary mask
├── {filename}_ela.png                   # (optional) ELA map
├── {filename}_noise.png                 # (optional) Noise map
├── {filename}_edge.png                  # (optional) Edge map
└── {filename}_jpeg.png                  # (optional) JPEG map
```

---

## 🧪 Testing

Run the example script:

```bash
python example_tampering_usage.py
```

Or test with your own images:

```python
from tampering_localization import detect_tampering_hybrid
import torch

model = torch.load("saved_models/best_model.pth")
result = detect_tampering_hybrid("your_document.jpg", model)
```

---

## 🎯 Performance Tips

### GPU Acceleration

```python
# Check CUDA availability
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

result = detect_tampering_hybrid(
    "document.jpg",
    model=model,
    device=device
)
```

### Optimize for Speed

```python
# Disable intermediate maps to save computation
result = detect_tampering_hybrid(
    "document.jpg",
    model=model,
    return_intermediate_maps=False,  # Faster
    save_results=False               # Even faster
)
```

### Batch Processing Optimization

```python
# Process multiple images efficiently
from pathlib import Path

image_paths = list(Path("documents/").glob("*.jpg"))

for img_path in image_paths:
    result = detect_tampering_hybrid(
        str(img_path),
        model=model,
        save_results=True
    )
    print(f"{img_path.name}: {result['probability']:.2%}")
```

---

## 🔍 Understanding Results

### Probability Interpretation

| Probability Range | Interpretation    | Action                       |
| ----------------- | ----------------- | ---------------------------- |
| 0.0 - 0.3         | Likely authentic  | ✅ Accept document           |
| 0.3 - 0.5         | Uncertain         | ⚠️ Manual review recommended |
| 0.5 - 0.7         | Likely tampered   | 🚨 Flag for investigation    |
| 0.7 - 1.0         | Highly suspicious | 🚫 Reject document           |

### Bounding Box Format

```python
for x, y, w, h in result['bboxes']:
    # x, y: Top-left corner coordinates
    # w, h: Width and height of suspicious region
    print(f"Region at ({x}, {y}) with size {w}×{h}")
```

---

## 🐛 Troubleshooting

### Issue: "pytorch-grad-cam not installed"

```bash
pip install pytorch-grad-cam
```

Or use the built-in custom Grad-CAM (automatic fallback).

### Issue: CUDA out of memory

```python
# Use CPU instead
result = detect_tampering_hybrid(
    "large_document.jpg",
    model=model,
    device="cpu"
)
```

### Issue: No regions detected

Try adjusting sensitivity:

```python
result = detect_tampering_hybrid(
    "document.jpg",
    model=model,
    sensitivity=0.3  # Lower = more sensitive
)
```

---

## 📚 Technical Details

### Detection Pipeline

```
Input Image
    ↓
┌─────────────────────────────────────────┐
│  1. Preprocessing                       │
│     - Resize to 224×224                 │
│     - Normalize (ImageNet stats)        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────┬───────────────────┐
│  2a. Deep Learning  │  2b. Classical    │
│      - Grad-CAM     │      Forensics    │
│      - ResNet50     │      - ELA        │
│                     │      - Noise      │
│                     │      - Edges      │
│                     │      - JPEG       │
└─────────────────────┴───────────────────┘
    ↓                        ↓
    └────────────┬───────────┘
                 ↓
┌─────────────────────────────────────────┐
│  3. Fusion Layer                        │
│     Weighted: 40% DL + 60% Classical    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  4. Post-processing                     │
│     - Thresholding                      │
│     - Morphological ops                 │
│     - Contour extraction                │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  5. Output Generation                   │
│     - Heatmap overlay                   │
│     - Binary mask                       │
│     - Bounding boxes                    │
│     - Probability score                 │
└─────────────────────────────────────────┘
```

### Algorithm Details

1. **Grad-CAM**: Visualizes CNN decision-making via gradient-weighted class activation mapping
2. **ELA**: Detects compression inconsistencies through re-compression analysis
3. **Noise Analysis**: Identifies unnatural noise patterns via local variance
4. **Edge Artifacts**: Detects splicing via high-frequency discontinuities
5. **JPEG Blocks**: Analyzes 8×8 DCT coefficient patterns

---

## 📝 Citation

If you use this module in your research, please cite:

```bibtex
@software{docuforge_tampering,
  title={Document Tampering Localization: Hybrid Deep Learning and Forensics},
  author={DocuForge Team},
  year={2025},
  url={https://github.com/yourusername/DocuForge}
}
```

---

## 📄 License

This module is part of the DocuForge project. See LICENSE file for details.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional forensic techniques
- Faster inference methods
- Support for more document types
- Integration with other models

---

## 📞 Support

For issues or questions:

1. Check this README
2. Review `example_tampering_usage.py`
3. Open an issue on GitHub

---

**Built with ❤️ for document security and forensic analysis**
