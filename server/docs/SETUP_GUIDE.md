# 🚀 Quick Setup Guide - Document Tampering Localization

## 📋 Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- Trained PyTorch model for document forgery detection

---

## ⚡ Installation (3 Steps)

### Step 1: Install Dependencies

```powershell
cd c:\Users\PC\Desktop\Apps\DocuForge\server
pip install -r requirements_tampering.txt
```

This will install:

- OpenCV (cv2)
- PyTorch & torchvision
- NumPy
- scikit-image
- SciPy
- Matplotlib
- PIL/Pillow
- pytorch-grad-cam (optional)

### Step 2: Verify Installation

```powershell
python test_tampering_module.py
```

This test script will:

- ✅ Check all dependencies
- ✅ Test module imports
- ✅ Verify forensic methods
- ✅ Run end-to-end pipeline
- ✅ Generate test report

Expected output: `ALL TESTS PASSED!`

### Step 3: Run Example

```powershell
python example_tampering_usage.py
```

---

## 🎯 Quick Start (30 seconds)

### Basic Detection

```python
from tampering_localization import detect_tampering_hybrid
import torch

# Load your model
model = torch.load("saved_models/best_model.pth")

# Detect tampering
result = detect_tampering_hybrid(
    image_path="document.jpg",
    model=model,
    device="cuda"  # or "cpu"
)

# View results
print(f"Tampering probability: {result['probability']:.2%}")
print(f"Detected regions: {len(result['bboxes'])}")
```

### View Visualizations

Results are automatically saved to `tampering_results/`:

- `{filename}_tampering_analysis.png` - 4-panel visualization
- `{filename}_heatmap.png` - Heatmap overlay
- `{filename}_mask.png` - Binary mask

---

## 📁 Files Overview

| File                            | Purpose                  | Size                |
| ------------------------------- | ------------------------ | ------------------- |
| `tampering_localization.py`     | Main module (800+ lines) | Core implementation |
| `requirements_tampering.txt`    | Dependencies             | Installation        |
| `example_tampering_usage.py`    | Usage examples           | Learning            |
| `integration_example.py`        | DocuForge integration    | Integration         |
| `test_tampering_module.py`      | Test suite               | Verification        |
| `TAMPERING_DETECTION_README.md` | Full documentation       | Reference           |
| `ARCHITECTURE_DIAGRAM.txt`      | System architecture      | Understanding       |
| `IMPLEMENTATION_SUMMARY.txt`    | Complete summary         | Overview            |

---

## 🎨 Features

### Deep Learning

- ✅ Grad-CAM attention maps
- ✅ Automatic layer detection
- ✅ ResNet/VGG/Custom model support

### Classical Forensics (5 techniques)

- ✅ Error Level Analysis (ELA)
- ✅ Noise Inconsistency Detection
- ✅ Edge Artifact Analysis
- ✅ JPEG Block Artifact Detection
- ✅ Copy-Move Detection (ORB)

### Outputs

- ✅ Tampering heatmap overlay
- ✅ Binary mask
- ✅ Bounding boxes
- ✅ Probability score [0-1]
- ✅ Individual forensic maps (optional)

---

## 🧪 Testing Your Setup

### 1. Quick Test

```powershell
python -c "from tampering_localization import detect_tampering_hybrid; print('✓ Module loaded successfully')"
```

### 2. Full Test Suite

```powershell
python test_tampering_module.py
```

### 3. Test with Your Model

```python
import torch
from tampering_localization import detect_tampering_hybrid

# Load your trained model
model = torch.load("saved_models/best_model.pth")
model.eval()

# Test image paths (update these)
test_images = [
    "dataset/test/authentic/sample1.jpg",
    "dataset/test/forged/sample2.jpg"
]

for img_path in test_images:
    result = detect_tampering_hybrid(img_path, model)
    print(f"{img_path}: {result['probability']:.2%}")
```

---

## 🎛️ Configuration Options

### Adjust Sensitivity

```python
# High sensitivity (catch subtle tampering)
result = detect_tampering_hybrid(
    "document.jpg",
    model,
    sensitivity=0.3
)

# Low sensitivity (only obvious tampering)
result = detect_tampering_hybrid(
    "document.jpg",
    model,
    sensitivity=0.7
)
```

### Change Fusion Weights

Edit `tampering_localization.py` line ~340:

```python
# Default: 40% Grad-CAM, 60% Classical
weights={'gradcam': 0.4, 'classical': 0.6}

# More weight on deep learning
weights={'gradcam': 0.7, 'classical': 0.3}
```

### Custom Output Directory

```python
from tampering_localization import DocumentTamperingDetector

detector = DocumentTamperingDetector(
    model=model,
    device="cuda",
    output_dir="my_custom_results/"
)
```

---

## 🐛 Troubleshooting

### Issue: Import errors

**Solution:**

```powershell
pip install -r requirements_tampering.txt --upgrade
```

### Issue: CUDA out of memory

**Solution:**

```python
result = detect_tampering_hybrid(
    "document.jpg",
    model,
    device="cpu"  # Use CPU instead
)
```

### Issue: pytorch-grad-cam not found

**Solution:**

```powershell
pip install pytorch-grad-cam
```

Or ignore - the module has a built-in fallback implementation.

### Issue: No regions detected

**Solution:**

```python
# Lower the sensitivity threshold
result = detect_tampering_hybrid(
    "document.jpg",
    model,
    sensitivity=0.3  # Default is 0.5
)
```

### Issue: Model compatibility

**Solution:**
The module works with any PyTorch model that:

1. Accepts `(1, 3, 224, 224)` input tensors
2. Has convolutional layers
3. Returns forgery predictions

If your model is different, adjust the preprocessing in `preprocess_image()`.

---

## 📊 Expected Output

### Console Output

```
🔍 Analyzing document: sample_passport.jpg
✓ Image loaded and preprocessed
🧠 Computing Grad-CAM heatmap...
✓ Grad-CAM completed
🔬 Running classical forensic analysis...
  ✓ ELA completed
  ✓ Noise analysis completed
  ✓ Edge artifact detection completed
  ✓ JPEG block analysis completed
  ✓ Copy-move detection completed
🔗 Fusing detection maps...
✓ Fusion completed
📊 Generating tampering mask...
✓ Found 2 suspicious region(s)
📈 Overall tampering probability: 68.34%
🎨 Creating visualizations...
✓ Visualization saved to: tampering_results/sample_passport_tampering_analysis.png
✓ Results saved to: tampering_results
✅ Tampering detection completed!
```

### File Output

```
tampering_results/
├── sample_passport_tampering_analysis.png
├── sample_passport_heatmap.png
├── sample_passport_mask.png
├── sample_passport_ela.png (if return_intermediate_maps=True)
├── sample_passport_noise.png
├── sample_passport_edge.png
└── sample_passport_jpeg.png
```

---

## 🚀 Next Steps

1. **Test with your documents:**

   ```python
   result = detect_tampering_hybrid("your_document.jpg", model)
   ```

2. **Integrate with your API:**
   See `integration_example.py` for guidance

3. **Batch process documents:**

   ```python
   from tampering_localization import batch_detect_tampering
   df = batch_detect_tampering(image_list, model)
   ```

4. **Customize for your needs:**
   - Adjust fusion weights
   - Modify sensitivity
   - Add custom forensic techniques

---

## 📚 Documentation

- **Full Documentation:** `TAMPERING_DETECTION_README.md`
- **Architecture Details:** `ARCHITECTURE_DIAGRAM.txt`
- **Implementation Notes:** `IMPLEMENTATION_SUMMARY.txt`
- **Code Examples:** `example_tampering_usage.py`
- **Integration Guide:** `integration_example.py`

---

## 💡 Pro Tips

1. **GPU Acceleration:** Always use CUDA if available for 10x speed boost
2. **Batch Processing:** Process multiple documents at once for efficiency
3. **Sensitivity Tuning:** Start with 0.5, adjust based on your requirements
4. **Intermediate Maps:** Enable for detailed forensic analysis
5. **Output Management:** Disable `save_results` for API use to save disk space

---

## ✅ Checklist

- [ ] Dependencies installed (`pip install -r requirements_tampering.txt`)
- [ ] Test suite passed (`python test_tampering_module.py`)
- [ ] Verified with your model
- [ ] Tested with sample documents
- [ ] Reviewed output visualizations
- [ ] Read documentation (`TAMPERING_DETECTION_README.md`)
- [ ] Ready for integration!

---

## 🎉 You're Ready!

The tampering localization module is now set up and ready to detect document forgeries!

```python
from tampering_localization import detect_tampering_hybrid
import torch

model = torch.load("saved_models/best_model.pth")
result = detect_tampering_hybrid("document.jpg", model)

print(f"🎯 Analysis complete!")
print(f"   Probability: {result['probability']:.2%}")
print(f"   Regions: {len(result['bboxes'])}")
print(f"   Status: {'TAMPERED ⚠️' if result['probability'] > 0.5 else 'AUTHENTIC ✅'}")
```

**Happy tampering detection! 🔍🛡️**
