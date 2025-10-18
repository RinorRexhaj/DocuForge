# Tampering Detection Pipeline - Visual Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT: Document Image                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
        ┌───────▼──────┐        ┌──────▼───────┐
        │  Deep Learning│        │   Classical   │
        │   (Grad-CAM)  │        │   Forensics   │
        │   Weight: 20% │        │  Weight: 80%  │
        └───────┬───────┘        └──────┬────────┘
                │                       │
                │               ┌───────┴────────┐
                │               │                │
                │      ┌────────▼─────┐  ┌───────▼────────┐
                │      │ HIGH PRIORITY │  │   SECONDARY    │
                │      │  (2x weight)  │  │  (1x weight)   │
                │      └───────┬───────┘  └───────┬────────┘
                │              │                  │
                │      ┌───────┴───────┐         │
                │      │               │         │
                │  ┌───▼────┐   ┌─────▼─────┐   │
                │  │ BLUR   │   │  COLOR    │   │
                │  │(x2)🎯 │   │   (x2)🎨  │   │
                │  └────────┘   └───────────┘   │
                │                               │
                │  ┌────────────┐  ┌─────────┐ │
                │  │COPY-MOVE🔄│  │LIGHTING💡│ │
                │  └────────────┘  └─────────┘ │
                │                               │
                │      ┌──────────────┐  ┌──────▼────┐
                │      │              │  │   ELA     │
                │      │              │  ├───────────┤
                │      │              │  │   Noise   │
                │      │              │  ├───────────┤
                │      │              │  │   Edge    │
                │      │              │  └───────────┘
                │      │              │
        ┌───────▼──────▼──────────────▼──────┐
        │         FUSION LAYER                │
        │  Weighted Combination of All Maps   │
        └──────────────┬──────────────────────┘
                       │
        ┌──────────────▼──────────────────────┐
        │      TAMPERING HEATMAP [0, 1]       │
        └──────────────┬──────────────────────┘
                       │
        ┌──────────────▼──────────────────────┐
        │   Binary Mask (threshold=sensitivity)│
        └──────────────┬──────────────────────┘
                       │
        ┌──────────────▼──────────────────────┐
        │   Bounding Box Extraction            │
        └──────────────┬──────────────────────┘
                       │
        ┌──────────────▼──────────────────────┐
        │          OUTPUT RESULTS              │
        │  - Heatmap (base64 or numpy)         │
        │  - Mask (base64 or numpy)            │
        │  - Tampered Regions with boxes       │
        │  - Bounding boxes coordinates        │
        │  - Tampering probability score       │
        │  - (Optional) Intermediate maps      │
        └──────────────────────────────────────┘
```

## Detection Method Details

### 🎯 BLUR DETECTION (2x weight)

```
Input Image → Grayscale → Laplacian Variance
                        → Motion Blur Detection (16 angles)
                        → Text Overlay Artifacts
                        → Splice Boundary Detection
                        → Combined Blur Map (weighted)
```

**Targets:**

- Gaussian blur (hiding edits)
- Motion blur (fake movement)
- Selective blur (obscuring details)
- Smudge tool artifacts

---

### 🎨 COLOR INCONSISTENCY (2x weight)

```
Input Image → LAB Color Space → Local Color Stats
                              → Variance Analysis
                              → Global Median Comparison
                              → Anomaly Highlighting
                              → Color Inconsistency Map
```

**Targets:**

- Different color profiles
- Spliced elements from different sources
- Color temperature mismatches
- Saturation/hue discrepancies

---

### 🔄 COPY-MOVE DETECTION

```
Input Image → Grayscale → ORB Keypoint Detection
                        → Self-Matching (excluding identity)
                        → Distance Filtering (>20px)
                        → Ratio Test (threshold=0.75)
                        → Heatmap Generation
                        → Copy-Move Map
```

**Targets:**

- Clone stamp tool usage
- Duplicated signatures/stamps
- Repeated patterns
- Cloning artifacts

---

### 💡 ILLUMINATION INCONSISTENCY

```
Input Image → HSV Color Space → Value Channel
                              → Sobel Gradients (x, y)
                              → Gradient Magnitude
                              → Local Statistics
                              → Illumination Map
```

**Targets:**

- Inconsistent light directions
- Shadow angle mismatches
- Brightness discontinuities
- Mixed lighting sources

---

### 📊 SECONDARY METHODS (standard weight)

#### ELA (Error Level Analysis)

```
Input Image → JPEG Re-compression (quality=90)
           → Difference Calculation
           → Grayscale Conversion
           → Contrast Enhancement
           → ELA Map
```

#### Noise Inconsistency

```
Input Image → Local Std Deviation
           → Median Filtering
           → Inconsistency Calculation
           → Noise Map
```

#### Edge Artifacts

```
Input Image → Laplacian Filter
           → Gaussian Smoothing
           → Local Statistics
           → Edge Map
```

---

## Weight Distribution

```
Total Weight: 100%
├─ 20% Grad-CAM (Deep Learning)
└─ 80% Classical Forensics
    ├─ High Priority (58%)
    │   ├─ 16% Blur (counted 2x)
    │   ├─ 16% Color (counted 2x)
    │   ├─ 13% Copy-Move
    │   └─ 13% Illumination
    └─ Secondary (22%)
        ├─ 7% ELA
        ├─ 7% Noise
        └─ 7% Edge
```

---

## Performance Metrics

| Metric           | Value                              |
| ---------------- | ---------------------------------- |
| Processing Time  | ~2-5 seconds per image             |
| Memory Usage     | ~500MB-1GB (depends on image size) |
| GPU Acceleration | Supported (Grad-CAM only)          |
| CPU Fallback     | Available                          |

---

## Output Format

### Standard Output (return_base64=False):

```python
{
    'heatmap': np.ndarray,          # RGB (H, W, 3)
    'mask': np.ndarray,             # Grayscale (H, W)
    'tampered_regions': np.ndarray, # RGB (H, W, 3)
    'bboxes': [(x,y,w,h), ...],    # List of tuples
    'probability': float,           # 0-1
    'fused_map': np.ndarray        # Float (H, W)
}
```

### API Output (return_base64=True):

```python
{
    'heatmap': "iVBORw0KGg...",        # Base64 PNG
    'mask': "iVBORw0KGg...",           # Base64 PNG
    'tampered_regions': "iVBORw0...",  # Base64 PNG
    'bboxes': [(x,y,w,h), ...],       # List of tuples
    'probability': 0.87,               # Float
    'fused_map': [[...], [...], ...]  # 2D list
}
```

---

## Usage Example

```python
from detection.tampering_localization import detect_tampering_hybrid
from models.predict import load_model

# Load model
model, device = load_model('models/saved_models/best_model.pth')

# Detect tampering
result = detect_tampering_hybrid(
    image_path="document.jpg",
    model=model,
    device=device,
    save_results=False,
    sensitivity=0.5,
    return_base64=True,  # For API
    return_intermediate_maps=True  # Get all detection maps
)

# Results
print(f"Tampered: {result['is_tampered']}")
print(f"Probability: {result['probability']:.1%}")
print(f"Regions: {len(result['bboxes'])}")

# Access specific detection maps
if 'intermediate_maps' in result:
    blur_map = result['intermediate_maps']['blur_combined']
    color_map = result['intermediate_maps']['color_inconsistency']
    copymove_map = result['intermediate_maps']['copymove']
    illumination_map = result['intermediate_maps']['illumination']
```

---

## API Integration

```python
# FastAPI endpoint
@app.post("/detect-tampering")
async def detect_tampering(file: UploadFile, sensitivity: float = 0.5):
    result = detect_tampering_hybrid(
        image_path=temp_file_path,
        model=model,
        device=device,
        save_results=False,
        sensitivity=sensitivity,
        return_base64=True  # Perfect for JSON response
    )

    return JSONResponse(content={
        'heatmap': result['heatmap'],
        'mask': result['mask'],
        'tampered_regions': result['tampered_regions'],
        'probability': result['probability'],
        'num_regions': len(result['bboxes']),
        'bboxes': result['bboxes']
    })
```

---

Perfect for detecting:

- ✨ Blur-based document forgeries
- 🎨 Photo splicing and replacement
- 🔄 Clone stamp and duplication
- 💡 Inconsistent lighting conditions
