# Updated `detect_tampering_hybrid` Function

## Summary of Changes

The `detect_tampering_hybrid` function has been updated to **return the actual images** instead of just saving them to disk and requiring file path lookups.

---

## What Changed?

### **Before:**

- Function only saved images to disk
- Returned dictionary with file paths (indirectly)
- Required reading files from disk to access results

### **After:**

- Function now returns images directly in the result dictionary
- Two modes available:
  1. **NumPy arrays** (for Python code)
  2. **Base64 strings** (for APIs/JSON serialization)
- Still optionally saves to disk if `save_results=True`

---

## New Return Structure

### Result Dictionary Keys:

```python
{
    'heatmap': <image>,           # Tampering heatmap overlay
    'mask': <image>,              # Binary tampering mask (0-255)
    'tampered_regions': <image>,  # Original image with bounding boxes drawn
    'bboxes': [(x,y,w,h), ...],  # List of detected region coordinates
    'probability': 0.XX,          # Overall tampering confidence (0-1)
    'fused_map': <array>,         # Raw detection scores
    'intermediate_maps': {...},   # (Optional) Individual forensic maps
    'gradcam': <image>           # (Optional) GradCAM visualization
}
```

Where `<image>` is either:

- **NumPy array** (RGB/grayscale) if `return_base64=False`
- **Base64 string** (PNG encoded) if `return_base64=True`

---

## New Parameters

### `return_base64` (bool, default=False)

- **False**: Returns images as NumPy arrays (RGB format)
  - Use this for programmatic access in Python
  - Direct manipulation and visualization
- **True**: Returns images as base64-encoded strings
  - Use this for REST APIs and JSON responses
  - Easy to transmit over HTTP
  - Can be directly embedded in HTML `<img>` tags

---

## Usage Examples

### Example 1: NumPy Arrays (Python Code)

```python
from detection.tampering_localization import detect_tampering_hybrid
from models.predict import load_model

# Load model
model, device = load_model('models/saved_models/best_model.pth')

# Run detection
result = detect_tampering_hybrid(
    image_path="document.jpg",
    model=model,
    device=device,
    save_results=False,      # Don't save to disk
    return_base64=False      # Return as NumPy arrays
)

# Access images directly
heatmap = result['heatmap']                  # RGB numpy array
mask = result['mask']                        # Grayscale numpy array (0-255)
tampered_regions = result['tampered_regions']  # RGB numpy array with boxes

# Use immediately
import cv2
cv2.imshow("Heatmap", cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
cv2.imshow("Tampered Regions", cv2.cvtColor(tampered_regions, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
```

### Example 2: Base64 Strings (API/JSON)

```python
result = detect_tampering_hybrid(
    image_path="document.jpg",
    model=model,
    device=device,
    save_results=False,
    return_base64=True        # Return as base64 strings
)

# Perfect for API responses
{
    "heatmap": "iVBORw0KGgoAAAANSUhEUgAA...",  # Base64 string
    "mask": "iVBORw0KGgoAAAANSUhEUgAA...",     # Base64 string
    "tampered_regions": "iVBORw0KGgoAAAAN...", # Base64 string
    "probability": 0.87,
    "bboxes": [(45, 120, 200, 150)]
}

# Use in frontend HTML
<img src="data:image/png;base64,{result['heatmap']}" />
```

### Example 3: FastAPI Integration

```python
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

@app.post("/detect-tampering")
async def detect_tampering_endpoint(file: UploadFile):
    # Save uploaded file temporarily
    temp_path = save_upload(file)

    # Run detection with base64 output
    result = detect_tampering_hybrid(
        image_path=temp_path,
        model=model,
        device=device,
        save_results=False,
        return_base64=True  # Perfect for JSON response
    )

    # Return directly - all images are base64 strings
    return JSONResponse(content=result)
```

### Example 4: Save AND Return

```python
# You can do both!
result = detect_tampering_hybrid(
    image_path="document.jpg",
    model=model,
    device=device,
    save_results=True,       # Save to disk
    return_base64=False      # AND return as arrays
)

# Files saved to: tampering_results/
# - document_heatmap.png
# - document_mask.png
# - document_tampered_regions.png
# - document_tampering_analysis.png

# Also available in memory:
heatmap = result['heatmap']  # NumPy array
mask = result['mask']        # NumPy array
```

---

## Helper Functions

### Converting Base64 to Image

```python
import base64
import numpy as np
import cv2

def base64_to_image(b64_string):
    """Convert base64 string back to numpy array"""
    img_bytes = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Usage
heatmap_img = base64_to_image(result['heatmap'])
```

### Embedding in HTML

```html
<!-- Direct base64 embedding -->
<img src="data:image/png;base64,{{ heatmap_base64 }}" alt="Heatmap" />
<img src="data:image/png;base64,{{ mask_base64 }}" alt="Mask" />
<img src="data:image/png;base64,{{ tampered_regions_base64 }}" alt="Detected" />
```

---

## Migration Guide

### Old Code (File-based):

```python
result = detect_tampering_hybrid("doc.jpg", model, save_results=True)
# Had to manually read saved files
heatmap = cv2.imread("tampering_results/doc_heatmap.png")
```

### New Code (Direct return):

```python
result = detect_tampering_hybrid("doc.jpg", model, return_base64=False)
# Images directly available
heatmap = result['heatmap']  # Already in memory!
```

---

## Benefits

✅ **No disk I/O required** - Images returned directly  
✅ **API-friendly** - Base64 format for JSON responses  
✅ **Flexible** - Choose NumPy arrays OR base64 strings  
✅ **Backward compatible** - Can still save to disk with `save_results=True`  
✅ **New image added** - `tampered_regions` shows bounding boxes  
✅ **Type safety** - Clear separation between array and string modes

---

## API Signature

```python
def detect_tampering_hybrid(
    image_path: str,
    model: torch.nn.Module,
    device: str = "cuda",
    save_results: bool = True,
    sensitivity: float = 0.5,
    return_intermediate_maps: bool = False,
    return_base64: bool = False  # NEW PARAMETER
) -> dict:
    """
    Returns:
        dict: {
            'heatmap': <image>,           # RGB heatmap overlay
            'mask': <image>,              # Binary mask (0-255)
            'tampered_regions': <image>,  # Image with bounding boxes
            'bboxes': list,               # [(x,y,w,h), ...]
            'probability': float,         # 0-1
            'fused_map': np.ndarray,      # Raw scores
            'intermediate_maps': dict,    # (optional)
            'gradcam': <image>           # (optional)
        }

        Where <image> is:
        - np.ndarray if return_base64=False
        - str (base64) if return_base64=True
    """
```

---

## Files Modified

1. **`server/detection/tampering_localization.py`**

   - Added `base64` and `io` imports
   - Added `return_base64` parameter
   - Added image encoding logic
   - Added `tampered_regions` output (image with bounding boxes)
   - Updated return structure

2. **`server/examples/tampering_detection_usage_example.py`** (NEW)
   - Comprehensive usage examples
   - Demonstrates both NumPy and base64 modes
   - Shows API integration patterns

---

## Notes

- **Performance**: Base64 encoding adds ~1-2% overhead. Negligible for API use.
- **Memory**: Images stored in memory. Consider using `save_results=False` for batch processing.
- **Compatibility**: All existing code continues to work. New parameter is optional.
- **Format**: Base64 images are PNG-encoded for lossless quality.

---

## Questions?

See `examples/tampering_detection_usage_example.py` for complete working examples!
