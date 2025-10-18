# 🎯 Quick Reference - Tampering Detection API

## API Endpoint

```
POST /detect-tampering
```

## Parameters

| Name                       | Type  | Required | Default | Description               |
| -------------------------- | ----- | -------- | ------- | ------------------------- |
| `file`                     | File  | ✅ Yes   | -       | Image to analyze          |
| `sensitivity`              | float | ❌ No    | 0.5     | Detection threshold (0-1) |
| `return_intermediate_maps` | bool  | ❌ No    | false   | Include forensic maps     |

## Quick Examples

### cURL

```bash
curl -X POST "http://localhost:8000/detect-tampering?sensitivity=0.5" \
  -F "file=@document.jpg"
```

### Python

```python
import requests
response = requests.post(
    "http://localhost:8000/detect-tampering",
    files={"file": open("document.jpg", "rb")},
    params={"sensitivity": 0.5}
)
result = response.json()
```

### JavaScript

```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const response = await fetch(
  "http://localhost:8000/detect-tampering?sensitivity=0.5",
  { method: "POST", body: formData }
);

const result = await response.json();
```

## Response Format

```json
{
  "filename": "document.jpg",
  "is_tampered": true,
  "probability": 0.87,
  "num_regions": 2,
  "heatmap": "iVBORw0KGg...",  // base64 PNG
  "mask": "iVBORw0KGg...",     // base64 PNG
  "tampered_regions": "iVB...",// base64 PNG
  "bboxes": [[x, y, w, h], ...]
}
```

## Display Images

```html
<img src="data:image/png;base64,{heatmap}" />
<img src="data:image/png;base64,{mask}" />
<img src="data:image/png;base64,{tampered_regions}" />
```

## Detection Focus

- 🎯 **Blur artifacts** (2x priority)
- 🎨 **Color inconsistencies** (2x priority)
- 🔄 **Copy-move forgery**
- 💡 **Lighting mismatches**

## Sensitivity Guide

| Value | Mode      | Use Case                             |
| ----- | --------- | ------------------------------------ |
| 0.6   | Strict    | Fewer false positives                |
| 0.5   | Balanced  | **Recommended**                      |
| 0.4   | Sensitive | Catch more, may have false positives |

## Status Codes

| Code | Meaning                         |
| ---- | ------------------------------- |
| 200  | Success                         |
| 400  | Invalid file type or parameters |
| 500  | Processing error                |
| 503  | Model not loaded                |

## Start Server

```bash
cd server
python api/main.py
```

## Test

```bash
# Health check
curl http://localhost:8000/health

# Test detection
curl -X POST "http://localhost:8000/detect-tampering" \
  -F "file=@test.jpg"
```

## Docs

```
http://localhost:8000/docs
```
