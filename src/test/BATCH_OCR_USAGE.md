# Batch OCR API Usage Guide

This guide shows how to use the batch OCR API with YOLO object detection for processing multiple wine bottles in a single image.

## Overview

The batch OCR workflow consists of:
1. **YOLO Detection** - Detect wine bottles and get bounding boxes
2. **Batch OCR** - Send all bounding boxes in a single API call
3. **Results** - Get OCR text associated with each detected bottle

## Quick Start

### Option 1: Complete Pipeline (Recommended)

Run the full YOLO + OCR pipeline in one command:

```bash
python src/test/yolo_ocr_pipeline.py --image path/to/wine_bottles.jpg
```

This will:
- Run YOLO detection to find bottles
- Send all regions to batch OCR API
- Display text extracted from each bottle

Save results to JSON:
```bash
python src/test/yolo_ocr_pipeline.py --image wine_shelf.jpg --save-results output.json
```

### Option 2: Step-by-Step Workflow

#### Step 1: Run YOLO Detection

```bash
python src/test/run_yolo_detection.py --image wine_bottles.jpg --output boundaries.json --visualize
```

This creates:
- `boundaries.json` - Bounding boxes in API format
- `wine_bottles_yolo_detection.jpg` - Annotated image (if --visualize used)

#### Step 2: Run Batch OCR

```bash
python src/test/test_batch_api.py --image wine_bottles.jpg --boundaries boundaries.json
```

## API Endpoint Details

### POST /api/v1/batch

**Request:**
```json
{
  "image": "data:image/jpeg;base64,...",
  "boundaries": [
    {
      "x": 100,
      "y": 50,
      "width": 200,
      "height": 350,
      "label": "bottle_1"
    },
    {
      "x": 350,
      "y": 60,
      "width": 200,
      "height": 350,
      "label": "bottle_2"
    }
  ],
  "confidence": 0.25,
  "languages": ["en", "fr", "it", "es"],
  "return_annotated": false
}
```

**Response:**
```json
{
  "success": true,
  "total_regions": 2,
  "successful_regions": 2,
  "languages_used": ["en", "fr", "it", "es"],
  "confidence_threshold": 0.25,
  "results": [
    {
      "region_index": 0,
      "label": "bottle_1",
      "bbox": {"x": 100, "y": 50, "width": 200, "height": 350},
      "success": true,
      "total_text_blocks": 3,
      "all_text": "Chateau Margaux 2015",
      "avg_confidence": 0.89,
      "detections": [
        {"text": "Chateau", "confidence": 0.92, "bbox": [[...]], "position": [150, 100]},
        {"text": "Margaux", "confidence": 0.88, "bbox": [[...]], "position": [150, 150]},
        {"text": "2015", "confidence": 0.87, "bbox": [[...]], "position": [150, 200]}
      ]
    },
    {
      "region_index": 1,
      "label": "bottle_2",
      "bbox": {"x": 350, "y": 60, "width": 200, "height": 350},
      "success": true,
      "total_text_blocks": 2,
      "all_text": "Barolo 2018",
      "avg_confidence": 0.91,
      "detections": [...]
    }
  ]
}
```

## Python Integration Examples

### Example 1: Using YOLO Results Directly

```python
from ultralytics import YOLO
import requests
import base64

# Run YOLO
model = YOLO('yolov8n.pt')
results = model('wine_bottles.jpg')

# Convert YOLO results to boundaries
boundaries = []
for idx, box in enumerate(results[0].boxes):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    boundaries.append({
        "x": int(x1),
        "y": int(y1),
        "width": int(x2 - x1),
        "height": int(y2 - y1),
        "label": f"bottle_{idx}"
    })

# Encode image
with open('wine_bottles.jpg', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

# Call batch OCR API
response = requests.post(
    'http://localhost:8001/api/v1/batch',
    json={
        "image": f"data:image/jpeg;base64,{image_base64}",
        "boundaries": boundaries,
        "confidence": 0.25,
        "languages": ["en", "fr", "it", "es"]
    }
)

# Process results
ocr_results = response.json()
for region in ocr_results['results']:
    print(f"{region['label']}: {region['all_text']}")
```

### Example 2: Using Saved Boundaries

```python
import json
import requests
import base64

# Load boundaries from JSON file
with open('boundaries.json', 'r') as f:
    boundaries = json.load(f)

# Encode image
with open('wine_bottles.jpg', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

# Call API
response = requests.post(
    'http://localhost:8001/api/v1/batch',
    json={
        "image": f"data:image/jpeg;base64,{image_base64}",
        "boundaries": boundaries,
        "confidence": 0.25,
        "languages": ["en", "fr", "it", "es"]
    }
)

results = response.json()
```

## Command Reference

### yolo_ocr_pipeline.py
Complete end-to-end pipeline.

```bash
# Basic usage
python src/test/yolo_ocr_pipeline.py --image photo.jpg

# With custom YOLO model
python src/test/yolo_ocr_pipeline.py --image photo.jpg --yolo-model yolov8m.pt

# Save results to file
python src/test/yolo_ocr_pipeline.py --image photo.jpg --save-results output.json

# Custom YOLO confidence threshold
python src/test/yolo_ocr_pipeline.py --image photo.jpg --yolo-confidence 0.5
```

### run_yolo_detection.py
Run only YOLO detection and save boundaries.

```bash
# Basic usage
python src/test/run_yolo_detection.py --image photo.jpg --output boundaries.json

# With visualization
python src/test/run_yolo_detection.py --image photo.jpg --output boundaries.json --visualize

# Custom model and confidence
python src/test/run_yolo_detection.py --image photo.jpg --output boundaries.json --model yolov8m.pt --confidence 0.5
```

### test_batch_api.py
Test batch OCR with existing image and boundaries.

```bash
# From JSON file
python src/test/test_batch_api.py --image photo.jpg --boundaries boundaries.json

# From inline JSON
python src/test/test_batch_api.py --image photo.jpg --boundaries '[{"x":100,"y":50,"width":200,"height":350}]'

# Custom OCR settings
python src/test/test_batch_api.py --image photo.jpg --boundaries boundaries.json --confidence 0.3 --languages en fr

# Show integration example
python src/test/test_batch_api.py --example
```

## Performance

The batch OCR API is optimized for processing multiple regions efficiently:

- **Model caching**: OCR model stays loaded in memory (permanent)
- **Single API call**: All regions processed in one request
- **Sequential processing**: ~0.7s per region (current implementation)
- **Efficient encoding**: Base64 image encoded once, reused for all regions

### Example Timings

- 3 bottles: ~2.1 seconds
- 6 bottles: ~4.2 seconds
- Per region overhead: ~0.7 seconds

## Troubleshooting

### API Not Running

```bash
# Start the API server
uvicorn src.api:app --reload --host 0.0.0.0 --port 8001
```

### YOLO Not Installed

```bash
pip install ultralytics
```

### No Detections Found

- Try lowering `--yolo-confidence` threshold
- Verify the YOLO model is trained for your object class
- Check if objects are clearly visible in the image

### OCR Returns No Text

- Check if bounding boxes actually contain text
- Try lowering `--confidence` threshold for OCR
- Verify correct languages are specified

## Next Steps

For even better performance with many bottles (6+), consider implementing parallel processing to reduce total processing time by ~2-3x.

See the API documentation at http://localhost:8001/docs for interactive testing.
