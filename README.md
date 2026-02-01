# Wine Bottle OCR with EasyOCR

EasyOCR-based text extraction service for wine bottle images. This project provides both a Python library and a containerized FastAPI service for extracting text from wine bottle labels in multiple languages.

## Features

- **Multi-language OCR**: Supports English, French, Italian, Spanish, German, Portuguese, and more
- **REST API**: FastAPI-based service with automatic documentation
- **Batch OCR Endpoint**: Process multiple regions (YOLO detections) in a single API call
- **YOLO Integration**: Complete pipeline for detecting bottles and extracting text
- **Model Caching**: OCR model stays loaded in memory for fast processing
- **Docker Support**: Fully containerized with Docker and docker-compose
- **Confidence Filtering**: Configurable confidence thresholds for text detection
- **Visual Annotations**: Optional annotated images with bounding boxes

## Quick Start

### Using Docker (Recommended)

1. **Build and run with docker-compose:**
   ```bash
   docker-compose up --build
   ```

2. **Access the API:**
   - API docs: http://localhost:8001/docs
   - Health check: http://localhost:8001/health
   - API info: http://localhost:8001/api/v1/info

### Local Installation

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API:**
   ```bash
   uvicorn src.api:app --reload --host 0.0.0.0 --port 8001
   ```

## API Usage

### Extract Text from Image

**Endpoint:** `POST /api/v1/extract`

**Request:**
```bash
curl -X POST "http://localhost:8001/api/v1/extract" \
     -H "Content-Type: application/json" \
     -d '{
           "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
           "confidence": 0.25,
           "languages": ["en", "fr", "it"],
           "return_annotated": true
         }'
```

**Python Example:**
```python
import base64
import requests

# Read and encode image
with open("wine_bottle.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Make request
response = requests.post(
    "http://localhost:8001/api/v1/extract",
    json={
        "image": f"data:image/jpeg;base64,{image_data}",
        "confidence": 0.25,
        "languages": ["en", "fr", "it", "es"],
        "return_annotated": True
    }
)

result = response.json()
print(f"Detected text: {result['all_text']}")
print(f"Total blocks: {result['total_text_blocks']}")
```

**Response:**
```json
{
  "success": true,
  "total_text_blocks": 5,
  "all_text": "Château Example 2015 Bordeaux France",
  "avg_confidence": 0.87,
  "detections": [
    {
      "text": "Château Example",
      "confidence": 0.92,
      "bbox": [[120, 50], [380, 50], [380, 90], [120, 90]],
      "position": [250, 70]
    }
  ],
  "languages_used": ["en", "fr"],
  "confidence_threshold": 0.25,
  "annotated_image": "data:image/jpeg;base64,..."
}
```

## Python Library Usage

### Basic Text Extraction

```python
from src.utils.ocr_reader import WineBottleOCR

# Initialize OCR reader
ocr = WineBottleOCR(
    languages=['en', 'fr', 'it', 'es'],
    confidence_threshold=0.25,
    gpu=True
)

# Extract text from image
count, detections = ocr.extract_all_text(
    'sample_images/wine_bottle.jpeg',
    visualize=True,
    save_path='output_ocr.jpg'
)

print(f"Found {count} text blocks")
for det in detections:
    print(f"  {det['text']} (confidence: {det['confidence']:.2f})")
```

### Get Text Summary

```python
summary = ocr.get_text_summary('sample_images/wine_bottle.jpeg')

print(f"Total text blocks: {summary['total_text_blocks']}")
print(f"All text: {summary['all_text']}")
print(f"Average confidence: {summary['avg_confidence']:.2f}")
```

### Batch Processing

```python
import glob

# Get all images
image_paths = glob.glob('sample_images/*.jpg')

# Process batch
results = ocr.batch_process(
    image_paths,
    save_dir='ocr_results'
)

# Print summary
for result in results:
    print(f"{result['image_path']}: {result['total_text_blocks']} text blocks")
```

## Supported Languages

The EasyOCR library supports 80+ languages. Common ones for wine labels:

- `en` - English
- `fr` - French
- `it` - Italian
- `es` - Spanish
- `de` - German
- `pt` - Portuguese
- `ru` - Russian
- `ar` - Arabic
- `zh` - Chinese
- `ja` - Japanese
- `ko` - Korean

For a full list, visit: https://www.jaided.ai/easyocr/

## API Endpoints

### Health Check
```bash
GET /health
```
Returns API health status.

### API Information
```bash
GET /api/v1/info
```
Returns API capabilities, supported languages, and available endpoints.

### Extract Text (Single Image)
```bash
POST /api/v1/extract
```
Extract text from a base64-encoded image.

### Batch OCR (Multiple Regions)
```bash
POST /api/v1/batch
```
Extract text from multiple regions in a single image (YOLO integration).

**Example:**
```bash
python src/test/test_batch_api.py --image photo.jpg --boundaries boundaries.json
```

See [src/test/BATCH_OCR_USAGE.md](src/test/BATCH_OCR_USAGE.md) for complete documentation.

## Configuration

### Confidence Threshold
Adjust the confidence threshold (0.0-1.0) to filter low-confidence detections:
- **0.1-0.3**: More detections, may include false positives
- **0.5-0.7**: Balanced (recommended)
- **0.8-1.0**: Only high-confidence detections

### GPU Acceleration
EasyOCR supports GPU acceleration with CUDA. To enable:
1. Install PyTorch with CUDA support
2. Set `gpu=True` when initializing `WineBottleOCR`

For CPU-only:
```python
ocr = WineBottleOCR(languages=['en'], gpu=False)
```

## Docker Build

### Build the image
```bash
docker build -t wine-bottle-ocr:latest .
```

### Run the container
```bash
docker run -p 8001:8001 wine-bottle-ocr:latest
```

### With docker-compose
```bash
docker-compose up --build
```

## Project Structure

```
easyocr/
├── src/
│   ├── api.py                 # FastAPI application with batch endpoint
│   ├── utils/
│   │   └── ocr_reader.py      # EasyOCR wrapper
│   └── test/
│       ├── yolo_ocr_pipeline.py      # End-to-end YOLO + OCR pipeline
│       ├── run_yolo_detection.py     # YOLO detection script
│       ├── test_batch_api.py         # Batch API testing tool
│       ├── test_batch_performance.py # Performance benchmarks
│       └── BATCH_OCR_USAGE.md        # Complete batch OCR guide
├── examples/                  # Usage examples
├── models/                    # Downloaded EasyOCR models (auto-created)
├── Dockerfile                 # Container definition
├── docker-compose.yml         # Docker compose configuration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Development

### Run tests
```bash
python src/test_api.py
```

### Run with auto-reload
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8001
```

### View API documentation
- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

## Performance Notes

- **First Run**: EasyOCR downloads language models on first use (~100MB per language)
- **GPU vs CPU**: GPU acceleration provides 3-5x speedup for OCR processing
- **Image Size**: Larger images take longer to process; consider resizing very large images
- **Language Selection**: Using fewer languages improves speed and accuracy

## YOLO + OCR Integration

This project provides a complete pipeline that combines YOLO object detection with batch OCR:

**Quick Start:**
```bash
python src/test/yolo_ocr_pipeline.py --image wine_bottles.jpg
```

**How it works:**
1. YOLO detects and locates wine bottles in the image
2. Bounding boxes are sent to the batch OCR API in a single request
3. Text is extracted from each bottle and associated with its detection

**For detailed documentation, see:** [src/test/BATCH_OCR_USAGE.md](src/test/BATCH_OCR_USAGE.md)

This approach is much more efficient than processing bottles individually:
- Single API call for all bottles
- Model stays loaded in memory
- ~2.1 seconds for 3 bottles (~0.7s per bottle)

## License

MIT License - feel free to use this project for commercial and non-commercial purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Issues

If you encounter any issues, please report them at: https://github.com/julienmansier/easyocr-wine-bottle/issues
