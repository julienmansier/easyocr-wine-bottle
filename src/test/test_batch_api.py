"""
Test script for the batch OCR API endpoint.

This script demonstrates how to use the /api/v1/batch endpoint to process
multiple regions (e.g., multiple wine bottles detected by YOLO) in a single API call.

Usage:
    # With YOLO boundaries from a JSON file
    python src/test_batch_api.py --image path/to/image.jpg --boundaries path/to/yolo_results.json

    # With manual boundaries
    python src/test_batch_api.py --image path/to/image.jpg --boundaries '[{"x":100,"y":50,"width":200,"height":350}]'

    # Interactive mode (prompts for image and boundaries)
    python src/test_batch_api.py
"""

import base64
import json
import sys
import argparse
import requests
from pathlib import Path
from typing import List, Dict


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    base64_string = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_string}"


def yolo_to_boundaries(yolo_results) -> List[Dict]:
    """
    Convert YOLO detection results to batch API boundary format.

    Args:
        yolo_results: YOLO results object from ultralytics

    Returns:
        List of boundary dictionaries
    """
    boundaries = []

    # Handle ultralytics YOLO results object
    if hasattr(yolo_results, 'boxes'):
        for idx, box in enumerate(yolo_results.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0]) if hasattr(box, 'cls') else 0

            boundaries.append({
                "x": int(x1),
                "y": int(y1),
                "width": int(x2 - x1),
                "height": int(y2 - y1),
                "label": f"detection_{idx}_conf{conf:.2f}_cls{cls}"
            })

    return boundaries


def load_boundaries_from_json(json_path: str) -> List[Dict]:
    """
    Load boundaries from a JSON file.

    Expected format:
    [
        {"x": 100, "y": 50, "width": 200, "height": 350, "label": "bottle_1"},
        ...
    ]

    Or YOLO format:
    {
        "detections": [
            {"bbox": [x1, y1, x2, y2], "confidence": 0.95, "class": 0},
            ...
        ]
    }
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # If it's already in our format (list of boundaries)
    if isinstance(data, list):
        return data

    # If it's YOLO format with detections
    if isinstance(data, dict) and 'detections' in data:
        boundaries = []
        for idx, det in enumerate(data['detections']):
            if 'bbox' in det:
                x1, y1, x2, y2 = det['bbox']
                conf = det.get('confidence', 0.0)
                cls = det.get('class', 0)

                boundaries.append({
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                    "label": f"detection_{idx}_conf{conf:.2f}_cls{cls}"
                })
        return boundaries

    raise ValueError(f"Unsupported JSON format in {json_path}")


def test_batch_ocr(
    image_path: str = None,
    boundaries: List[Dict] = None,
    api_url: str = "http://localhost:8001/api/v1/batch",
    confidence: float = 0.25,
    languages: List[str] = None
):
    """
    Test the batch OCR endpoint with multiple bounding boxes.

    Args:
        image_path: Path to the image file
        boundaries: List of boundary dictionaries or path to JSON file
        api_url: API endpoint URL
        confidence: OCR confidence threshold
        languages: List of language codes
    """
    if languages is None:
        languages = ["en", "fr", "it", "es"]

    # Validate image path
    if not image_path or not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        print("\nUsage:")
        print("  python src/test_batch_api.py --image <path> --boundaries <json_file_or_string>")
        return None

    # Load or validate boundaries
    if not boundaries:
        print("Error: No boundaries provided")
        print("\nProvide boundaries as:")
        print("  1. JSON file: --boundaries boundaries.json")
        print("  2. JSON string: --boundaries '[{\"x\":100,\"y\":50,\"width\":200,\"height\":350}]'")
        return None

    # If boundaries is a string path to JSON file, load it
    if isinstance(boundaries, str):
        if Path(boundaries).exists():
            print(f"Loading boundaries from: {boundaries}")
            boundaries = load_boundaries_from_json(boundaries)
        else:
            # Try to parse as JSON string
            try:
                boundaries = json.loads(boundaries)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in boundaries: {boundaries}")
                return None

    # Validate boundaries format
    if not isinstance(boundaries, list) or len(boundaries) == 0:
        print("Error: Boundaries must be a non-empty list")
        return None

    # Encode image
    print(f"Encoding image: {image_path}")
    base64_image = encode_image_to_base64(image_path)

    # Create request payload
    payload = {
        "image": base64_image,
        "boundaries": boundaries,
        "confidence": confidence,
        "languages": languages,
        "return_annotated": False
    }

    print(f"\nSending batch OCR request with {len(boundaries)} regions...")
    print(f"API: {api_url}")
    print(f"Confidence: {confidence}")
    print(f"Languages: {', '.join(languages)}")

    try:
        # Send request
        response = requests.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120
        )

        # Check response
        if response.status_code == 200:
            result = response.json()

            print(f"\n{'='*70}")
            print(f"Batch OCR Results")
            print(f"{'='*70}")
            print(f"Success: {result['success']}")
            print(f"Total Regions: {result['total_regions']}")
            print(f"Successful Regions: {result['successful_regions']}")
            print(f"Languages: {', '.join(result['languages_used'])}")
            print(f"Confidence Threshold: {result['confidence_threshold']}")
            print(f"{'='*70}\n")

            # Display results for each region
            for region_result in result['results']:
                print(f"\nRegion {region_result['region_index']}: {region_result.get('label', 'N/A')}")
                print(f"  Bounding Box: x={region_result['bbox']['x']}, y={region_result['bbox']['y']}, "
                      f"w={region_result['bbox']['width']}, h={region_result['bbox']['height']}")
                print(f"  Success: {region_result['success']}")
                print(f"  Text Blocks: {region_result['total_text_blocks']}")
                print(f"  Avg Confidence: {region_result['avg_confidence']:.2f}")
                print(f"  All Text: {region_result['all_text']}")

                if region_result['detections']:
                    print(f"  Individual Detections:")
                    for det in region_result['detections']:
                        print(f"    - '{det['text']}' (confidence: {det['confidence']:.2f})")
                print(f"  {'-'*60}")

            return result

        else:
            print(f"\nError: {response.status_code}")
            print(response.json())
            return None

    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to API server.")
        print("Make sure the API is running:")
        print("  uvicorn src.api:app --reload --host 0.0.0.0 --port 8001")
        return None

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_integration_with_yolo():
    """
    Example showing how to integrate YOLO detection with batch OCR.

    This is a conceptual example - in practice, you would:
    1. Run YOLO on your image to detect wine bottles
    2. Extract bounding boxes from YOLO results
    3. Send bounding boxes + image to batch OCR endpoint
    """

    print("\n" + "="*70)
    print("Integration Example: YOLO + Batch OCR")
    print("="*70)

    example_code = '''
# Step 1: Run YOLO to detect wine bottles
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model('wine_bottles_image.jpg')

# Step 2: Extract bounding boxes from YOLO detections
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

# Step 3: Prepare image for API
import base64
with open('wine_bottles_image.jpg', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

# Step 4: Send to batch OCR API
import requests
response = requests.post(
    'http://localhost:8001/api/v1/batch',
    json={
        "image": f"data:image/jpeg;base64,{image_base64}",
        "boundaries": boundaries,
        "confidence": 0.25,
        "languages": ["en", "fr", "it", "es"]
    }
)

# Step 5: Process results
ocr_results = response.json()
for region in ocr_results['results']:
    bottle_label = region['label']
    text = region['all_text']
    print(f"{bottle_label}: {text}")
'''

    print(example_code)
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test batch OCR API with YOLO detection results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With JSON file containing boundaries
  python src/test_batch_api.py --image photo.jpg --boundaries yolo_results.json

  # With inline JSON boundaries
  python src/test_batch_api.py --image photo.jpg --boundaries '[{"x":100,"y":50,"width":200,"height":350,"label":"bottle_1"}]'

  # With custom confidence and languages
  python src/test_batch_api.py --image photo.jpg --boundaries bounds.json --confidence 0.3 --languages en fr

  # Show integration example only
  python src/test_batch_api.py --example
        """
    )

    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Path to the image file'
    )
    parser.add_argument(
        '--boundaries', '-b',
        type=str,
        help='Path to JSON file with boundaries or JSON string'
    )
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.25,
        help='OCR confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--languages', '-l',
        nargs='+',
        default=['en', 'fr', 'it', 'es'],
        help='Language codes for OCR (default: en fr it es)'
    )
    parser.add_argument(
        '--api-url',
        type=str,
        default='http://localhost:8001/api/v1/batch',
        help='API endpoint URL (default: http://localhost:8001/api/v1/batch)'
    )
    parser.add_argument(
        '--example',
        action='store_true',
        help='Show YOLO integration example and exit'
    )

    args = parser.parse_args()

    print("Batch OCR API Test\n")

    # If --example flag is set, just show the integration example
    if args.example:
        test_integration_with_yolo()
        sys.exit(0)

    # If no arguments provided, show help
    if not args.image and not args.boundaries:
        parser.print_help()
        print("\n" + "="*70)
        print("No arguments provided. Showing integration example:")
        print("="*70)
        test_integration_with_yolo()
        sys.exit(0)

    # Run the test with provided arguments
    result = test_batch_ocr(
        image_path=args.image,
        boundaries=args.boundaries,
        api_url=args.api_url,
        confidence=args.confidence,
        languages=args.languages
    )

    if result:
        print("\n✓ Test completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Test failed!")
        sys.exit(1)
