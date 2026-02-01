"""
Complete YOLO + OCR Pipeline

This script demonstrates the full workflow:
1. Run YOLO to detect wine bottles
2. Send detected regions to batch OCR API
3. Display results with bottle text associations

Usage:
    python yolo_ocr_pipeline.py --image path/to/wine_bottles.jpg
    python yolo_ocr_pipeline.py --image path/to/wine_bottles.jpg --save-results results.json
"""

import argparse
import base64
import json
import sys
from pathlib import Path
import requests


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    base64_string = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_string}"


def run_yolo_detection(image_path: str, model_path: str = "yolov8n.pt", confidence: float = 0.25):
    """Run YOLO detection and return boundaries."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not found")
        print("Install it with: pip install ultralytics")
        return None

    print(f"[1/3] Running YOLO detection...")
    model = YOLO(model_path)
    results = model(image_path, conf=confidence)

    boundaries = []
    if results and len(results) > 0:
        result = results[0]
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            for idx, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0]) if hasattr(box, 'cls') else 0
                class_name = model.names[cls] if hasattr(model, 'names') else f"class_{cls}"

                boundaries.append({
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                    "label": f"{class_name}_{idx}_conf{conf:.2f}"
                })

    print(f"   Found {len(boundaries)} detections")
    return boundaries


def run_batch_ocr(image_path: str, boundaries: list, api_url: str = "http://localhost:8001/api/v1/batch"):
    """Send image and boundaries to batch OCR API."""
    if not boundaries:
        print("Error: No boundaries to process")
        return None

    print(f"\n[2/3] Running batch OCR on {len(boundaries)} regions...")

    # Encode image
    base64_image = encode_image_to_base64(image_path)

    # Create request
    payload = {
        "image": base64_image,
        "boundaries": boundaries,
        "confidence": 0.25,
        "languages": ["en", "fr", "it", "es"],
        "return_annotated": False
    }

    try:
        response = requests.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            print(f"   Successfully processed {result['successful_regions']}/{result['total_regions']} regions")
            return result
        else:
            print(f"   Error: {response.status_code}")
            print(response.json())
            return None

    except requests.exceptions.ConnectionError:
        print("\n   Error: Could not connect to API server.")
        print("   Make sure the API is running:")
        print("     uvicorn src.api:app --reload --host 0.0.0.0 --port 8001")
        return None


def display_results(ocr_results):
    """Display OCR results in a readable format."""
    if not ocr_results:
        return

    print(f"\n[3/3] Results:")
    print(f"{'='*70}")

    for region in ocr_results['results']:
        print(f"\n{region['label']}:")
        print(f"  Location: ({region['bbox']['x']}, {region['bbox']['y']}) "
              f"Size: {region['bbox']['width']}x{region['bbox']['height']}")
        print(f"  Text: {region['all_text']}")
        print(f"  Confidence: {region['avg_confidence']:.2f}")

        if region['detections']:
            print(f"  Detections:")
            for det in region['detections']:
                print(f"    - \"{det['text']}\" (conf: {det['confidence']:.2f})")

    print(f"\n{'='*70}")


def save_results(ocr_results, output_path: str):
    """Save results to JSON file."""
    # Create simplified output
    simplified_results = []

    for region in ocr_results['results']:
        simplified_results.append({
            "label": region['label'],
            "text": region['all_text'],
            "confidence": region['avg_confidence'],
            "bbox": region['bbox'],
            "detections": [
                {"text": d['text'], "confidence": d['confidence']}
                for d in region['detections']
            ]
        })

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            "total_regions": len(simplified_results),
            "results": simplified_results
        }, f, indent=2)

    print(f"\n✓ Saved results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Complete YOLO + Batch OCR pipeline for wine bottles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python yolo_ocr_pipeline.py --image wine_shelf.jpg
  python yolo_ocr_pipeline.py --image wine_shelf.jpg --save-results output.json --yolo-model yolov8m.pt
        """
    )

    parser.add_argument(
        '--image', '-i',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--save-results', '-o',
        type=str,
        help='Path to save results JSON (optional)'
    )
    parser.add_argument(
        '--yolo-model', '-m',
        type=str,
        default='yolov8n.pt',
        help='YOLO model to use (default: yolov8n.pt)'
    )
    parser.add_argument(
        '--yolo-confidence', '-c',
        type=float,
        default=0.25,
        help='YOLO detection confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--api-url',
        type=str,
        default='http://localhost:8001/api/v1/batch',
        help='OCR API endpoint (default: http://localhost:8001/api/v1/batch)'
    )

    args = parser.parse_args()

    # Validate image
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return 1

    print(f"YOLO + OCR Pipeline")
    print(f"{'='*70}")
    print(f"Input: {args.image}")
    print(f"YOLO Model: {args.yolo_model}")
    print(f"OCR API: {args.api_url}")
    print(f"{'='*70}\n")

    # Step 1: YOLO detection
    boundaries = run_yolo_detection(
        args.image,
        model_path=args.yolo_model,
        confidence=args.yolo_confidence
    )

    if not boundaries:
        print("No objects detected!")
        return 1

    # Step 2: Batch OCR
    ocr_results = run_batch_ocr(args.image, boundaries, api_url=args.api_url)

    if not ocr_results:
        print("OCR failed!")
        return 1

    # Step 3: Display results
    display_results(ocr_results)

    # Save results if requested
    if args.save_results:
        save_results(ocr_results, args.save_results)

    print("\n✓ Pipeline completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
