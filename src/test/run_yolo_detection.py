"""
YOLO Detection Script for Wine Bottles

This script runs YOLO object detection on an image and saves the bounding boxes
in a format compatible with the batch OCR API.

Usage:
    python run_yolo_detection.py --image path/to/image.jpg --output boundaries.json
    python run_yolo_detection.py --image path/to/image.jpg --output boundaries.json --model yolov8n.pt
    python run_yolo_detection.py --image path/to/image.jpg --output boundaries.json --visualize
"""

import argparse
import json
from pathlib import Path


def run_yolo_detection(
    image_path: str,
    output_path: str = None,
    model_path: str = "yolov8n.pt",
    confidence: float = 0.25,
    visualize: bool = False
):
    """
    Run YOLO detection on an image and save boundaries.

    Args:
        image_path: Path to input image
        output_path: Path to save boundaries JSON (optional)
        model_path: Path to YOLO model weights
        confidence: Detection confidence threshold
        visualize: Whether to save annotated image
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not found")
        print("Install it with: pip install ultralytics")
        return None

    # Load YOLO model
    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    # Run detection
    print(f"Running detection on: {image_path}")
    results = model(image_path, conf=confidence)

    # Extract bounding boxes
    boundaries = []
    detections_info = []

    if results and len(results) > 0:
        result = results[0]

        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            for idx, box in enumerate(result.boxes):
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0]) if hasattr(box, 'cls') else 0

                # Get class name if available
                class_name = model.names[cls] if hasattr(model, 'names') else f"class_{cls}"

                # Create boundary in API format
                boundary = {
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                    "label": f"{class_name}_{idx}"
                }

                boundaries.append(boundary)

                # Store detection info for display
                detections_info.append({
                    "index": idx,
                    "class": class_name,
                    "confidence": conf,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })

    # Print results
    print(f"\n{'='*70}")
    print(f"YOLO Detection Results")
    print(f"{'='*70}")
    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print(f"Confidence Threshold: {confidence}")
    print(f"Detections: {len(boundaries)}")
    print(f"{'='*70}\n")

    if detections_info:
        for det in detections_info:
            print(f"Detection {det['index']}:")
            print(f"  Class: {det['class']}")
            print(f"  Confidence: {det['confidence']:.2f}")
            print(f"  BBox: {det['bbox']}")
            print()
    else:
        print("No detections found!")

    # Save boundaries to JSON
    if output_path and boundaries:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(boundaries, f, indent=2)

        print(f"✓ Saved {len(boundaries)} boundaries to: {output_path}")
        print(f"\nYou can now run batch OCR with:")
        print(f"  python src/test_batch_api.py --image {image_path} --boundaries {output_path}")

    # Save visualized results
    if visualize and results:
        vis_path = Path(image_path).parent / f"{Path(image_path).stem}_yolo_detection.jpg"
        results[0].save(str(vis_path))
        print(f"✓ Saved visualization to: {vis_path}")

    return boundaries


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO detection and save boundaries for batch OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic detection
  python run_yolo_detection.py --image photo.jpg --output boundaries.json

  # With custom model and visualization
  python run_yolo_detection.py --image photo.jpg --output boundaries.json --model yolov8m.pt --visualize

  # Adjust confidence threshold
  python run_yolo_detection.py --image photo.jpg --output boundaries.json --confidence 0.5
        """
    )

    parser.add_argument(
        '--image', '-i',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='boundaries.json',
        help='Path to output JSON file (default: boundaries.json)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='yolov8n.pt',
        help='YOLO model to use (default: yolov8n.pt)'
    )
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.25,
        help='Detection confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Save annotated image with detections'
    )

    args = parser.parse_args()

    # Validate image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return 1

    # Run detection
    boundaries = run_yolo_detection(
        image_path=args.image,
        output_path=args.output,
        model_path=args.model,
        confidence=args.confidence,
        visualize=args.visualize
    )

    return 0 if boundaries else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
