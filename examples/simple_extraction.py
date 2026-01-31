"""
Simple OCR Text Extraction Example

This script demonstrates basic text extraction from a wine bottle image.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.ocr_reader import WineBottleOCR


def main():
    """Extract text from a single wine bottle image."""

    # Initialize OCR reader
    print("Initializing EasyOCR reader...")
    ocr = WineBottleOCR(
        languages=['en', 'fr', 'it', 'es'],
        confidence_threshold=0.25,
        gpu=True  # Set to False if no GPU available
    )

    # Path to sample image
    image_path = 'sample_images/wine_bottle.jpeg'

    print(f"\nExtracting text from: {image_path}")

    # Extract text with visualization
    count, detections = ocr.extract_all_text(
        image_path,
        visualize=False,
        save_path='ocr_result.jpg'
    )

    # Print results
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Total text blocks detected: {count}")
    print()

    if count > 0:
        for i, det in enumerate(detections, 1):
            print(f"{i}. Text: {det['text']}")
            print(f"   Confidence: {det['confidence']:.2f}")
            print(f"   Position: {det['position']}")
            print()

        print(f"Annotated image saved to: ocr_result.jpg")
    else:
        print("No text detected in the image.")


if __name__ == "__main__":
    main()
