"""
Batch OCR Processing Example

This script demonstrates batch processing of multiple wine bottle images.
"""

import sys
from pathlib import Path
import glob

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.ocr_reader import WineBottleOCR


def main():
    """Process multiple wine bottle images in batch."""

    # Initialize OCR reader
    print("Initializing EasyOCR reader...")
    ocr = WineBottleOCR(
        languages=['en', 'fr', 'it', 'es'],
        confidence_threshold=0.25,
        gpu=True  # Set to False if no GPU available
    )

    # Get all sample images
    image_patterns = [
        'sample_images/*.jpg',
        'sample_images/*.jpeg',
        'sample_images/*.png'
    ]

    image_paths = []
    for pattern in image_patterns:
        image_paths.extend(glob.glob(pattern))

    if not image_paths:
        print("No images found in sample_images/ directory")
        return

    print(f"\nFound {len(image_paths)} images to process")

    # Process batch
    print("\nProcessing images...")
    results = ocr.batch_process(
        image_paths,
        save_dir='batch_results'
    )

    # Print summary
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING SUMMARY")
    print(f"{'='*70}")

    total_blocks = 0
    for i, result in enumerate(results, 1):
        filename = Path(result['image_path']).name

        if 'error' in result:
            print(f"{i}. {filename}: ERROR - {result['error']}")
        else:
            blocks = result['total_text_blocks']
            total_blocks += blocks
            avg_conf = result['avg_confidence']

            print(f"{i}. {filename}:")
            print(f"   Text blocks: {blocks}")
            print(f"   Avg confidence: {avg_conf:.2f}")
            if blocks > 0:
                text_preview = result['all_text'][:60]
                print(f"   Text preview: {text_preview}...")

    print(f"\n{'='*70}")
    print(f"Total text blocks detected across all images: {total_blocks}")
    print(f"Average blocks per image: {total_blocks / len(results):.1f}")
    print(f"\nAnnotated images saved to: batch_results/")


if __name__ == "__main__":
    main()
