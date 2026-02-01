"""
Test script to verify batch OCR processing and measure performance.

This script:
1. Creates a test image with text regions (simulating multiple wine bottles)
2. Tests the batch API endpoint
3. Measures sequential processing time
4. Helps determine if parallel processing would be beneficial
"""

import base64
import json
import time
import requests
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_test_image_with_text(output_path: str = "/tmp/test_wine_bottles.jpg"):
    """
    Create a test image with multiple text regions simulating wine bottle labels.
    """
    # Create a large image (simulating a photo with 3 wine bottles side by side)
    width = 1200
    height = 800
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Try to use a decent font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Define 3 "bottle" regions with text
    bottles = [
        {
            "region": (50, 100, 350, 700),  # x1, y1, x2, y2
            "texts": ["CHATEAU", "MARGAUX", "2015", "Bordeaux"],
            "label": "bottle_1"
        },
        {
            "region": (400, 100, 700, 700),
            "texts": ["BAROLO", "RISERVA", "2018", "Piedmont"],
            "label": "bottle_2"
        },
        {
            "region": (750, 100, 1050, 700),
            "texts": ["NAPA VALLEY", "CABERNET", "2020", "California"],
            "label": "bottle_3"
        }
    ]

    # Draw each bottle region
    for bottle in bottles:
        x1, y1, x2, y2 = bottle["region"]

        # Draw border
        draw.rectangle([x1, y1, x2, y2], outline='gray', width=3)

        # Draw text in the region
        y_offset = y1 + 100
        for text in bottle["texts"]:
            # Center the text
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            x_centered = x1 + (x2 - x1 - text_width) // 2

            draw.text((x_centered, y_offset), text, fill='black', font=font)
            y_offset += 80

    # Save the image
    img.save(output_path, quality=95)
    print(f"Created test image: {output_path}")
    print(f"Image size: {width}x{height}")

    # Return bounding boxes for API testing
    return [
        {
            "x": 50,
            "y": 100,
            "width": 300,
            "height": 600,
            "label": "bottle_1"
        },
        {
            "x": 400,
            "y": 100,
            "width": 300,
            "height": 600,
            "label": "bottle_2"
        },
        {
            "x": 750,
            "y": 100,
            "width": 300,
            "height": 600,
            "label": "bottle_3"
        }
    ]


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    base64_string = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_string}"


def test_batch_api(api_url: str, image_path: str, boundaries: list):
    """
    Test the batch OCR API endpoint and measure performance.
    """
    print(f"\n{'='*70}")
    print(f"Testing Batch OCR API")
    print(f"{'='*70}")

    # Encode image
    print(f"Encoding image: {image_path}")
    base64_image = encode_image_to_base64(image_path)

    # Create request payload
    payload = {
        "image": base64_image,
        "boundaries": boundaries,
        "confidence": 0.25,
        "languages": ["en"],
        "return_annotated": False
    }

    print(f"Sending batch OCR request with {len(boundaries)} regions...")

    # Measure time
    start_time = time.time()

    try:
        # Send request
        response = requests.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120  # 2 minute timeout
        )

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Check response
        if response.status_code == 200:
            result = response.json()

            print(f"\n{'='*70}")
            print(f"SUCCESS - Batch OCR Results")
            print(f"{'='*70}")
            print(f"Total Processing Time: {elapsed_time:.2f} seconds")
            print(f"Average Time per Region: {elapsed_time / len(boundaries):.2f} seconds")
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

            # Performance analysis
            print(f"\n{'='*70}")
            print(f"Performance Analysis")
            print(f"{'='*70}")
            print(f"Sequential Processing:")
            print(f"  Total Time: {elapsed_time:.2f}s")
            print(f"  Time per Region: {elapsed_time / len(boundaries):.2f}s")
            print(f"\nEstimated Parallel Processing (ideal 3x speedup):")
            print(f"  Estimated Time: {elapsed_time / 3:.2f}s")
            print(f"  Potential Savings: {elapsed_time - (elapsed_time / 3):.2f}s ({((elapsed_time - (elapsed_time / 3)) / elapsed_time * 100):.1f}%)")

            if elapsed_time > 5:
                print(f"\n⚠️  Total processing time > 5s - Parallel processing HIGHLY RECOMMENDED")
            elif elapsed_time > 3:
                print(f"\n💡 Total processing time > 3s - Parallel processing would provide noticeable benefit")
            else:
                print(f"\n✓  Processing time < 3s - Parallel processing optional")

            return True, elapsed_time

        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.json())
            return False, None

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server.")
        print("Make sure the API is running:")
        print("  uvicorn src.api:app --reload --host 0.0.0.0 --port 8001")
        return False, None

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def test_single_region_timing(api_url: str, image_path: str, boundaries: list):
    """
    Test individual region processing to understand per-region overhead.
    """
    print(f"\n{'='*70}")
    print(f"Testing Individual Region Timing")
    print(f"{'='*70}")

    base64_image = encode_image_to_base64(image_path)

    times = []
    for idx, boundary in enumerate(boundaries):
        payload = {
            "image": base64_image,
            "boundaries": [boundary],  # Single region
            "confidence": 0.25,
            "languages": ["en"],
            "return_annotated": False
        }

        start = time.time()
        response = requests.post(api_url, json=payload, headers={"Content-Type": "application/json"})
        elapsed = time.time() - start
        times.append(elapsed)

        if response.status_code == 200:
            print(f"Region {idx} ({boundary.get('label', 'N/A')}): {elapsed:.2f}s")
        else:
            print(f"Region {idx} ({boundary.get('label', 'N/A')}): FAILED")

    total_sequential = sum(times)
    print(f"\nTotal time if processed individually: {total_sequential:.2f}s")
    print(f"Average time per region: {total_sequential / len(times):.2f}s")

    return times


def main():
    """Main test function."""
    print("Wine Bottle OCR - Batch Processing Performance Test\n")

    # Configuration
    api_url = "http://localhost:8001/api/v1/batch"
    test_image_path = "/tmp/test_wine_bottles.jpg"

    # Step 1: Create test image
    print("Step 1: Creating test image with simulated wine bottle labels...")
    boundaries = create_test_image_with_text(test_image_path)
    print(f"Created {len(boundaries)} regions for testing\n")

    # Step 2: Test batch API
    print("Step 2: Testing batch API with all regions...")
    success, batch_time = test_batch_api(api_url, test_image_path, boundaries)

    if not success:
        print("\n⚠️  Batch test failed. Please ensure the API is running:")
        print("  uvicorn src.api:app --reload --host 0.0.0.0 --port 8001")
        return

    # Step 3: Test individual region timing for comparison
    print("\nStep 3: Testing individual region processing for comparison...")
    individual_times = test_single_region_timing(api_url, test_image_path, boundaries)

    # Final recommendation
    print(f"\n{'='*70}")
    print(f"FINAL RECOMMENDATION")
    print(f"{'='*70}")

    if batch_time and individual_times:
        total_individual = sum(individual_times)
        speedup_potential = total_individual / batch_time if batch_time > 0 else 1

        print(f"Batch processing time: {batch_time:.2f}s")
        print(f"Sum of individual times: {total_individual:.2f}s")
        print(f"Current batch efficiency: {speedup_potential:.2f}x faster than individual calls")

        if batch_time > 4:
            print(f"\n✅ RECOMMENDATION: Implement parallel processing")
            print(f"   Expected benefit: 2-3x speedup ({batch_time:.2f}s → ~{batch_time/2.5:.2f}s)")
        elif batch_time > 2:
            print(f"\n💡 RECOMMENDATION: Parallel processing would provide moderate benefit")
            print(f"   Expected benefit: ~2x speedup ({batch_time:.2f}s → ~{batch_time/2:.2f}s)")
        else:
            print(f"\n⚠️  RECOMMENDATION: Parallel processing may not be worth the complexity")
            print(f"   Processing is already fast at {batch_time:.2f}s")


if __name__ == "__main__":
    main()
