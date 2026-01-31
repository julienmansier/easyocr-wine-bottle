"""
API Test Script

Test the Wine Bottle OCR API endpoints.
"""

import base64
import requests
import sys
from pathlib import Path


def test_health():
    """Test the health endpoint."""
    print("\n" + "="*60)
    print("Testing /health endpoint")
    print("="*60)

    response = requests.get("http://localhost:8000/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'
    print("✓ Health check passed")


def test_info():
    """Test the info endpoint."""
    print("\n" + "="*60)
    print("Testing /api/v1/info endpoint")
    print("="*60)

    response = requests.get("http://localhost:8000/api/v1/info")
    print(f"Status Code: {response.status_code}")

    data = response.json()
    print(f"Service: {data['service']}")
    print(f"Version: {data['version']}")
    print(f"Default Languages: {data['default_languages']}")
    print(f"Endpoints: {data['endpoints']}")

    assert response.status_code == 200
    print("✓ Info endpoint passed")


def test_extract_text(image_path: str = "sample_images/wine_bottle.jpeg"):
    """Test the extract text endpoint."""
    print("\n" + "="*60)
    print("Testing /api/v1/extract endpoint")
    print("="*60)

    # Check if image exists
    if not Path(image_path).exists():
        print(f"Warning: Image not found at {image_path}")
        print("Skipping OCR test")
        return

    # Read and encode image
    print(f"Reading image: {image_path}")
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    # Make request
    print("Sending OCR request...")
    response = requests.post(
        "http://localhost:8000/api/v1/extract",
        json={
            "image": f"data:image/jpeg;base64,{image_data}",
            "confidence": 0.25,
            "languages": ["en", "fr", "it", "es"],
            "return_annotated": True
        }
    )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\nOCR Results:")
        print(f"  Success: {data['success']}")
        print(f"  Total text blocks: {data['total_text_blocks']}")
        print(f"  Average confidence: {data['avg_confidence']:.2f}")
        print(f"  Languages used: {data['languages_used']}")
        print(f"\n  Detected text: {data['all_text']}")

        if data['total_text_blocks'] > 0:
            print(f"\n  Individual detections:")
            for i, det in enumerate(data['detections'][:5], 1):  # Show first 5
                print(f"    {i}. \"{det['text']}\" (confidence: {det['confidence']:.2f})")

        print("✓ Extract text endpoint passed")
    else:
        print(f"Error: {response.text}")
        print("✗ Extract text endpoint failed")


def test_extract_text_invalid():
    """Test the extract endpoint with invalid data."""
    print("\n" + "="*60)
    print("Testing /api/v1/extract with invalid data")
    print("="*60)

    # Send invalid base64
    response = requests.post(
        "http://localhost:8000/api/v1/extract",
        json={
            "image": "invalid_base64_data",
            "confidence": 0.5,
            "languages": ["en"],
            "return_annotated": False
        }
    )

    print(f"Status Code: {response.status_code}")
    assert response.status_code == 422  # Validation error
    print("✓ Invalid data handling passed")


def main():
    """Run all API tests."""
    print("\n" + "="*60)
    print("Wine Bottle OCR API Tests")
    print("="*60)
    print("\nMake sure the API is running on http://localhost:8000")
    print("Start with: uvicorn src.api:app --reload")

    try:
        # Run tests
        test_health()
        test_info()
        test_extract_text()
        test_extract_text_invalid()

        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)

    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to API")
        print("Make sure the API is running on http://localhost:8000")
        sys.exit(1)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
