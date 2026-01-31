"""
FastAPI Wine Bottle OCR API

This API provides endpoints for extracting text from wine bottle images using EasyOCR.

Endpoints:
    POST /api/v1/extract - Extract text from a base64-encoded image
    GET /health - Health check endpoint
    GET /api/v1/info - Get API and model information

Usage:
    uvicorn src.api:app --host 0.0.0.0 --port 8000

    Or with reload for development:
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
"""

import base64
import io
import os
from typing import List, Optional, Dict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import numpy as np
from PIL import Image
import cv2

from src.utils.ocr_reader import WineBottleOCR


# Pydantic models for request/response
class OCRRequest(BaseModel):
    """Request model for wine bottle OCR."""
    image: str = Field(
        ...,
        description="Base64-encoded image (JPEG, PNG, etc.)",
        example="data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    )
    confidence: Optional[float] = Field(
        0.25,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for text detection (0.0-1.0)"
    )
    languages: Optional[List[str]] = Field(
        ['en', 'fr', 'it', 'es'],
        description="Language codes for OCR (e.g., en, fr, it, es, de)"
    )
    return_annotated: Optional[bool] = Field(
        True,
        description="Whether to return the annotated image with text boxes"
    )

    @validator('image')
    def validate_base64(cls, v):
        """Validate that the image is valid base64."""
        try:
            # Handle data URI scheme
            if v.startswith('data:image'):
                v = v.split(',', 1)[1]

            # Try to decode
            base64.b64decode(v)
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 image data: {str(e)}")


class TextDetection(BaseModel):
    """Single text detection result."""
    text: str = Field(..., description="Detected text string")
    confidence: float = Field(..., description="Detection confidence score (0-1)")
    bbox: List[List[int]] = Field(..., description="Bounding box [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]")
    position: List[int] = Field(..., description="Center position [x, y]")


class OCRResponse(BaseModel):
    """Response model for wine bottle OCR."""
    success: bool = Field(..., description="Whether the OCR was successful")
    total_text_blocks: int = Field(..., description="Total number of text blocks detected")
    all_text: str = Field(..., description="Combined text from all detections")
    avg_confidence: float = Field(..., description="Average confidence score")
    detections: List[TextDetection] = Field(..., description="List of individual text detections")
    languages_used: List[str] = Field(..., description="Languages used for OCR")
    confidence_threshold: float = Field(..., description="Confidence threshold applied")
    annotated_image: Optional[str] = Field(
        None,
        description="Base64-encoded annotated image with text boxes (if requested)"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    version: str


class InfoResponse(BaseModel):
    """API information response."""
    service: str
    version: str
    description: str
    supported_languages: List[str]
    default_languages: List[str]
    default_confidence: float
    endpoints: List[str]


# Initialize FastAPI app
app = FastAPI(
    title="Wine Bottle OCR API",
    description="EasyOCR-based text extraction service for wine bottle images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# Global OCR reader instance cache
ocr_cache = {}


def get_ocr_reader(languages: List[str], confidence: float) -> WineBottleOCR:
    """
    Get or create an OCR reader instance for the specified configuration.

    Args:
        languages: List of language codes
        confidence: Confidence threshold

    Returns:
        WineBottleOCR instance
    """
    cache_key = f"{'-'.join(sorted(languages))}_{confidence}"

    if cache_key not in ocr_cache:
        ocr_cache[cache_key] = WineBottleOCR(
            languages=languages,
            confidence_threshold=confidence,
            gpu=True  # Set to False if GPU not available
        )

    return ocr_cache[cache_key]


def decode_image(base64_string: str) -> np.ndarray:
    """
    Decode a base64 string to a numpy array (OpenCV image).

    Args:
        base64_string: Base64-encoded image

    Returns:
        OpenCV image as numpy array
    """
    # Remove data URI prefix if present
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',', 1)[1]

    # Decode base64 to bytes
    image_bytes = base64.b64decode(base64_string)

    # Convert to PIL Image
    pil_image = Image.open(io.BytesIO(image_bytes))

    # Convert to OpenCV format (BGR)
    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return opencv_image


def encode_image(image: np.ndarray) -> str:
    """
    Encode an OpenCV image to base64 string.

    Args:
        image: OpenCV image as numpy array

    Returns:
        Base64-encoded image string
    """
    # Encode image to JPEG
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Failed to encode image")

    # Convert to base64
    jpg_bytes = buffer.tobytes()
    base64_string = base64.b64encode(jpg_bytes).decode('utf-8')

    return f"data:image/jpeg;base64,{base64_string}"


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.

    Returns:
        Health status information
    """
    return HealthResponse(
        status="healthy",
        service="wine-bottle-ocr-api",
        version="1.0.0"
    )


@app.get("/api/v1/info", response_model=InfoResponse)
async def get_info():
    """
    Get API information including supported languages and endpoints.

    Returns:
        API configuration and capabilities
    """
    return InfoResponse(
        service="Wine Bottle OCR API",
        version="1.0.0",
        description="EasyOCR-based text extraction from wine bottle images",
        supported_languages=[
            "en (English)", "fr (French)", "it (Italian)", "es (Spanish)",
            "de (German)", "pt (Portuguese)", "ru (Russian)", "ar (Arabic)",
            "zh (Chinese)", "ja (Japanese)", "ko (Korean)"
        ],
        default_languages=["en", "fr", "it", "es"],
        default_confidence=0.25,
        endpoints=["/health", "/api/v1/info", "/api/v1/extract"]
    )


@app.post("/api/v1/extract", response_model=OCRResponse)
async def extract_text(request: OCRRequest):
    """
    Extract text from a base64-encoded wine bottle image.

    Args:
        request: OCRRequest containing base64 image and parameters

    Returns:
        OCR results including detected text, confidence scores, and optionally annotated image

    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/extract" \\
             -H "Content-Type: application/json" \\
             -d '{
                   "image": "data:image/jpeg;base64,/9j/4AAQ...",
                   "confidence": 0.25,
                   "languages": ["en", "fr"],
                   "return_annotated": true
                 }'
        ```
    """
    try:
        # Decode the image
        image = decode_image(request.image)

        # Save to temporary file for OCR processing
        temp_dir = Path("/tmp/wine_bottle_ocr_api")
        temp_dir.mkdir(exist_ok=True)
        temp_image_path = temp_dir / "temp_input.jpg"
        cv2.imwrite(str(temp_image_path), image)

        # Get OCR reader instance
        ocr_reader = get_ocr_reader(request.languages, request.confidence)

        # Extract text
        count, detections = ocr_reader.extract_all_text(
            str(temp_image_path),
            visualize=False,
            save_path=None
        )

        # Calculate summary statistics
        all_text = ' '.join([d['text'] for d in detections]) if detections else ''
        avg_confidence = (
            sum([d['confidence'] for d in detections]) / count
            if count > 0 else 0.0
        )

        # Format detections
        formatted_detections = [
            TextDetection(
                text=det['text'],
                confidence=det['confidence'],
                bbox=det['bbox'],
                position=list(det['position'])
            )
            for det in detections
        ]

        # Generate annotated image if requested
        annotated_image_base64 = None
        if request.return_annotated and count > 0:
            results = ocr_reader.read_text(str(temp_image_path))
            annotated_image = ocr_reader._annotate_image(str(temp_image_path), results)
            annotated_image_base64 = encode_image(annotated_image)

        # Clean up temp file
        temp_image_path.unlink(missing_ok=True)

        # Return response
        return OCRResponse(
            success=True,
            total_text_blocks=count,
            all_text=all_text,
            avg_confidence=avg_confidence,
            detections=formatted_detections,
            languages_used=request.languages,
            confidence_threshold=request.confidence,
            annotated_image=annotated_image_base64
        )

    except Exception as e:
        # Clean up temp file on error
        if 'temp_image_path' in locals():
            temp_image_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"OCR extraction failed: {str(e)}"
        )


@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "service": "Wine Bottle OCR API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "extract": "POST /api/v1/extract",
            "info": "GET /api/v1/info"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
