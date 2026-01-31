"""
Wine Bottle OCR Reader

This module provides OCR functionality for extracting text from wine bottle images
using the EasyOCR library.
"""

import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import cv2
import easyocr
from PIL import Image, ImageDraw, ImageFont


class WineBottleOCR:
    """
    Wine Bottle OCR Reader using EasyOCR.

    This class provides methods to extract text from wine bottle images,
    with options for visualization and filtering.

    Attributes:
        reader: EasyOCR Reader instance
        languages: List of language codes for OCR
        confidence_threshold: Minimum confidence score for text detection
    """

    def __init__(
        self,
        languages: List[str] = ['en', 'fr', 'it', 'es'],
        confidence_threshold: float = 0.25,
        gpu: bool = True
    ):
        """
        Initialize the Wine Bottle OCR reader.

        Args:
            languages: List of language codes (e.g., ['en', 'fr', 'it', 'es'])
            confidence_threshold: Minimum confidence for text detection (0.0-1.0)
            gpu: Whether to use GPU acceleration
        """
        self.languages = languages
        self.confidence_threshold = confidence_threshold
        self.gpu = gpu

        # Initialize EasyOCR reader
        print(f"Initializing EasyOCR reader with languages: {', '.join(languages)}")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        print("EasyOCR reader initialized successfully")

    def read_text(
        self,
        image_path: str,
        detail: int = 1,
        paragraph: bool = False
    ) -> List[Tuple[List[List[int]], str, float]]:
        """
        Extract text from an image.

        Args:
            image_path: Path to the image file
            detail: Level of detail (0 = low, 1 = high)
            paragraph: Whether to combine text into paragraphs

        Returns:
            List of tuples containing (bounding_box, text, confidence)
            where bounding_box is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Read text from image
        results = self.reader.readtext(
            image_path,
            detail=detail,
            paragraph=paragraph
        )

        # Filter by confidence threshold
        filtered_results = [
            (bbox, text, conf)
            for bbox, text, conf in results
            if conf >= self.confidence_threshold
        ]

        return filtered_results

    def extract_all_text(
        self,
        image_path: str,
        visualize: bool = False,
        save_path: Optional[str] = None
    ) -> Tuple[int, List[Dict[str, any]]]:
        """
        Extract all text from a wine bottle image.

        Args:
            image_path: Path to the image file
            visualize: Whether to display the annotated image
            save_path: Optional path to save the annotated image

        Returns:
            Tuple of (text_count, text_detections)
            where text_detections is a list of dicts with keys:
            - text: The detected text string
            - confidence: Detection confidence
            - bbox: Bounding box coordinates
            - position: Center position (x, y)
        """
        # Read text from image
        results = self.read_text(image_path)

        # Format results
        detections = []
        for bbox, text, confidence in results:
            # Calculate center position
            bbox_array = np.array(bbox)
            center_x = int(np.mean(bbox_array[:, 0]))
            center_y = int(np.mean(bbox_array[:, 1]))

            detections.append({
                'text': text,
                'confidence': float(confidence),
                'bbox': bbox,
                'position': (center_x, center_y)
            })

        # Optionally visualize or save
        if visualize or save_path:
            annotated_image = self._annotate_image(image_path, results)

            if visualize:
                cv2.imshow('Wine Bottle OCR', annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if save_path:
                cv2.imwrite(save_path, annotated_image)
                print(f"Annotated image saved to: {save_path}")

        return len(detections), detections

    def _annotate_image(
        self,
        image_path: str,
        results: List[Tuple[List[List[int]], str, float]]
    ) -> np.ndarray:
        """
        Annotate image with detected text and bounding boxes.

        Args:
            image_path: Path to the original image
            results: OCR results from read_text()

        Returns:
            Annotated image as numpy array
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Draw bounding boxes and text
        for bbox, text, confidence in results:
            # Convert bbox to integer coordinates
            bbox_array = np.array(bbox, dtype=np.int32)

            # Draw bounding box
            cv2.polylines(
                image,
                [bbox_array],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2
            )

            # Prepare text label
            label = f"{text} ({confidence:.2f})"

            # Get text position (top-left corner of bbox)
            text_position = (bbox_array[0][0], bbox_array[0][1] - 10)

            # Draw text background
            (text_width, text_height), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )
            cv2.rectangle(
                image,
                (text_position[0], text_position[1] - text_height - 5),
                (text_position[0] + text_width, text_position[1] + 5),
                (0, 255, 0),
                -1
            )

            # Draw text
            cv2.putText(
                image,
                label,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        return image

    def get_text_summary(
        self,
        image_path: str
    ) -> Dict[str, any]:
        """
        Get a summary of all detected text in an image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing:
            - total_text_blocks: Number of text blocks detected
            - all_text: Combined text string
            - avg_confidence: Average confidence score
            - detections: List of individual text detections
        """
        count, detections = self.extract_all_text(image_path)

        if count == 0:
            return {
                'total_text_blocks': 0,
                'all_text': '',
                'avg_confidence': 0.0,
                'detections': []
            }

        all_text = ' '.join([d['text'] for d in detections])
        avg_confidence = sum([d['confidence'] for d in detections]) / count

        return {
            'total_text_blocks': count,
            'all_text': all_text,
            'avg_confidence': avg_confidence,
            'detections': detections
        }

    def batch_process(
        self,
        image_paths: List[str],
        save_dir: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Process multiple images in batch.

        Args:
            image_paths: List of paths to image files
            save_dir: Optional directory to save annotated images

        Returns:
            List of text summaries for each image
        """
        results = []

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for i, image_path in enumerate(image_paths):
            print(f"\nProcessing image {i+1}/{len(image_paths)}: {image_path}")

            try:
                # Get text summary
                summary = self.get_text_summary(image_path)
                summary['image_path'] = image_path

                # Save annotated image if requested
                if save_dir:
                    filename = Path(image_path).stem
                    save_path = os.path.join(save_dir, f"{filename}_ocr.jpg")
                    _, _ = self.extract_all_text(
                        image_path,
                        visualize=False,
                        save_path=save_path
                    )

                results.append(summary)
                print(f"  Detected {summary['total_text_blocks']} text blocks")
                print(f"  Text: {summary['all_text'][:100]}...")

            except Exception as e:
                print(f"  Error processing {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'total_text_blocks': 0
                })

        return results
