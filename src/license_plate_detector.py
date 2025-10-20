"""
Core license plate detection and recognition module.

This module provides the main functionality for detecting license plates
in images and extracting text using modern computer vision techniques.
"""

import cv2
import numpy as np
import pytesseract
from typing import List, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LicensePlateResult:
    """Data class to store license plate detection results."""
    plate_text: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    plate_image: np.ndarray
    original_image: np.ndarray


class LicensePlateDetector:
    """
    Modern license plate detector using computer vision techniques.
    
    This class provides methods for detecting and recognizing license plates
    in vehicle images using OpenCV and Tesseract OCR.
    """
    
    def __init__(self, 
                 min_plate_area: int = 1000,
                 max_plate_area: int = 50000,
                 tesseract_config: str = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
        """
        Initialize the license plate detector.
        
        Args:
            min_plate_area: Minimum area for a valid license plate
            max_plate_area: Maximum area for a valid license plate
            tesseract_config: Tesseract OCR configuration
        """
        self.min_plate_area = min_plate_area
        self.max_plate_area = max_plate_area
        self.tesseract_config = tesseract_config
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for better plate detection.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply edge detection
        edged = cv2.Canny(blurred, 30, 200)
        
        return edged
    
    def find_plate_contours(self, edged_image: np.ndarray) -> List[np.ndarray]:
        """
        Find potential license plate contours in the edge-detected image.
        
        Args:
            edged_image: Edge-detected image
            
        Returns:
            List of potential plate contours
        """
        contours, _ = cv2.findContours(
            edged_image.copy(), 
            cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Sort contours by area and keep top candidates
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        return contours
    
    def is_valid_plate_shape(self, contour: np.ndarray) -> bool:
        """
        Check if a contour has a valid license plate shape.
        
        Args:
            contour: Contour to check
            
        Returns:
            True if contour resembles a license plate
        """
        # Approximate the contour
        epsilon = 0.018 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if it's roughly rectangular (4 corners)
        if len(approx) == 4:
            # Check area constraints
            area = cv2.contourArea(contour)
            return self.min_plate_area < area < self.max_plate_area
        
        return False
    
    def extract_plate_region(self, image: np.ndarray, contour: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract the license plate region from the image.
        
        Args:
            image: Original image
            contour: License plate contour
            
        Returns:
            Tuple of (plate_image, bounding_box)
        """
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract plate region
        plate_img = image[y:y+h, x:x+w]
        
        return plate_img, (x, y, w, h)
    
    def enhance_plate_image(self, plate_image: np.ndarray) -> np.ndarray:
        """
        Enhance the license plate image for better OCR results.
        
        Args:
            plate_image: Cropped license plate image
            
        Returns:
            Enhanced plate image
        """
        # Convert to grayscale if needed
        if len(plate_image.shape) == 3:
            plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        plate_image = cv2.adaptiveThreshold(
            plate_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        plate_image = cv2.morphologyEx(plate_image, cv2.MORPH_CLOSE, kernel)
        
        return plate_image
    
    def extract_text_from_plate(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """
        Extract text from the license plate image using OCR.
        
        Args:
            plate_image: Enhanced license plate image
            
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        try:
            # Use pytesseract to extract text
            plate_text = pytesseract.image_to_string(
                plate_image, 
                config=self.tesseract_config
            ).strip()
            
            # Get confidence score
            data = pytesseract.image_to_data(
                plate_image, 
                config=self.tesseract_config, 
                output_type=pytesseract.Output.DICT
            )
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return plate_text, confidence / 100.0  # Normalize to 0-1
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return "", 0.0
    
    def detect_license_plate(self, image_path: str) -> Optional[LicensePlateResult]:
        """
        Detect and recognize license plate from an image file.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            LicensePlateResult object or None if no plate detected
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return None
            
            logger.info(f"Processing image: {image_path}")
            
            # Preprocess image
            edged = self.preprocess_image(image)
            
            # Find contours
            contours = self.find_plate_contours(edged)
            
            # Look for valid plate shapes
            license_plate_contour = None
            for contour in contours:
                if self.is_valid_plate_shape(contour):
                    license_plate_contour = contour
                    break
            
            if license_plate_contour is None:
                logger.warning("No valid license plate contour found")
                return None
            
            # Extract plate region
            plate_img, bounding_box = self.extract_plate_region(image, license_plate_contour)
            
            # Enhance plate image
            enhanced_plate = self.enhance_plate_image(plate_img)
            
            # Extract text
            plate_text, confidence = self.extract_text_from_plate(enhanced_plate)
            
            if not plate_text:
                logger.warning("No text extracted from license plate")
                return None
            
            logger.info(f"Detected license plate: {plate_text} (confidence: {confidence:.2f})")
            
            return LicensePlateResult(
                plate_text=plate_text,
                confidence=confidence,
                bounding_box=bounding_box,
                plate_image=enhanced_plate,
                original_image=image
            )
            
        except Exception as e:
            logger.error(f"License plate detection failed: {e}")
            return None
    
    def detect_license_plate_from_array(self, image: np.ndarray) -> Optional[LicensePlateResult]:
        """
        Detect and recognize license plate from a numpy array image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            LicensePlateResult object or None if no plate detected
        """
        try:
            logger.info("Processing image array")
            
            # Preprocess image
            edged = self.preprocess_image(image)
            
            # Find contours
            contours = self.find_plate_contours(edged)
            
            # Look for valid plate shapes
            license_plate_contour = None
            for contour in contours:
                if self.is_valid_plate_shape(contour):
                    license_plate_contour = contour
                    break
            
            if license_plate_contour is None:
                logger.warning("No valid license plate contour found")
                return None
            
            # Extract plate region
            plate_img, bounding_box = self.extract_plate_region(image, license_plate_contour)
            
            # Enhance plate image
            enhanced_plate = self.enhance_plate_image(plate_img)
            
            # Extract text
            plate_text, confidence = self.extract_text_from_plate(enhanced_plate)
            
            if not plate_text:
                logger.warning("No text extracted from license plate")
                return None
            
            logger.info(f"Detected license plate: {plate_text} (confidence: {confidence:.2f})")
            
            return LicensePlateResult(
                plate_text=plate_text,
                confidence=confidence,
                bounding_box=bounding_box,
                plate_image=enhanced_plate,
                original_image=image
            )
            
        except Exception as e:
            logger.error(f"License plate detection failed: {e}")
            return None
    
    def visualize_result(self, result: LicensePlateResult) -> np.ndarray:
        """
        Create a visualization of the detection result.
        
        Args:
            result: LicensePlateResult object
            
        Returns:
            Image with bounding box and text overlay
        """
        image_display = result.original_image.copy()
        x, y, w, h = result.bounding_box
        
        # Draw bounding box
        cv2.rectangle(image_display, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Add text label
        label = f'Plate: {result.plate_text} ({result.confidence:.2f})'
        cv2.putText(image_display, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        return image_display


def main():
    """Main function for testing the license plate detector."""
    import matplotlib.pyplot as plt
    
    # Initialize detector
    detector = LicensePlateDetector()
    
    # Test with sample image (if available)
    sample_image = "data/sample_car.jpg"
    if Path(sample_image).exists():
        result = detector.detect_license_plate(sample_image)
        
        if result:
            print(f"Detected License Plate: {result.plate_text}")
            print(f"Confidence: {result.confidence:.2f}")
            
            # Visualize result
            vis_image = detector.visualize_result(result)
            vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(12, 6))
            plt.imshow(vis_image_rgb)
            plt.title(f"License Plate Detected: {result.plate_text}")
            plt.axis('off')
            plt.show()
        else:
            print("No license plate detected")
    else:
        print(f"Sample image not found: {sample_image}")


if __name__ == "__main__":
    main()
