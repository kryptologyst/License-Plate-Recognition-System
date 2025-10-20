"""
Synthetic data generator for license plate recognition system.

This module generates synthetic license plate images and vehicle images
with license plates for testing and demonstration purposes.
"""

import cv2
import numpy as np
import random
import string
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
import os

logger = logging.getLogger(__name__)


class LicensePlateGenerator:
    """Generates synthetic license plate images."""
    
    def __init__(self, 
                 plate_width: int = 300,
                 plate_height: int = 80,
                 font_size: int = 40):
        """
        Initialize license plate generator.
        
        Args:
            plate_width: Width of generated license plates
            plate_height: Height of generated license plates
            font_size: Font size for plate text
        """
        self.plate_width = plate_width
        self.plate_height = plate_height
        self.font_size = font_size
        
        # Common license plate formats
        self.plate_formats = [
            "ABC123",  # US format
            "AB12CD",  # UK format
            "123ABC",  # Alternative format
            "ABC1234", # Extended format
            "AB123CD", # Mixed format
        ]
        
        # License plate colors
        self.plate_colors = [
            (255, 255, 255),  # White background
            (255, 255, 0),    # Yellow background
            (200, 200, 200),  # Light gray background
        ]
        
        self.text_colors = [
            (0, 0, 0),        # Black text
            (0, 0, 255),      # Blue text
            (255, 0, 0),      # Red text
        ]
    
    def generate_random_plate_text(self) -> str:
        """Generate random license plate text."""
        return random.choice(self.plate_formats)
    
    def generate_plate_image(self, plate_text: str) -> np.ndarray:
        """
        Generate a synthetic license plate image.
        
        Args:
            plate_text: Text to display on the plate
            
        Returns:
            Generated license plate image as numpy array
        """
        # Create image
        img = Image.new('RGB', (self.plate_width, self.plate_height), 
                       random.choice(self.plate_colors))
        draw = ImageDraw.Draw(img)
        
        # Try to use a system font, fallback to default
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", self.font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", self.font_size)
            except:
                font = ImageFont.load_default()
        
        # Calculate text position (centered)
        bbox = draw.textbbox((0, 0), plate_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (self.plate_width - text_width) // 2
        y = (self.plate_height - text_height) // 2
        
        # Draw text
        draw.text((x, y), plate_text, fill=random.choice(self.text_colors), font=font)
        
        # Add border
        draw.rectangle([0, 0, self.plate_width-1, self.plate_height-1], 
                      outline=(0, 0, 0), width=2)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Add some noise and blur for realism
        noise = np.random.normal(0, 5, img_array.shape).astype(np.uint8)
        img_array = cv2.add(img_array, noise)
        
        # Apply slight blur
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
        
        return img_array
    
    def generate_multiple_plates(self, count: int) -> List[Tuple[str, np.ndarray]]:
        """
        Generate multiple license plate images.
        
        Args:
            count: Number of plates to generate
            
        Returns:
            List of (plate_text, plate_image) tuples
        """
        plates = []
        for _ in range(count):
            plate_text = self.generate_random_plate_text()
            plate_image = self.generate_plate_image(plate_text)
            plates.append((plate_text, plate_image))
        
        return plates


class VehicleImageGenerator:
    """Generates synthetic vehicle images with license plates."""
    
    def __init__(self, 
                 image_width: int = 800,
                 image_height: int = 600):
        """
        Initialize vehicle image generator.
        
        Args:
            image_width: Width of generated vehicle images
            image_height: Height of generated vehicle images
        """
        self.image_width = image_width
        self.image_height = image_height
        self.plate_generator = LicensePlateGenerator()
    
    def generate_car_background(self) -> np.ndarray:
        """Generate a simple car-like background."""
        # Create gradient background
        img = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        
        # Add gradient from top to bottom
        for y in range(self.image_height):
            intensity = int(100 + (y / self.image_height) * 100)
            img[y, :] = [intensity, intensity, intensity]
        
        # Add some car-like shapes
        # Car body (rectangle)
        car_x = self.image_width // 4
        car_y = self.image_height // 3
        car_w = self.image_width // 2
        car_h = self.image_height // 2
        
        cv2.rectangle(img, (car_x, car_y), (car_x + car_w, car_y + car_h), 
                     (50, 50, 50), -1)
        
        # Car roof (trapezoid)
        roof_points = np.array([
            [car_x + car_w//4, car_y],
            [car_x + 3*car_w//4, car_y],
            [car_x + car_w, car_y + car_h//3],
            [car_x, car_y + car_h//3]
        ], np.int32)
        cv2.fillPoly(img, [roof_points], (40, 40, 40))
        
        # Wheels
        wheel_radius = 30
        cv2.circle(img, (car_x + car_w//4, car_y + car_h), wheel_radius, (20, 20, 20), -1)
        cv2.circle(img, (car_x + 3*car_w//4, car_y + car_h), wheel_radius, (20, 20, 20), -1)
        
        return img
    
    def add_license_plate_to_image(self, 
                                 background: np.ndarray, 
                                 plate_text: str) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Add a license plate to the vehicle image.
        
        Args:
            background: Background vehicle image
            plate_text: License plate text
            
        Returns:
            Tuple of (image_with_plate, plate_bounding_box)
        """
        # Generate license plate
        plate_img = self.plate_generator.generate_plate_image(plate_text)
        
        # Resize plate to fit nicely
        plate_h = 60
        plate_w = int(plate_img.shape[1] * plate_h / plate_img.shape[0])
        plate_img = cv2.resize(plate_img, (plate_w, plate_h))
        
        # Position plate on the car (rear bumper area)
        plate_x = self.image_width // 2 - plate_w // 2
        plate_y = self.image_height - plate_h - 20
        
        # Ensure plate is within image bounds
        plate_x = max(0, min(plate_x, self.image_width - plate_w))
        plate_y = max(0, min(plate_y, self.image_height - plate_h))
        
        # Create result image
        result = background.copy()
        
        # Add plate to image
        result[plate_y:plate_y+plate_h, plate_x:plate_x+plate_w] = plate_img
        
        # Add some shadow/reflection effect
        shadow = np.zeros((plate_h, plate_w, 3), dtype=np.uint8)
        cv2.rectangle(shadow, (0, 0), (plate_w-1, plate_h-1), (0, 0, 0), 2)
        
        # Blend shadow
        alpha = 0.3
        result[plate_y:plate_y+plate_h, plate_x:plate_x+plate_w] = \
            cv2.addWeighted(result[plate_y:plate_y+plate_h, plate_x:plate_x+plate_w], 
                           1-alpha, shadow, alpha, 0)
        
        bounding_box = (plate_x, plate_y, plate_w, plate_h)
        
        return result, bounding_box
    
    def generate_vehicle_with_plate(self, plate_text: Optional[str] = None) -> Tuple[np.ndarray, str, Tuple[int, int, int, int]]:
        """
        Generate a complete vehicle image with license plate.
        
        Args:
            plate_text: Optional specific plate text, otherwise random
            
        Returns:
            Tuple of (vehicle_image, plate_text, plate_bounding_box)
        """
        if plate_text is None:
            plate_text = self.plate_generator.generate_random_plate_text()
        
        # Generate background
        background = self.generate_car_background()
        
        # Add license plate
        vehicle_img, bounding_box = self.add_license_plate_to_image(background, plate_text)
        
        return vehicle_img, plate_text, bounding_box
    
    def generate_dataset(self, 
                        count: int, 
                        output_dir: str = "data/synthetic") -> List[Dict[str, Any]]:
        """
        Generate a synthetic dataset of vehicle images with license plates.
        
        Args:
            count: Number of images to generate
            output_dir: Directory to save images
            
        Returns:
            List of metadata dictionaries
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        dataset_metadata = []
        
        for i in range(count):
            # Generate vehicle with plate
            vehicle_img, plate_text, bounding_box = self.generate_vehicle_with_plate()
            
            # Save image
            image_filename = f"vehicle_{i:04d}.jpg"
            image_path = output_path / image_filename
            
            cv2.imwrite(str(image_path), vehicle_img)
            
            # Create metadata
            metadata = {
                "image_id": i,
                "filename": image_filename,
                "plate_text": plate_text,
                "bounding_box": bounding_box,
                "image_size": (self.image_width, self.image_height),
                "plate_size": (bounding_box[2], bounding_box[3])
            }
            
            dataset_metadata.append(metadata)
            
            logger.info(f"Generated image {i+1}/{count}: {plate_text}")
        
        # Save metadata
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        logger.info(f"Generated {count} synthetic images in {output_dir}")
        logger.info(f"Metadata saved to {metadata_path}")
        
        return dataset_metadata


class MockDatabase:
    """Mock database for storing license plate detection results."""
    
    def __init__(self, db_path: str = "data/mock_database.json"):
        """
        Initialize mock database.
        
        Args:
            db_path: Path to database file
        """
        self.db_path = Path(db_path)
        self.data = []
        self.load_data()
    
    def load_data(self) -> None:
        """Load data from database file."""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    self.data = json.load(f)
                logger.info(f"Loaded {len(self.data)} records from database")
            except Exception as e:
                logger.error(f"Failed to load database: {e}")
                self.data = []
        else:
            logger.info("Database file not found, starting with empty database")
            self.data = []
    
    def save_data(self) -> None:
        """Save data to database file."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.db_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            logger.info(f"Saved {len(self.data)} records to database")
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
    
    def add_detection(self, 
                     image_path: str,
                     plate_text: str,
                     confidence: float,
                     bounding_box: Tuple[int, int, int, int],
                     detection_method: str,
                     processing_time: float) -> None:
        """
        Add a detection result to the database.
        
        Args:
            image_path: Path to the processed image
            plate_text: Detected license plate text
            confidence: Detection confidence score
            bounding_box: Plate bounding box coordinates
            detection_method: Method used for detection
            processing_time: Time taken for processing
        """
        import time
        
        record = {
            "id": len(self.data),
            "timestamp": time.time(),
            "image_path": image_path,
            "plate_text": plate_text,
            "confidence": confidence,
            "bounding_box": bounding_box,
            "detection_method": detection_method,
            "processing_time": processing_time
        }
        
        self.data.append(record)
        self.save_data()
    
    def get_all_detections(self) -> List[Dict[str, Any]]:
        """Get all detection records."""
        return self.data.copy()
    
    def get_detections_by_plate(self, plate_text: str) -> List[Dict[str, Any]]:
        """Get all detections for a specific license plate."""
        return [record for record in self.data if record["plate_text"] == plate_text]
    
    def get_detections_by_confidence(self, min_confidence: float) -> List[Dict[str, Any]]:
        """Get all detections above a confidence threshold."""
        return [record for record in self.data if record["confidence"] >= min_confidence]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.data:
            return {"total_detections": 0}
        
        confidences = [record["confidence"] for record in self.data]
        processing_times = [record["processing_time"] for record in self.data]
        
        return {
            "total_detections": len(self.data),
            "average_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "average_processing_time": sum(processing_times) / len(processing_times),
            "unique_plates": len(set(record["plate_text"] for record in self.data))
        }


def main():
    """Main function for testing the synthetic data generator."""
    import matplotlib.pyplot as plt
    
    # Generate synthetic dataset
    generator = VehicleImageGenerator()
    dataset = generator.generate_dataset(10, "data/synthetic")
    
    # Test mock database
    db = MockDatabase()
    
    # Add some sample detections
    for i, metadata in enumerate(dataset[:3]):
        db.add_detection(
            image_path=f"data/synthetic/vehicle_{i:04d}.jpg",
            plate_text=metadata["plate_text"],
            confidence=random.uniform(0.7, 0.95),
            bounding_box=metadata["bounding_box"],
            detection_method="Synthetic",
            processing_time=random.uniform(0.5, 2.0)
        )
    
    # Print statistics
    stats = db.get_statistics()
    print("Database Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Display sample images
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(min(6, len(dataset))):
        img_path = f"data/synthetic/vehicle_{i:04d}.jpg"
        if Path(img_path).exists():
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)
            axes[i].set_title(f"Plate: {dataset[i]['plate_text']}")
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
