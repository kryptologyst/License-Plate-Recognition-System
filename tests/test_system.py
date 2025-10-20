"""
Test suite for the License Plate Recognition System.

This module contains unit tests for all major components of the system.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os
import sys

# Add src to path for imports
sys.path.append('src')

from license_plate_detector import LicensePlateDetector, LicensePlateResult
from advanced_detector import YOLOLicensePlateDetector, AdvancedPlateResult
from data_generator import LicensePlateGenerator, VehicleImageGenerator, MockDatabase
from config import ConfigManager, DetectionConfig, ModelConfig


class TestLicensePlateDetector:
    """Test cases for the traditional license plate detector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = LicensePlateDetector()
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        assert self.detector.min_plate_area == 1000
        assert self.detector.max_plate_area == 50000
        assert isinstance(self.detector.tesseract_config, str)
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        # Create a test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test preprocessing
        processed = self.detector.preprocess_image(test_image)
        
        assert processed.shape == (100, 100)
        assert processed.dtype == np.uint8
    
    def test_find_plate_contours(self):
        """Test contour finding."""
        # Create a test edge image
        test_edges = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(test_edges, (20, 20), (80, 60), 255, -1)
        
        contours = self.detector.find_plate_contours(test_edges)
        
        assert isinstance(contours, list)
        assert len(contours) <= 10  # Should limit to top 10
    
    def test_is_valid_plate_shape(self):
        """Test plate shape validation."""
        # Create a rectangular contour
        rect_contour = np.array([[[10, 10]], [[50, 10]], [[50, 30]], [[10, 30]]], dtype=np.int32)
        
        # Should be valid
        assert self.detector.is_valid_plate_shape(rect_contour)
        
        # Create a triangular contour
        triangle_contour = np.array([[[10, 10]], [[30, 10]], [[20, 30]]], dtype=np.int32)
        
        # Should be invalid
        assert not self.detector.is_valid_plate_shape(triangle_contour)
    
    def test_extract_plate_region(self):
        """Test plate region extraction."""
        # Create test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Create contour
        contour = np.array([[[20, 20]], [[60, 20]], [[60, 40]], [[20, 40]]], dtype=np.int32)
        
        plate_img, bbox = self.detector.extract_plate_region(test_image, contour)
        
        assert plate_img.shape[0] > 0
        assert plate_img.shape[1] > 0
        assert len(bbox) == 4
        assert all(isinstance(x, int) for x in bbox)
    
    def test_enhance_plate_image(self):
        """Test plate image enhancement."""
        # Create test plate image
        test_plate = np.random.randint(0, 255, (50, 150), dtype=np.uint8)
        
        enhanced = self.detector.enhance_plate_image(test_plate)
        
        assert enhanced.shape[0] > 0
        assert enhanced.shape[1] > 0
        assert enhanced.dtype == np.uint8


class TestAdvancedDetector:
    """Test cases for the advanced YOLO-based detector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = YOLOLicensePlateDetector()
    
    def test_detector_initialization(self):
        """Test advanced detector initialization."""
        assert self.detector.confidence_threshold == 0.5
        assert self.detector.nms_threshold == 0.4
        assert isinstance(self.detector.class_names, list)
    
    def test_detect_plates_traditional(self):
        """Test traditional plate detection fallback."""
        # Create test image with rectangular region
        test_image = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (50, 50), (150, 100), (255, 255, 255), -1)
        
        detections = self.detector.detect_plates_traditional(test_image)
        
        assert isinstance(detections, list)
        # Should find the rectangular region
        assert len(detections) > 0
    
    def test_enhance_plate_for_ocr(self):
        """Test plate enhancement for OCR."""
        # Create test plate image
        test_plate = np.random.randint(0, 255, (50, 150, 3), dtype=np.uint8)
        
        enhanced = self.detector.enhance_plate_for_ocr(test_plate)
        
        assert enhanced.shape[0] > 0
        assert enhanced.shape[1] > 0
        assert len(enhanced.shape) == 2  # Should be grayscale


class TestDataGenerator:
    """Test cases for the synthetic data generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plate_generator = LicensePlateGenerator()
        self.vehicle_generator = VehicleImageGenerator()
    
    def test_plate_generator_initialization(self):
        """Test plate generator initialization."""
        assert self.plate_generator.plate_width == 300
        assert self.plate_generator.plate_height == 80
        assert self.plate_generator.font_size == 40
    
    def test_generate_random_plate_text(self):
        """Test random plate text generation."""
        plate_text = self.plate_generator.generate_random_plate_text()
        
        assert isinstance(plate_text, str)
        assert len(plate_text) > 0
        assert plate_text in self.plate_generator.plate_formats
    
    def test_generate_plate_image(self):
        """Test plate image generation."""
        plate_text = "ABC123"
        plate_image = self.plate_generator.generate_plate_image(plate_text)
        
        assert plate_image.shape == (self.plate_generator.plate_height, 
                                   self.plate_generator.plate_width, 3)
        assert plate_image.dtype == np.uint8
    
    def test_generate_multiple_plates(self):
        """Test multiple plate generation."""
        count = 5
        plates = self.plate_generator.generate_multiple_plates(count)
        
        assert len(plates) == count
        for plate_text, plate_image in plates:
            assert isinstance(plate_text, str)
            assert isinstance(plate_image, np.ndarray)
    
    def test_vehicle_generator_initialization(self):
        """Test vehicle generator initialization."""
        assert self.vehicle_generator.image_width == 800
        assert self.vehicle_generator.image_height == 600
    
    def test_generate_car_background(self):
        """Test car background generation."""
        background = self.vehicle_generator.generate_car_background()
        
        assert background.shape == (self.vehicle_generator.image_height,
                                  self.vehicle_generator.image_width, 3)
        assert background.dtype == np.uint8
    
    def test_generate_vehicle_with_plate(self):
        """Test complete vehicle generation."""
        vehicle_img, plate_text, bbox = self.vehicle_generator.generate_vehicle_with_plate()
        
        assert vehicle_img.shape == (self.vehicle_generator.image_height,
                                   self.vehicle_generator.image_width, 3)
        assert isinstance(plate_text, str)
        assert len(bbox) == 4
        assert all(isinstance(x, int) for x in bbox)


class TestMockDatabase:
    """Test cases for the mock database."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        self.db = MockDatabase(self.temp_file.name)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_database_initialization(self):
        """Test database initialization."""
        assert isinstance(self.db.data, list)
        assert len(self.db.data) == 0
    
    def test_add_detection(self):
        """Test adding detection to database."""
        self.db.add_detection(
            image_path="test.jpg",
            plate_text="ABC123",
            confidence=0.95,
            bounding_box=(100, 200, 300, 80),
            detection_method="Test",
            processing_time=1.2
        )
        
        assert len(self.db.data) == 1
        assert self.db.data[0]["plate_text"] == "ABC123"
        assert self.db.data[0]["confidence"] == 0.95
    
    def test_get_all_detections(self):
        """Test getting all detections."""
        # Add some test data
        self.db.add_detection("test1.jpg", "ABC123", 0.9, (0, 0, 100, 50), "Test", 1.0)
        self.db.add_detection("test2.jpg", "XYZ789", 0.8, (0, 0, 100, 50), "Test", 1.1)
        
        all_detections = self.db.get_all_detections()
        
        assert len(all_detections) == 2
        assert all_detections[0]["plate_text"] == "ABC123"
        assert all_detections[1]["plate_text"] == "XYZ789"
    
    def test_get_detections_by_plate(self):
        """Test getting detections by plate text."""
        # Add test data
        self.db.add_detection("test1.jpg", "ABC123", 0.9, (0, 0, 100, 50), "Test", 1.0)
        self.db.add_detection("test2.jpg", "XYZ789", 0.8, (0, 0, 100, 50), "Test", 1.1)
        self.db.add_detection("test3.jpg", "ABC123", 0.85, (0, 0, 100, 50), "Test", 1.2)
        
        abc_detections = self.db.get_detections_by_plate("ABC123")
        
        assert len(abc_detections) == 2
        assert all(det["plate_text"] == "ABC123" for det in abc_detections)
    
    def test_get_detections_by_confidence(self):
        """Test getting detections by confidence threshold."""
        # Add test data
        self.db.add_detection("test1.jpg", "ABC123", 0.9, (0, 0, 100, 50), "Test", 1.0)
        self.db.add_detection("test2.jpg", "XYZ789", 0.7, (0, 0, 100, 50), "Test", 1.1)
        self.db.add_detection("test3.jpg", "DEF456", 0.8, (0, 0, 100, 50), "Test", 1.2)
        
        high_conf_detections = self.db.get_detections_by_confidence(0.8)
        
        assert len(high_conf_detections) == 2
        assert all(det["confidence"] >= 0.8 for det in high_conf_detections)
    
    def test_get_statistics(self):
        """Test getting database statistics."""
        # Add test data
        self.db.add_detection("test1.jpg", "ABC123", 0.9, (0, 0, 100, 50), "Test", 1.0)
        self.db.add_detection("test2.jpg", "XYZ789", 0.8, (0, 0, 100, 50), "Test", 1.1)
        
        stats = self.db.get_statistics()
        
        assert stats["total_detections"] == 2
        assert stats["average_confidence"] == 0.85
        assert stats["unique_plates"] == 2


class TestConfigManager:
    """Test cases for configuration management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_config = tempfile.NamedTemporaryFile(delete=False, suffix='.yaml')
        self.temp_config.close()
        self.config_manager = ConfigManager(self.temp_config.name)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)
    
    def test_create_default_config(self):
        """Test default configuration creation."""
        config = self.config_manager._create_default_config()
        
        assert isinstance(config.detection, DetectionConfig)
        assert isinstance(config.model, ModelConfig)
        assert config.detection.min_plate_area == 1000
        assert config.detection.max_plate_area == 50000
    
    def test_load_config(self):
        """Test configuration loading."""
        config = self.config_manager.load_config()
        
        assert isinstance(config.detection, DetectionConfig)
        assert isinstance(config.model, ModelConfig)
    
    def test_save_config(self):
        """Test configuration saving."""
        config = self.config_manager.load_config()
        self.config_manager.save_config()
        
        # Verify file was created
        assert os.path.exists(self.temp_config.name)


# Integration tests
class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_synthetic_data(self):
        """Test end-to-end processing with synthetic data."""
        # Generate synthetic data
        generator = VehicleImageGenerator()
        vehicle_img, plate_text, bbox = generator.generate_vehicle_with_plate()
        
        # Test detection
        detector = LicensePlateDetector()
        result = detector.detect_license_plate_from_array(vehicle_img)
        
        # Note: Detection might not work perfectly on synthetic data
        # This test mainly ensures the system doesn't crash
        assert vehicle_img is not None
        assert isinstance(plate_text, str)
        assert len(bbox) == 4
    
    def test_database_integration(self):
        """Test database integration."""
        db = MockDatabase()
        
        # Add detection
        db.add_detection(
            image_path="test.jpg",
            plate_text="TEST123",
            confidence=0.95,
            bounding_box=(100, 200, 300, 80),
            detection_method="Test",
            processing_time=1.0
        )
        
        # Query database
        stats = db.get_statistics()
        assert stats["total_detections"] == 1
        assert stats["unique_plates"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
