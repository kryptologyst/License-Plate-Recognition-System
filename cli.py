#!/usr/bin/env python3
"""
Command Line Interface for License Plate Recognition System.

This module provides a CLI for batch processing and system management.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional
import json
import time

# Add src to path
sys.path.append('src')

from license_plate_detector import LicensePlateDetector
from advanced_detector import YOLOLicensePlateDetector
from data_generator import VehicleImageGenerator, MockDatabase
from config import load_config, get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LicensePlateCLI:
    """Command Line Interface for the License Plate Recognition System."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.config = get_config()
        self.detector = LicensePlateDetector()
        self.advanced_detector = YOLOLicensePlateDetector()
        self.database = MockDatabase()
    
    def detect_single_image(self, image_path: str, method: str = "both", 
                          output_dir: Optional[str] = None) -> None:
        """
        Detect license plate in a single image.
        
        Args:
            image_path: Path to the input image
            method: Detection method ("traditional", "advanced", or "both")
            output_dir: Directory to save results
        """
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return
        
        logger.info(f"Processing image: {image_path}")
        
        results = []
        
        if method in ["traditional", "both"]:
            logger.info("Running traditional detection...")
            start_time = time.time()
            result = self.detector.detect_license_plate(str(image_path))
            processing_time = time.time() - start_time
            
            if result:
                results.append({
                    'method': 'Traditional CV',
                    'plate_text': result.plate_text,
                    'confidence': result.confidence,
                    'processing_time': processing_time,
                    'bounding_box': result.bounding_box
                })
                logger.info(f"Traditional: {result.plate_text} (confidence: {result.confidence:.2f})")
            else:
                logger.warning("Traditional detection failed")
        
        if method in ["advanced", "both"]:
            logger.info("Running advanced detection...")
            start_time = time.time()
            result = self.advanced_detector.detect_license_plate(str(image_path))
            processing_time = time.time() - start_time
            
            if result:
                results.append({
                    'method': 'Advanced (YOLO + OCR)',
                    'plate_text': result.plate_text,
                    'confidence': result.confidence,
                    'processing_time': processing_time,
                    'bounding_box': result.bounding_box,
                    'detection_method': result.detection_method
                })
                logger.info(f"Advanced: {result.plate_text} (confidence: {result.confidence:.2f})")
            else:
                logger.warning("Advanced detection failed")
        
        if not results:
            logger.error("No license plates detected")
            return
        
        # Save results
        if output_dir:
            self._save_results(results, image_path, output_dir)
        
        # Store in database
        for result in results:
            self.database.add_detection(
                image_path=str(image_path),
                plate_text=result['plate_text'],
                confidence=result['confidence'],
                bounding_box=result['bounding_box'],
                detection_method=result['method'],
                processing_time=result['processing_time']
            )
    
    def batch_process(self, input_dir: str, method: str = "both", 
                     output_dir: Optional[str] = None) -> None:
        """
        Process multiple images in batch.
        
        Args:
            input_dir: Directory containing input images
            method: Detection method
            output_dir: Directory to save results
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error(f"Input directory not found: {input_path}")
            return
        
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            logger.error(f"No image files found in {input_path}")
            return
        
        logger.info(f"Found {len(image_files)} images to process")
        
        successful_detections = 0
        total_processing_time = 0
        
        for i, image_file in enumerate(image_files, 1):
            logger.info(f"Processing {i}/{len(image_files)}: {image_file.name}")
            
            try:
                start_time = time.time()
                self.detect_single_image(str(image_file), method, output_dir)
                processing_time = time.time() - start_time
                total_processing_time += processing_time
                successful_detections += 1
                
            except Exception as e:
                logger.error(f"Error processing {image_file.name}: {e}")
        
        # Print summary
        logger.info(f"Batch processing completed:")
        logger.info(f"  Total images: {len(image_files)}")
        logger.info(f"  Successful detections: {successful_detections}")
        logger.info(f"  Total processing time: {total_processing_time:.2f}s")
        logger.info(f"  Average time per image: {total_processing_time/len(image_files):.2f}s")
    
    def generate_synthetic_data(self, count: int, output_dir: str) -> None:
        """
        Generate synthetic vehicle images with license plates.
        
        Args:
            count: Number of images to generate
            output_dir: Directory to save generated images
        """
        logger.info(f"Generating {count} synthetic vehicle images...")
        
        generator = VehicleImageGenerator()
        dataset = generator.generate_dataset(count, output_dir)
        
        logger.info(f"Generated {len(dataset)} synthetic images in {output_dir}")
        
        # Print sample of generated plates
        logger.info("Sample generated license plates:")
        for i, metadata in enumerate(dataset[:5]):
            logger.info(f"  {metadata['filename']}: {metadata['plate_text']}")
    
    def show_database_stats(self) -> None:
        """Show database statistics."""
        stats = self.database.get_statistics()
        
        print("\n📊 Database Statistics:")
        print(f"  Total detections: {stats['total_detections']}")
        print(f"  Average confidence: {stats['average_confidence']:.2f}")
        print(f"  Min confidence: {stats['min_confidence']:.2f}")
        print(f"  Max confidence: {stats['max_confidence']:.2f}")
        print(f"  Average processing time: {stats['average_processing_time']:.2f}s")
        print(f"  Unique license plates: {stats['unique_plates']}")
    
    def query_database(self, plate_text: Optional[str] = None, 
                      min_confidence: Optional[float] = None) -> None:
        """
        Query the database for specific detections.
        
        Args:
            plate_text: Filter by license plate text
            min_confidence: Filter by minimum confidence
        """
        if plate_text:
            detections = self.database.get_detections_by_plate(plate_text)
            print(f"\n🔍 Detections for plate '{plate_text}':")
        elif min_confidence:
            detections = self.database.get_detections_by_confidence(min_confidence)
            print(f"\n🔍 Detections with confidence >= {min_confidence}:")
        else:
            detections = self.database.get_all_detections()
            print(f"\n🔍 All detections:")
        
        if not detections:
            print("  No detections found")
            return
        
        for detection in detections[-10:]:  # Show last 10
            print(f"  {detection['plate_text']} - "
                  f"Confidence: {detection['confidence']:.2f} - "
                  f"Method: {detection['detection_method']} - "
                  f"Time: {detection['processing_time']:.2f}s")
    
    def _save_results(self, results: List[dict], image_path: Path, 
                     output_dir: str) -> None:
        """Save detection results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        results_file = output_path / f"{image_path.stem}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="License Plate Recognition System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect license plate in a single image
  python cli.py detect image.jpg
  
  # Use advanced detection method
  python cli.py detect image.jpg --method advanced
  
  # Batch process all images in a directory
  python cli.py batch-process /path/to/images --output-dir results/
  
  # Generate synthetic data
  python cli.py generate-synthetic 10 --output-dir data/synthetic
  
  # Show database statistics
  python cli.py stats
  
  # Query database for specific plate
  python cli.py query --plate ABC123
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect license plate in an image')
    detect_parser.add_argument('image', help='Path to input image')
    detect_parser.add_argument('--method', choices=['traditional', 'advanced', 'both'], 
                             default='both', help='Detection method')
    detect_parser.add_argument('--output-dir', help='Directory to save results')
    
    # Batch process command
    batch_parser = subparsers.add_parser('batch-process', help='Process multiple images')
    batch_parser.add_argument('input_dir', help='Directory containing input images')
    batch_parser.add_argument('--method', choices=['traditional', 'advanced', 'both'], 
                            default='both', help='Detection method')
    batch_parser.add_argument('--output-dir', help='Directory to save results')
    
    # Generate synthetic data command
    gen_parser = subparsers.add_parser('generate-synthetic', help='Generate synthetic data')
    gen_parser.add_argument('count', type=int, help='Number of images to generate')
    gen_parser.add_argument('--output-dir', default='data/synthetic', 
                           help='Directory to save generated images')
    
    # Database commands
    subparsers.add_parser('stats', help='Show database statistics')
    
    query_parser = subparsers.add_parser('query', help='Query database')
    query_parser.add_argument('--plate', help='Filter by license plate text')
    query_parser.add_argument('--min-confidence', type=float, 
                             help='Filter by minimum confidence')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = LicensePlateCLI()
    
    try:
        if args.command == 'detect':
            cli.detect_single_image(args.image, args.method, args.output_dir)
        
        elif args.command == 'batch-process':
            cli.batch_process(args.input_dir, args.method, args.output_dir)
        
        elif args.command == 'generate-synthetic':
            cli.generate_synthetic_data(args.count, args.output_dir)
        
        elif args.command == 'stats':
            cli.show_database_stats()
        
        elif args.command == 'query':
            cli.query_database(args.plate, args.min_confidence)
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
