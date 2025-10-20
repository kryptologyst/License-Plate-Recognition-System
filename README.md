# License Plate Recognition System

A modular license plate recognition system using computer vision and OCR techniques for automatic number plate recognition (ANPR).

## Features

- **Multiple Detection Methods**: Traditional computer vision and advanced YOLO-based detection
- **Modern OCR**: Tesseract OCR with multiple configuration options for better accuracy
- **Web Interface**: Beautiful Streamlit web app for easy interaction and testing
- **Synthetic Data Generation**: Generate synthetic vehicle images with license plates for testing
- **Analytics Dashboard**: Track detection performance and statistics
- **Mock Database**: Store and query detection results
- **Configuration Management**: YAML-based configuration system
- **Type Hints & Documentation**: Full type annotations and comprehensive docstrings
- **Modular Architecture**: Clean, maintainable code structure

## 📁 Project Structure

```
├── src/                          # Source code
│   ├── __init__.py
│   ├── license_plate_detector.py # Traditional CV-based detector
│   ├── advanced_detector.py      # YOLO-based advanced detector
│   ├── config.py                 # Configuration management
│   └── data_generator.py         # Synthetic data generation
├── web_app/                      # Web interface
│   └── app.py                   # Streamlit application
├── data/                         # Data directory
│   └── synthetic/               # Generated synthetic images
├── models/                       # Model files (YOLO weights, etc.)
├── config/                       # Configuration files
├── tests/                        # Test files
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore file
└── README.md                    # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR installed on your system

### Install Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download and install from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

### Install Python Dependencies

```bash
# Clone the repository
git clone https://github.com/kryptologyst/License-Plate-Recognition-System.git
cd License-Plate-Recognition-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Synthetic Data (Optional)

```bash
python -c "from src.data_generator import main; main()"
```

This will generate 10 synthetic vehicle images with license plates in the `data/synthetic/` directory.

### 2. Run the Web Interface

```bash
streamlit run web_app/app.py
```

Open your browser and navigate to `http://localhost:8501` to access the web interface.

### 3. Use the Python API

```python
from src.license_plate_detector import LicensePlateDetector
from src.advanced_detector import YOLOLicensePlateDetector

# Traditional detector
detector = LicensePlateDetector()
result = detector.detect_license_plate("path/to/image.jpg")

if result:
    print(f"Detected plate: {result.plate_text}")
    print(f"Confidence: {result.confidence:.2f}")

# Advanced detector
advanced_detector = YOLOLicensePlateDetector()
result = advanced_detector.detect_license_plate("path/to/image.jpg")

if result:
    print(f"Detected plate: {result.plate_text}")
    print(f"Method: {result.detection_method}")
    print(f"Processing time: {result.processing_time:.2f}s")
```

## Usage Examples

### Basic Detection

```python
import cv2
from src.license_plate_detector import LicensePlateDetector

# Initialize detector
detector = LicensePlateDetector()

# Detect license plate
result = detector.detect_license_plate("car_image.jpg")

if result:
    print(f"License Plate: {result.plate_text}")
    print(f"Confidence: {result.confidence:.2f}")
    
    # Visualize result
    vis_image = detector.visualize_result(result)
    cv2.imshow("Detection Result", vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

### Advanced Detection with Multiple Methods

```python
from src.advanced_detector import YOLOLicensePlateDetector

# Initialize advanced detector
detector = YOLOLicensePlateDetector()

# Detect with advanced methods
result = detector.detect_license_plate("car_image.jpg")

if result:
    print(f"Plate: {result.plate_text}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Method: {result.detection_method}")
    print(f"Time: {result.processing_time:.2f}s")
```

### Generate Synthetic Data

```python
from src.data_generator import VehicleImageGenerator, MockDatabase

# Generate synthetic dataset
generator = VehicleImageGenerator()
dataset = generator.generate_dataset(20, "data/synthetic")

# Use mock database
db = MockDatabase()
db.add_detection(
    image_path="test.jpg",
    plate_text="ABC123",
    confidence=0.95,
    bounding_box=(100, 200, 300, 80),
    detection_method="Traditional",
    processing_time=1.2
)

# Query database
stats = db.get_statistics()
print(f"Total detections: {stats['total_detections']}")
```

## Configuration

The system uses YAML configuration files. Create `config/config.yaml`:

```yaml
detection:
  min_plate_area: 1000
  max_plate_area: 50000
  confidence_threshold: 0.5
  nms_threshold: 0.4
  tesseract_config: "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

model:
  yolo_model_path: null
  ocr_model_path: null
  use_gpu: true
  batch_size: 1

ui:
  streamlit_port: 8501
  streamlit_host: "localhost"
  debug_mode: false
  show_confidence: true
  show_processing_time: true

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file_path: null

app:
  data_dir: "data"
  models_dir: "models"
  output_dir: "output"
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_detector.py
```

## Web Interface Features

The Streamlit web interface provides:

1. **Image Upload**: Upload vehicle images for license plate detection
2. **Multiple Detection Methods**: Choose between traditional CV and advanced YOLO methods
3. **Real-time Results**: See detection results with confidence scores and processing times
4. **Analytics Dashboard**: View detection statistics and performance metrics
5. **Synthetic Data Generator**: Generate test images with license plates
6. **Detection History**: Track all previous detections

## 🔧 Advanced Usage

### Custom Detection Parameters

```python
from src.license_plate_detector import LicensePlateDetector

# Custom detector with specific parameters
detector = LicensePlateDetector(
    min_plate_area=500,
    max_plate_area=30000,
    tesseract_config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
)
```

### Batch Processing

```python
import os
from pathlib import Path
from src.license_plate_detector import LicensePlateDetector

detector = LicensePlateDetector()
results = []

# Process multiple images
image_dir = Path("images")
for image_path in image_dir.glob("*.jpg"):
    result = detector.detect_license_plate(str(image_path))
    if result:
        results.append({
            'image': image_path.name,
            'plate': result.plate_text,
            'confidence': result.confidence
        })

# Save results
import json
with open('batch_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenCV community for computer vision tools
- Tesseract OCR for text recognition
- Streamlit for the web interface
- The computer vision and machine learning communities

## Future Enhancements

- [ ] Real-time video processing
- [ ] Mobile app integration
- [ ] Cloud deployment options
- [ ] Advanced ML models (Transformers, etc.)
- [ ] Multi-language support
- [ ] API endpoints for integration
- [ ] Docker containerization
- [ ] Performance optimization
- [ ] Additional visualization features
- [ ] Export capabilities (PDF, Excel, etc.)


# License-Plate-Recognition-System
