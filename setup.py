#!/usr/bin/env python3
"""
Setup script for License Plate Recognition System.

This script helps set up the project environment and dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path
import platform


def run_command(command: str, description: str) -> bool:
    """
    Run a command and return success status.
    
    Args:
        command: Command to run
        description: Description of what the command does
        
    Returns:
        True if command succeeded, False otherwise
    """
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported.")
        print("Please use Python 3.8 or higher.")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def check_tesseract():
    """Check if Tesseract OCR is installed."""
    print("🔍 Checking Tesseract OCR installation...")
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Tesseract OCR is installed")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Tesseract OCR is not installed")
    print("Please install Tesseract OCR:")
    
    system = platform.system().lower()
    if system == "darwin":  # macOS
        print("  brew install tesseract")
    elif system == "linux":
        print("  sudo apt-get install tesseract-ocr")
    elif system == "windows":
        print("  Download from: https://github.com/UB-Mannheim/tesseract/wiki")
    
    return False


def create_directories():
    """Create necessary directories."""
    print("📁 Creating project directories...")
    
    directories = [
        "data",
        "data/synthetic",
        "models",
        "output",
        "output/visualizations",
        "config",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created directory: {directory}")
    
    return True


def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing Python dependencies...")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    # Install dependencies
    success = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing dependencies from requirements.txt"
    )
    
    return success


def generate_sample_config():
    """Generate sample configuration file."""
    print("⚙️ Generating sample configuration...")
    
    config_content = """detection:
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
"""
    
    config_file = Path("config/config.yaml")
    if not config_file.exists():
        config_file.write_text(config_content)
        print("  ✅ Created sample configuration file")
    else:
        print("  ℹ️ Configuration file already exists")
    
    return True


def generate_sample_data():
    """Generate sample synthetic data."""
    print("🎲 Generating sample synthetic data...")
    
    try:
        # Import and run data generator
        sys.path.append('src')
        from data_generator import VehicleImageGenerator
        
        generator = VehicleImageGenerator()
        dataset = generator.generate_dataset(5, "data/synthetic")
        
        print(f"  ✅ Generated {len(dataset)} sample images")
        return True
        
    except Exception as e:
        print(f"  ⚠️ Could not generate sample data: {e}")
        return False


def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    
    success = run_command(
        f"{sys.executable} -m pytest tests/ -v",
        "Running test suite"
    )
    
    return success


def main():
    """Main setup function."""
    print("🚀 License Plate Recognition System Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_tesseract():
        print("\n⚠️ Please install Tesseract OCR and run setup again")
        sys.exit(1)
    
    # Setup steps
    steps = [
        ("Creating directories", create_directories),
        ("Installing dependencies", install_dependencies),
        ("Generating configuration", generate_sample_config),
        ("Generating sample data", generate_sample_data),
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}")
        if not step_func():
            failed_steps.append(step_name)
    
    # Optional: Run tests
    print(f"\n🧪 Running tests (optional)")
    run_tests()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Setup Summary")
    
    if failed_steps:
        print(f"❌ Failed steps: {', '.join(failed_steps)}")
        print("\nYou may need to resolve these issues manually.")
    else:
        print("✅ All setup steps completed successfully!")
    
    print("\n🚀 Next steps:")
    print("1. Run the web interface: streamlit run web_app/app.py")
    print("2. Or use the CLI: python cli.py --help")
    print("3. Check the README.md for detailed usage instructions")
    
    print("\n📚 Documentation:")
    print("- README.md: Complete usage guide")
    print("- src/: Source code with docstrings")
    print("- tests/: Test suite")
    print("- config/: Configuration files")


if __name__ == "__main__":
    main()
