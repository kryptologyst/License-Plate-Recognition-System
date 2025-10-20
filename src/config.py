"""
Configuration management for the license plate recognition system.

This module provides configuration loading and management capabilities
using YAML files for easy customization.
"""

import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    """Configuration for license plate detection."""
    min_plate_area: int = 1000
    max_plate_area: int = 50000
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    tesseract_config: str = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    yolo_model_path: Optional[str] = None
    ocr_model_path: Optional[str] = None
    use_gpu: bool = True
    batch_size: int = 1


@dataclass
class UIConfig:
    """Configuration for user interface."""
    streamlit_port: int = 8501
    streamlit_host: str = "localhost"
    debug_mode: bool = False
    show_confidence: bool = True
    show_processing_time: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None


@dataclass
class AppConfig:
    """Main application configuration."""
    detection: DetectionConfig
    model: ModelConfig
    ui: UIConfig
    logging: LoggingConfig
    data_dir: str = "data"
    models_dir: str = "models"
    output_dir: str = "output"


class ConfigManager:
    """Manages application configuration loading and saving."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config: Optional[AppConfig] = None
        
    def load_config(self) -> AppConfig:
        """
        Load configuration from file.
        
        Returns:
            AppConfig object
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Create config objects
                detection_config = DetectionConfig(**config_data.get('detection', {}))
                model_config = ModelConfig(**config_data.get('model', {}))
                ui_config = UIConfig(**config_data.get('ui', {}))
                logging_config = LoggingConfig(**config_data.get('logging', {}))
                
                self.config = AppConfig(
                    detection=detection_config,
                    model=model_config,
                    ui=ui_config,
                    logging=logging_config,
                    **config_data.get('app', {})
                )
                
                logger.info(f"Configuration loaded from {self.config_path}")
                
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                self.config = self._create_default_config()
        else:
            logger.warning(f"Configuration file not found: {self.config_path}")
            self.config = self._create_default_config()
            self.save_config()
        
        return self.config
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        if self.config is None:
            logger.error("No configuration to save")
            return
        
        try:
            # Ensure config directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dictionary
            config_dict = {
                'detection': asdict(self.config.detection),
                'model': asdict(self.config.model),
                'ui': asdict(self.config.ui),
                'logging': asdict(self.config.logging),
                'app': {
                    'data_dir': self.config.data_dir,
                    'models_dir': self.config.models_dir,
                    'output_dir': self.config.output_dir
                }
            }
            
            with open(self.config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def _create_default_config(self) -> AppConfig:
        """Create default configuration."""
        return AppConfig(
            detection=DetectionConfig(),
            model=ModelConfig(),
            ui=UIConfig(),
            logging=LoggingConfig()
        )
    
    def get_config(self) -> AppConfig:
        """Get current configuration."""
        if self.config is None:
            return self.load_config()
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values."""
        if self.config is None:
            self.config = self._create_default_config()
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get the global configuration."""
    return config_manager.get_config()


def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    """Load configuration from file."""
    global config_manager
    config_manager = ConfigManager(config_path)
    return config_manager.load_config()


def save_config() -> None:
    """Save current configuration."""
    config_manager.save_config()


def update_config(**kwargs) -> None:
    """Update configuration with new values."""
    config_manager.update_config(**kwargs)
