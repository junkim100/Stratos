import yaml
from pathlib import Path
from typing import Dict, Any


class Settings:
    def __init__(self):
        # Load YAML config
        self.config = self._load_yaml_config()

        # Initialize settings
        self._initialize_settings()

    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = Path(__file__).parent.parent.parent / "config.yml"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _initialize_settings(self) -> None:
        """Initialize all settings from YAML"""
        # Model settings from YAML
        self.GENERATION_MODEL: str = self.config["model"]["generation"]["name"]
        self.GENERATION_CONFIG: Dict[str, Any] = self.config["model"]["generation"][
            "parameters"
        ]
        self.MODEL_DEVICE_CONFIG: Dict[str, Any] = self.config["model"]["generation"][
            "device"
        ]

        # Chunk settings from YAML
        self.CHUNK_SIZE: int = self.config["chunks"]["size"]
        self.CHUNK_OVERLAP: int = self.config["chunks"]["overlap"]

        # Source settings from YAML
        self.SOURCE_NUM: int = self.config["source"]["num_sources"]

        # Processing settings from YAML
        self.PROCESSING_CONFIG: Dict[str, Any] = self.config["processing"]

    @property
    def model_dtype(self) -> str:
        """Get model dtype setting"""
        return self.MODEL_DEVICE_CONFIG["dtype"]

    @property
    def model_device_map(self) -> str:
        """Get model device mapping setting"""
        return self.MODEL_DEVICE_CONFIG["map"]


# Create a global settings instance
settings = Settings()
