import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class Settings:
    def __init__(self):
        # Load YAML config
        self.config = self._load_yaml_config()

        # Initialize settings
        self._initialize_settings()

    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = Path(__file__).parent / "config.yml"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _initialize_settings(self) -> None:
        """Initialize all settings from YAML and environment variables"""
        # API Settings
        self.API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT: int = int(os.getenv("API_PORT", "51441"))
        self.CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "http://localhost:51440").split(",")

        # Google API Settings
        self.GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
        self.GOOGLE_CSE_ID: str = os.getenv("GOOGLE_CSE_ID")

        # Hugging Face API Settings
        self.HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY")

        # OpenAI API Settings
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

        # Query Decomposer settings
        self.QUERY_DECOMPOSER_MODEL: str = self.config["agents"]["query_decomposer"]["model"]
        self.QUERY_DECOMPOSER_PARAMS: Dict[str, Any] = self.config["agents"]["query_decomposer"]["parameters"]

        # Chunk Processor settings
        self.CHUNK_PROCESSOR_MODEL: str = self.config["agents"]["chunk_processor"]["model"]
        self.CHUNK_PROCESSOR_PARAMS: Dict[str, Any] = self.config["agents"]["chunk_processor"]["parameters"]

        # Result Reranker settings
        self.RERANKER_MODEL: str = self.config["agents"]["result_reranker"]["model"]
        self.RERANKER_PARAMS: Dict[str, Any] = self.config["agents"]["result_reranker"]["parameters"]

        # Response Generator settings
        self.RESPONSE_GENERATOR_MODEL: str = self.config["agents"]["response_generator"]["model"]
        self.RESPONSE_GENERATOR_PARAMS: Dict[str, Any] = self.config["agents"]["response_generator"]["parameters"]
        self.RESPONSE_GENERATOR_DEVICE: Dict[str, str] = self.config["agents"]["response_generator"]["device"]

        # Source settings
        self.SOURCE_NUM: int = self.config["source"]["num_sources"]

        # Processing settings
        self.MAX_CHUNKS: int = self.config["processing"]["max_chunks"]
        self.MIN_CHUNK_LENGTH: int = self.config["processing"]["min_chunk_length"]
        self.MAX_SUMMARY_LENGTH: int = self.config["processing"]["max_summary_length"]

    @property
    def model_dtype(self) -> str:
        """Get model dtype setting"""
        return self.RESPONSE_GENERATOR_DEVICE["dtype"]

    @property
    def model_device_map(self) -> str:
        """Get model device mapping setting"""
        return self.RESPONSE_GENERATOR_DEVICE["map"]

    @property
    def generation_params(self) -> Dict[str, Any]:
        """Get all generation parameters"""
        return self.RESPONSE_GENERATOR_PARAMS

    @property
    def api_url(self) -> str:
        """Get the complete API URL"""
        return f"http://{self.API_HOST}:{self.API_PORT}"

    def __str__(self) -> str:
        """String representation of current settings"""
        return f"Settings(API: {self.api_url}, Models: {self.RESPONSE_GENERATOR_MODEL})"


# Create a global settings instance that will be imported by the agents and other modules
settings = Settings()
