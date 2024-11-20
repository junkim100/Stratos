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
        ########## .env Settings ##########
        # Google API Settings
        self.GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
        self.GOOGLE_CSE_ID: str = os.getenv("GOOGLE_CSE_ID")

        # Hugging Face API Settings
        self.HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY")

        # OpenAI API Settings
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

        ########## YML Settings ##########
        # Agents settings
        self.ActionDecider: Dict[str, Any] = self.config["agents"]["ActionDecider"]
        self.ActionGenerator: Dict[str, Any] = self.config["agents"]["ActionGenerator"]
        self.ActionExecutor: Dict[str, Any] = self.config["agents"]["ActionExecutor"]
        self.Searcher: Dict[str, Any] = self.config["agents"]["Searcher"]
        self.Summarizer: Dict[str, Any] = self.config["agents"]["Summarizer"]
        self.Responder: Dict[str, Any] = self.config["agents"]["Responder"]

        # Logging settings
        self.LOG_LEVEL: str = self.config["logging_level"]

        # Number of sources to search per query
        self.NUM_SOURCES: int = self.config["num_sources"]


# Create a global settings instance that will be imported by the agents and other modules
settings = Settings()
