import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from ..config.settings import settings
import datetime


class BaseAgent:
    def __init__(self, settings):
        self._setup_logging()
        self.logger.debug(f"Initializing {self.__class__.__name__}")

        self.params = getattr(settings, f"{self.__class__.__name__}")
        self.model, self.tokenizer = self._initialize_model()

        self.date = datetime.datetime.now().strftime("%Y-%m-%d")

    def _setup_logging(self) -> None:
        logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
        self.logger = logging.getLogger(self.__class__.__name__)

    def _initialize_model(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        try:
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.params.get("model"),
                legacy=False,
                padding_side="left",
                trust_remote_code=True,
            )

            # Set pad token if needed
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Initialize model
            model = AutoModelForCausalLM.from_pretrained(
                self.params.get("model"),
                torch_dtype=torch.float16,
                device_map="auto",
                pad_token_id=tokenizer.pad_token_id,
                trust_remote_code=True,
                token=settings.HUGGINGFACE_API_KEY,
            )

            return model, tokenizer

        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    async def __call__(self, *args, **kwargs):
        raise NotImplementedError("All agents must implement __call__ method")
