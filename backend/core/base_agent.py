import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from ..config.settings import settings
from typing import List, Dict
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
                trust_remote_code=True,
                token=settings.HUGGINGFACE_API_KEY,
            )

            return model, tokenizer

        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def _create_prompt(self) -> str:
        raise NotImplementedError("All agents must implement _create_prompt method")

    def _tokenize_input(
        self, chat_template: List[Dict[str, str]], max_length: int
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize the input based on the provided chat template using apply_chat_template.
        """
        # Apply the chat template
        inputs = self.tokenizer.apply_chat_template(
            conversation=chat_template,
            add_generation_prompt=True,  # Adds a generation prompt for models that need it
            return_dict=True,
            return_tensors="pt",
            max_length=max_length,
        )

        # Move tensors to the correct device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        return inputs

    def _generate_response(self) -> str:
        raise NotImplementedError("All agents must implement _generate_response method")

    async def __call__(self, *args, **kwargs):
        raise NotImplementedError("All agents must implement __call__ method")
