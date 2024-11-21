from ..base_agent import BaseAgent
from ...config.settings import settings
import torch
from typing import List
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class Summarizer(BaseAgent):
    def __init__(self):
        super().__init__(settings)

    def _initialize_model(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.params.get("model"),
                legacy=False,
                padding_side="left",
                trust_remote_code=True,
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForSeq2SeqLM.from_pretrained(
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

    def _create_prompt(self, text: str, summary_type: str = "brief") -> str:
        """
        Create different prompts based on the type of summary needed.
        """
        if summary_type == "final":
            # TODO: Filter dates or specifics based on user query. User might ask date sensitive questions.
            return f"""Today's date is {self.date}\nCreate a comprehensive and detailed summary of the following findings, maintaining key information and citing sources where relevant. Include specific details and examples:\n\n{text}\n\nGenerate a thorough summary that:\n1. Synthesizes all major points\n2. Maintains source attributions\n3. Provides specific details and examples\n4. Organizes information logically"""
        elif summary_type == "individual":
            return f"""Today's date is {self.date}\nExtract and summarize the key points from this text, maintaining essential details:\n\n{text}"""
        else:  # brief/default
            return f"""Today's date is {self.date}\nSummarize this text concisely while maintaining key information:\n\n{text}"""

    def _tokenize_input(self, text: str) -> dict:
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.params.get("max_length"),
            padding=True,
        ).to(self.model.device)

    def _generate_text(self, text: str, summary_type: str = "brief") -> str:
        try:
            inputs = self._tokenize_input(self._create_prompt(text, summary_type))

            # Adjust generation parameters based on summary type
            if summary_type == "final":
                max_length = self.params.get("final_max_length")
            elif summary_type == "individual":
                max_length = self.params.get("individual_max_length")
            else:
                raise ValueError("Invalid summary type")

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_length=max_length,
                    temperature=self.params.get("temperature", 0.7),
                    num_beams=self.params.get("num_beams", 4),
                    do_sample=self.params.get("do_sample", True),
                    top_p=self.params.get("top_p", 0.9),
                    top_k=self.params.get("top_k", 50),
                    repetition_penalty=self.params.get("repetition_penalty", 1.0),
                    early_stopping=True,
                )

            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if not summary or len(summary) < 50 // 4:
                raise ValueError("Generated summary is too short or empty")

            return summary.strip()

        except Exception as e:
            self.logger.error(f"Error in generate_summary: {str(e)}")
            raise

    async def summarize(self, text: str, summary_type: str = "brief") -> str:
        try:
            if not text or len(text.strip()) < 10:
                raise ValueError("Input text is too short or empty")

            summary = self._generate_text(text, summary_type)
            self.logger.info(f"Generated {summary_type} summary: {summary}")
            return summary

        except Exception as e:
            self.logger.error(f"Error in summarization: {str(e)}")
            return text.strip()

    async def __call__(self, text: str, summary_type: str = "brief") -> str:
        return await self.summarize(text, summary_type)
