from typing import List, Optional
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from ..models.schema import ProcessedChunk, SearchResponse
from ..config.settings import settings

class ResponseGenerator:
    """
    Generates responses using a language model based on processed chunks of text.
    """

    def __init__(self):
        self._setup_logging()
        self.model, self.tokenizer = self._initialize_model()
        self.params = settings.RESPONSE_GENERATOR_PARAMS

    def _setup_logging(self) -> None:
        """Initialize logging configuration."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _initialize_model(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Initialize the model and tokenizer."""
        try:
            # self.logger.info(f"Loading model: {settings.RESPONSE_GENERATOR_MODEL}")

            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                settings.RESPONSE_GENERATOR_MODEL,
                legacy=False,
                padding_side="left",
                trust_remote_code=True
            )

            # Set pad token if needed
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Initialize model
            model = AutoModelForCausalLM.from_pretrained(
                settings.RESPONSE_GENERATOR_MODEL,
                torch_dtype=torch.float16,
                device_map="auto",
                pad_token_id=tokenizer.pad_token_id,
                trust_remote_code=True,
                token=settings.HUGGINGFACE_API_KEY
            )

            return model, tokenizer

        except Exception as e:
            self.logger.error(f"\n////////// Error initializing model: {str(e)} //////////\n")
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def _prepare_prompt(self, query: str, chunks: List[ProcessedChunk]) -> str:
        """
        Prepare the prompt for the model using the query and context chunks.
        """
        # Sort chunks by score if available
        sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)

        # Combine context from top chunks
        context_parts = []
        for i, chunk in enumerate(sorted_chunks[:self.params.get('max_chunks', 5)]):
            context_parts.append(f"Source {i+1}: {chunk.text}")

        context = "\n".join(context_parts)

        return f"""Based on the following sources, provide a comprehensive answer to the question.

Sources:
{context}

Question: {query}

Answer: """

    def _tokenize_input(self, prompt: str) -> torch.Tensor:
        """Tokenize the input prompt."""
        return self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.params.get('max_length', 512),
            padding=True
        ).to(self.model.device)

    def _generate_text(self, inputs: torch.Tensor) -> str:
        """Generate text using the model."""
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                max_length=self.params.get('max_length', 512),
                min_length=self.params.get('min_length', 50),
                num_beams=self.params.get('num_beams', 4),
                temperature=self.params.get('temperature', 0.7),
                do_sample=self.params.get('do_sample', True),
                top_p=self.params.get('top_p', 0.9),
                top_k=self.params.get('top_k', 50),
                repetition_penalty=self.params.get('repetition_penalty', 1.2)
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _extract_answer(self, generated_text: str) -> str:
        """Extract the answer portion from the generated text."""
        answer_parts = generated_text.split("Answer: ")
        if len(answer_parts) > 1:
            return answer_parts[-1].strip()
        return generated_text.strip()

    def _get_unique_sources(self, chunks: List[ProcessedChunk]) -> List[str]:
        """Get unique sources from chunks."""
        return list(set(chunk.source for chunk in chunks if chunk.source))

    async def generate(self, query: str, chunks: List[ProcessedChunk]) -> SearchResponse:
        """
        Generate a response based on the query and context chunks.
        """
        try:
            self.logger.info(f"Generating response for: {query}")

            if not chunks:
                raise ValueError("No context chunks provided")

            # Prepare and tokenize input
            prompt = self._prepare_prompt(query, chunks)
            inputs = self._tokenize_input(prompt)

            # Generate and process response
            generated_text = self._generate_text(inputs)
            answer = self._extract_answer(generated_text)
            sources = self._get_unique_sources(chunks)

            # self.logger.info("Successfully generated response")

            return SearchResponse(
                answer=answer,
                sources=sources
            )

        except Exception as e:
            self.logger.error(f"\n////////// Error in generation: {str(e)} //////////\n")
            # Return a graceful failure response
            return SearchResponse(
                answer="I apologize, but I encountered an error while generating the response.",
                sources=[]
            )

    async def __call__(self, query: str, chunks: List[ProcessedChunk]) -> SearchResponse:
        """Make the class callable for easier pipeline integration."""
        return await self.generate(query, chunks)