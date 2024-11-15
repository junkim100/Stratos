from typing import List, Optional
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from ..config.settings import settings

class QueryDecomposer:
    """
    Responsible for decomposing complex queries into simpler sub-queries
    using a language model.
    """

    def __init__(self):
        self._setup_logging()
        self.model, self.tokenizer = self._initialize_model()
        self.params = settings.QUERY_DECOMPOSER_PARAMS

    def _setup_logging(self) -> None:
        """Initialize logging configuration."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _initialize_model(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Initialize the model and tokenizer."""
        try:
            # self.logger.info(f"Loading model: {settings.QUERY_DECOMPOSER_MODEL}")

            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                settings.QUERY_DECOMPOSER_MODEL,
                legacy=False,
                padding_side="left",
                trust_remote_code=True
            )

            # Set pad token if needed
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Initialize model
            model = AutoModelForCausalLM.from_pretrained(
                settings.QUERY_DECOMPOSER_MODEL,
                torch_dtype=torch.float16,
                device_map="auto",
                pad_token_id=tokenizer.pad_token_id,
                trust_remote_code=True,
                token=settings.HUGGINGFACE_API_KEY
            )

            return model, tokenizer

        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def _create_prompt(self, query: str) -> str:
        """Create the decomposition prompt."""
        return f"""Break down this complex query into simpler, atomic sub-queries.
        Each sub-query should focus on a specific aspect of the main query.
        Generate between {self.params['min_queries']} and {self.params['max_queries']} sub-queries, whatever you think is enough to break down the query effectively.

        Original query: {query}

        Sub-queries:"""

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
                temperature=self.params.get('temperature', 0.3),
                num_beams=self.params.get('num_beams', 4),
                do_sample=self.params.get('do_sample', True),
                top_p=self.params.get('top_p', 0.9),
                top_k=self.params.get('top_k', 50),
                repetition_penalty=self.params.get('repetition_penalty', 1.2)
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _parse_output(self, output: str) -> List[str]:
        """Parse the model output into sub-queries."""
        try:
            # Split into lines and clean up
            lines = output.strip().split('\n')

            # Process each line
            sub_queries = []
            for line in lines:
                # Clean up the line
                cleaned = line.strip().lstrip('1234567890.-) ')
                if cleaned and len(cleaned) > 10:  # Minimum length check
                    sub_queries.append(cleaned)

            # Limit number of sub-queries
            max_queries = self.params.get('max_queries', 3)
            return sub_queries[:max_queries]

        except Exception as e:
            self.logger.error(f"Error parsing output: {str(e)}")
            return []

    def _validate_sub_queries(self, sub_queries: List[str], original_query: str) -> List[str]:
        """Validate the generated sub-queries."""
        min_queries = self.params.get('min_queries', 1)

        if not sub_queries or len(sub_queries) < min_queries:
            self.logger.warning(
                f"Generated fewer than {min_queries} valid sub-queries. Using original query."
            )
            return [original_query]

        return sub_queries

    async def decompose(self, query: str) -> List[str]:
        """
        Decompose a complex query into multiple simpler sub-queries.
        """
        try:
            # self.logger.info(f"Decomposing query: {query}")

            # Create and tokenize prompt
            prompt = self._create_prompt(query)
            inputs = self._tokenize_input(prompt)

            # Generate text
            generated_text = self._generate_text(inputs)
            # just keep the response
            generated_text = generated_text.split("Sub-queries:")[1]

            # Parse and validate output
            sub_queries = self._parse_output(generated_text)
            validated_queries = self._validate_sub_queries(sub_queries, query)

            for i, sq in enumerate(validated_queries, 1):
                self.logger.info(f"\n////////// Sub-query {i}: {sq} //////////\n")

            for i, sq in enumerate(validated_queries, 1):
                self.logger.info(f"Sub-query {i}: {sq}")

            return validated_queries

        except Exception as e:
            self.logger.error(f"Error in query decomposition: {str(e)}")
            return [query]

    async def __call__(self, query: str) -> List[str]:
        """Make the class callable for easier pipeline integration."""
        return await self.decompose(query)