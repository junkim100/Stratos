from typing import List
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..models.schema import ProcessedChunk, SearchResponse
from ..config.settings import settings

class Generator:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize model and tokenizer with legacy=False
        self.logger.info(f"Loading model: {settings.GENERATION_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.GENERATION_MODEL,
            legacy=False,  # Use new tokenizer behavior
            padding_side="left"  # Better for causal LMs
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate settings
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.GENERATION_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            pad_token_id=self.tokenizer.pad_token_id
        )

    def _prepare_prompt(self, query: str, chunks: List[ProcessedChunk]) -> str:
        """Prepare the prompt for the model"""
        # Combine context from chunks
        context = "\n".join([
            f"Source {i+1}: {chunk.text}"
            for i, chunk in enumerate(chunks)
        ])
        
        prompt = f"""Based on the following sources, provide a comprehensive answer to the question.

Sources:
{context}

Question: {query}

Answer: """
        
        return prompt

    async def generate(self, query: str, chunks: List[ProcessedChunk]) -> SearchResponse:
        try:
            self.logger.info(f"Generating response for query: {query}")
            
            prompt = self._prepare_prompt(query, chunks)
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=settings.GENERATION_CONFIG["max_length"],
                padding=True
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **settings.GENERATION_CONFIG
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Extract answer part
            answer = generated_text.split("Answer: ")[-1].strip()
            sources = list(set(chunk.source for chunk in chunks))
            
            self.logger.info("Successfully generated response")
            
            return SearchResponse(
                answer=answer,
                sources=sources
            )
            
        except Exception as e:
            self.logger.error(f"Error in generation: {str(e)}")
            raise
