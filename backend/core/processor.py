from typing import List
import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer
from ..models.schema import ProcessedChunk

class Processor:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize T5 model for summarization
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')

        # Configuration
        self.max_chunk_length = 512
        self.min_chunk_length = 50
        self.max_summary_length = 150
        self.min_summary_length = 40

    def summarize(self, text: str) -> str:
        """
        Summarize text using T5 model
        """
        # self.logger.info(f'Summarizing text of length: {len(text)}')

        # Prepare input text
        input_text = f"summarize: {text}"

        # Tokenize
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=self.max_chunk_length,
            truncation=True
        )

        # Generate summary
        summary_ids = self.model.generate(
            inputs,
            max_length=self.max_summary_length,
            min_length=self.min_summary_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        # self.logger.info(f'Generated summary of length: {len(summary)}')

        return summary

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove very short texts
        if len(text) < self.min_chunk_length:
            return ""
        return text

    def remove_duplicates(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        """
        Remove duplicate chunks while preserving order
        """
        seen = set()
        unique_chunks = []

        for chunk in chunks:
            if chunk.text not in seen:
                seen.add(chunk.text)
                unique_chunks.append(chunk)

        return unique_chunks

    async def process(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        """
        Main processing pipeline
        """
        try:
            # self.logger.info(f'Starting to process {len(chunks)} chunks')

            # Step 1: Clean texts
            cleaned_chunks = [
                ProcessedChunk(
                    text=self.clean_text(chunk.text),
                    source=chunk.source
                )
                for chunk in chunks
            ]

            # Remove empty chunks after cleaning
            cleaned_chunks = [
                chunk for chunk in cleaned_chunks
                if chunk.text
            ]

            # Step 2: Remove duplicates
            unique_chunks = self.remove_duplicates(cleaned_chunks)
            # self.logger.info(f'Removed duplicates, {len(unique_chunks)} chunks remaining')

            # Step 3: Summarize chunks
            processed_chunks = [
                ProcessedChunk(
                    text=self.summarize(chunk.text),
                    source=chunk.source
                )
                for chunk in unique_chunks
            ]

            # self.logger.info(f'Finished processing, generated {len(processed_chunks)} summaries')

            return processed_chunks

        except Exception as e:
            self.logger.error(f'////////////////////////////////////////Error in processing: {str(e)}////////////////////////////////////////')
            raise

    def __call__(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        """
        Make the class callable
        """
        return self.process(chunks)
