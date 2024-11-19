from typing import List
import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer
from ..models.schema import ProcessedChunk
from FlagEmbedding import FlagReranker

class Processor:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize T5 model for summarization
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')

        # Initialize reranker model for reranking
        self.reranker = self._initialize_reranker()

        # Configuration
        self.max_chunk_length = 512  # Max length for chunk input
        self.min_chunk_length = 50   # Minimum length for valid chunks
        self.max_summary_length = 150  # Max length for summary output
        self.min_summary_length = 40   # Min length for summary output
    
    def _initialize_reranker(self) -> FlagReranker:
        """Initialize the reranker model and tokenizer."""
        try:
            reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
            return reranker

        except Exception as e:
            self.logger.error(f"\n////////// Error initializing model: {str(e)} //////////\n")
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def summarize(self, text: str) -> str:
        """
        Summarize text using the T5 model.
        """
        # TODO: summarize via query or intent of query
        try:
            # Prepare input text for summarization
            input_text = f"summarize: {text}"

            # Tokenize input text for the model
            inputs = self.tokenizer.encode(
                input_text,
                return_tensors="pt",
                max_length=self.max_chunk_length,
                truncation=True
            )

            # Generate summary using the model
            summary_ids = self.model.generate(
                inputs,
                max_length=self.max_summary_length,
                min_length=self.min_summary_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )

            # Decode generated summary into text
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            return summary

        except Exception as e:
            self.logger.error(f"Error during summarization: {str(e)}")
            return text  # Return original text if summarization fails

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text by removing extra whitespace and short texts.
        """
        cleaned_text = ' '.join(text.split())  # Remove extra whitespace

        if len(cleaned_text) < self.min_chunk_length:
            return ""  # Return empty string if text is too short

        return cleaned_text

    def remove_duplicates(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        """
        Remove duplicate chunks while preserving order.
        """
        seen = set()
        unique_chunks = []

        for chunk in chunks:
            if chunk.text not in seen:
                seen.add(chunk.text)
                unique_chunks.append(chunk)

        return unique_chunks

    def rerank(self, query: str, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        """
        Rerank chunks based on their relevance to the query.
        """
        q_p_pairs = [[query, chunk.text] for chunk in chunks]
        scored_chunks = self.reranker.compute_score(q_p_pairs)
        
        scored_chunks_with_data = [
            (score, chunk) for score, chunk in zip(scored_chunks, chunks)
        ]
        scored_chunks_with_data.sort(key=lambda x: x[0], reverse=True)
        
        # TODO: set topk in config.yml
        top_chunks = scored_chunks_with_data[:5]
        
        reranked_chunks = [
            ProcessedChunk(text=chunk.text, source=chunk.source, score=chunk.score, metadata=chunk.metadata) for _, chunk in top_chunks
        ]

        return reranked_chunks

    async def process(self, query: str, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        """
        Main processing pipeline to clean, deduplicate, and summarize chunks.

        Args:
            chunks (List[ProcessedChunk]): List of processed chunks

        Returns:
            List[ProcessedChunk]: List of processed and summarized chunks
        """
        try:
            # Log incoming data type and content for debugging
            self.logger.info(f"Received {len(chunks)} chunks for processing")

            if not all(isinstance(chunk, ProcessedChunk) for chunk in chunks):
                raise ValueError("All elements in 'chunks' must be instances of ProcessedChunk")

            # Step 1: Clean texts in each chunk
            cleaned_chunks = [
                ProcessedChunk(
                    text=self.clean_text(chunk.text),
                    source=chunk.source,
                    score=chunk.score,
                    metadata=chunk.metadata
                )
                for chunk in chunks
            ]

            # Remove empty chunks after cleaning
            cleaned_chunks = [chunk for chunk in cleaned_chunks if chunk.text]

            if not cleaned_chunks:
                raise ValueError("No valid chunks after cleaning")

            # Step 2: Remove duplicate chunks based on their text content
            unique_chunks = self.remove_duplicates(cleaned_chunks)

            if not unique_chunks:
                raise ValueError("No valid unique chunks after deduplication")
            
            # Step 3: Rerank chunks based on their relevance to the query
            reranked_chunks = self.rerank(query, unique_chunks)

            # Step 4: Summarize each chunk's text content
            summarized_chunks = [
                ProcessedChunk(
                    text=self.summarize(chunk.text),
                    source=chunk.source,
                    score=chunk.score,
                    metadata=chunk.metadata
                )
                for chunk in reranked_chunks
            ]

            return summarized_chunks

        except Exception as e:
            self.logger.error(f"Error in processing: {str(e)}")
            raise e

    async def __call__(self, query: str, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        """
        Make the class callable so it can be used directly in pipelines.

        Args:
            chunks (List[ProcessedChunk]): List of processed chunks

        Returns:
            List[ProcessedChunk]: Processed and summarized chunks
        """
        return await self.process(query, chunks)