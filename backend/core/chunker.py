from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..models.schema import ProcessedChunk
from ..config.settings import settings

class Chunker:
    """
    Responsible for splitting text into chunks using various strategies
    and preprocessing techniques.
    """

    def __init__(self):
        self.params = settings.CHUNK_PROCESSOR_PARAMS
        self.splitter = self._initialize_splitter()

    def _initialize_splitter(self) -> RecursiveCharacterTextSplitter:
        """Initialize the text splitter with configured parameters."""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.params['size'],
            chunk_overlap=self.params['overlap'],
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before chunking."""
        # Remove excessive whitespace
        text = " ".join(text.split())
        # Add periods to make sure sentences end properly
        if text and text[-1] not in ".!?":
            text += "."
        return text

    def _filter_chunks(self, chunks: List[str]) -> List[str]:
        """Filter chunks based on minimum length and other criteria."""
        return [
            chunk for chunk in chunks
            if len(chunk) >= self.params['min_length']
        ][:self.params['max_chunks']]

    def process(self, text: str, metadata: Dict[str, Any]) -> List[ProcessedChunk]:
        """
        Process text into chunks with metadata.

        Args:
            text (str): Text to be chunked
            metadata (Dict[str, Any]): Metadata to be attached to each chunk

        Returns:
            List[ProcessedChunk]: List of processed chunks with metadata
        """
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)

            # Split text into chunks
            raw_chunks = self.splitter.split_text(processed_text)

            # Filter chunks
            filtered_chunks = self._filter_chunks(raw_chunks)

            # Create ProcessedChunk objects
            return [
                ProcessedChunk(
                    text=chunk,
                    source=metadata.get('source', ''),
                    score=metadata.get('score', 0.0),
                    metadata=metadata
                )
                for chunk in filtered_chunks
            ]

        except Exception as e:
            print(f"Error in chunk processing: {str(e)}")
            # Return single chunk if processing fails
            return [ProcessedChunk(
                text=text[:self.params['size']],
                source=metadata.get('source', ''),
                metadata=metadata
            )]