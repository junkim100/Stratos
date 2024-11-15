from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from langchain_google_community import GoogleSearchAPIWrapper
from ..models.schema import ProcessedChunk
from ..config.settings import settings
from ..core.chunker import Chunker

class Retriever:
    """
    Responsible for retrieving and processing search results
    from various sources.
    """

    def __init__(self):
        self._load_credentials()
        self.chunker = Chunker()
        self.search_wrapper = self._initialize_search_wrapper()

    def _load_credentials(self) -> None:
        """Load and validate API credentials."""
        load_dotenv()
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")

        if not self.google_api_key or not self.google_cse_id:
            raise ValueError("GOOGLE_API_KEY and GOOGLE_CSE_ID must be set in .env file")

    def _initialize_search_wrapper(self) -> GoogleSearchAPIWrapper:
        """Initialize the Google Search API wrapper."""
        return GoogleSearchAPIWrapper(
            google_api_key=self.google_api_key,
            google_cse_id=self.google_cse_id,
            k=settings.SOURCE_NUM
        )

    def _process_search_result(self, result: Dict[str, Any]) -> List[ProcessedChunk]:
        """Process a single search result into chunks."""
        metadata = {
            'source': result['link'],
            'title': result.get('title', ''),
            'score': result.get('score', 0.0),
            'position': result.get('position', 0)
        }

        return self.chunker.process(result['snippet'], metadata)

    async def retrieve(self, query: str) -> List[ProcessedChunk]:
        """
        Retrieve and process search results for a query.

        Args:
            query (str): Search query

        Returns:
            List[ProcessedChunk]: List of processed chunks from search results
        """
        try:
            # Get search results
            search_results = self.search_wrapper.results(
                query=query,
                num_results=settings.SOURCE_NUM
            )

            # Process all results
            all_chunks = []
            for result in search_results:
                chunks = self._process_search_result(result)
                all_chunks.extend(chunks)

            return all_chunks[:settings.CHUNK_PROCESSOR_PARAMS['max_chunks']]

        except Exception as e:
            print(f"Error in retrieval: {str(e)}")
            return []

    async def __call__(self, query: str) -> List[ProcessedChunk]:
        """Make the class callable for easier pipeline integration."""
        return await self.retrieve(query)