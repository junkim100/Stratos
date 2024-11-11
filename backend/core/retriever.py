from typing import List
import os
from dotenv import load_dotenv
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..models.schema import ProcessedChunk
from ..config.settings import settings  # Import settings instance, not class

class Retriever:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get API credentials from environment variables
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.google_cse_id = os.getenv('GOOGLE_CSE_ID')
        
        if not self.google_api_key or not self.google_cse_id:
            raise ValueError("GOOGLE_API_KEY and GOOGLE_CSE_ID must be set in .env file")

        self.search_wrapper = GoogleSearchAPIWrapper(
            google_api_key=self.google_api_key,
            google_cse_id=self.google_cse_id,
            k=5
        )
        
        # Use chunk settings from config.yml
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    def retrieve(self, query: str) -> List[ProcessedChunk]:
        search_results = self.search_wrapper.results(
            query=query,
            num_results=5
        )
        chunks = []
        
        for result in search_results:
            text_chunks = self.text_splitter.split_text(result['snippet'])
            chunks.extend([
                ProcessedChunk(text=chunk, source=result['link'])
                for chunk in text_chunks
            ])
        
        return chunks
