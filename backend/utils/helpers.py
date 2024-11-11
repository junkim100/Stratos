# backend/utils/helpers.py

import time
from typing import Any, Dict, List
from datetime import datetime


class SearchHelpers:
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks of specified size."""
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    @staticmethod
    def remove_duplicates(chunks: List[str]) -> List[str]:
        """Remove duplicate chunks while preserving order."""
        seen = set()
        return [x for x in chunks if not (x in seen or seen.add(x))]


class ValidationHelpers:
    @staticmethod
    def validate_query(query: str) -> bool:
        """Validate search query."""
        if not query or len(query.strip()) == 0:
            return False
        if len(query) > 1000:  # Maximum query length
            return False
        return True


class ResponseHelpers:
    @staticmethod
    def format_response(answer: str, sources: List[str]) -> Dict[str, Any]:
        """Format the final response."""
        return {
            "answer": answer,
            "sources": sources,
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time": time.time(),
        }


class LoggingHelpers:
    @staticmethod
    def log_query(query: str, processing_time: float) -> None:
        """Log search query and processing time."""
        timestamp = datetime.utcnow().isoformat()
        print(f"[{timestamp}] Query: {query} (Processing time: {processing_time}s)")
