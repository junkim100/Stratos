from typing import List, Dict, Any
from pydantic import BaseModel, Field

class ProcessedChunk(BaseModel):
    text: str
    source: str
    score: float = 0.0  # Ensure score is included and has a default value
    metadata: Dict[str, Any] = {}

class SearchResponse(BaseModel):
    answer: str = Field(..., min_length=1)
    sources: List[str] = Field(default_factory=list)