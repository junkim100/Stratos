from typing import List
from pydantic import BaseModel, Field


class ProcessedChunk(BaseModel):
    text: str
    source: str


class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)

class SearchResponse(BaseModel):
    answer: str = Field(..., min_length=1)
    sources: List[str] = Field(default_factory=list)