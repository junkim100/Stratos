from typing import List
from pydantic import BaseModel, Field


class ProcessedChunk(BaseModel):
    text: str
    source: str


class SearchResponse(BaseModel):
    answer: str = Field(..., min_length=1)
    sources: List[str] = Field(default_factory=list)
