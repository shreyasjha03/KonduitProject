"""Pydantic models for API requests and responses."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, HttpUrl, Field
from datetime import datetime

class CrawlRequest(BaseModel):
    start_url: HttpUrl
    max_pages: int = Field(default=50, ge=1, le=100)
    max_depth: int = Field(default=3, ge=1, le=5)
    crawl_delay_ms: int = Field(default=1000, ge=100, le=5000)

class CrawlResponse(BaseModel):
    page_count: int
    skipped_count: int
    urls: List[str]
    crawl_time_seconds: float
    errors: List[str] = []

class IndexRequest(BaseModel):
    chunk_size: int = Field(default=800, ge=100, le=2000)
    chunk_overlap: int = Field(default=100, ge=0, le=500)
    embedding_model: str = "llama3.2"

class IndexResponse(BaseModel):
    vector_count: int
    chunk_count: int
    indexing_time_seconds: float
    errors: List[str] = []

class SourceSnippet(BaseModel):
    url: str = Field(description="Source URL of the content")
    snippet: str = Field(description="Relevant text snippet from the source", max_length=500)
    relevance_score: float = Field(description="Relevance score (0.0 to 1.0)", ge=0.0, le=1.0)

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)

class Timings(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float

class AskResponse(BaseModel):
    answer: str = Field(description="Generated answer based on retrieved content")
    sources: List[SourceSnippet] = Field(description="Source URLs and snippets used for the answer")
    timings: Timings = Field(description="Performance timing metrics")
    grounded: bool = Field(description="Whether the answer is grounded in retrieved content", default=True)
    refusal_reason: Optional[str] = Field(description="Reason for refusal if answer cannot be provided", default=None)

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    ollama_available: bool
    database_available: bool
    vector_index_available: bool
    stats: Dict[str, Any] = {}
