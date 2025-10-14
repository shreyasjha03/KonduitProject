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
    url: str
    snippet: str
    relevance_score: float

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)

class Timings(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceSnippet]
    timings: Timings
    grounded: bool = True
    refusal_reason: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    ollama_available: bool
    database_available: bool
    vector_index_available: bool
    stats: Dict[str, Any] = {}
