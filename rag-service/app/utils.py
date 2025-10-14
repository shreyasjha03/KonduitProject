"""Utility functions for timing, normalization, and helpers."""

import time
import re
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse
from functools import wraps
import logging

logger = logging.getLogger("rag_service")

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        return result, execution_time
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        return result, execution_time
    
    # Return appropriate wrapper based on whether function is async
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

# Simple timing decorator that doesn't modify return values
def simple_timing_decorator(func):
    """Simple timing decorator that logs timing without modifying return values."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
        logger.info(f"{func.__name__} took {execution_time:.2f}ms")
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
        logger.info(f"{func.__name__} took {execution_time:.2f}ms")
        return result
    
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

def normalize_url(url: str, base_url: str = None) -> str:
    """Normalize URL by removing fragments and query parameters for comparison."""
    parsed = urlparse(url)
    if base_url:
        parsed = urlparse(urljoin(base_url, url))
    
    # Remove fragment and normalize
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    
    return normalized

def extract_domain(url: str) -> str:
    """Extract registrable domain from URL."""
    parsed = urlparse(url)
    domain_parts = parsed.netloc.split('.')
    
    # Handle cases like subdomain.example.com -> example.com
    if len(domain_parts) >= 2:
        return '.'.join(domain_parts[-2:])
    return parsed.netloc

def is_same_domain(url1: str, url2: str) -> bool:
    """Check if two URLs belong to the same registrable domain."""
    return extract_domain(url1) == extract_domain(url2)

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Remove non-printable characters except newlines
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
    
    return text

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[tuple]:
    """Split text into overlapping chunks.
    
    Returns:
        List of (chunk_text, start_pos, end_pos) tuples
    """
    if not text or len(text) <= chunk_size:
        return [(text, 0, len(text))]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            search_start = max(start + chunk_size - 100, start)
            sentence_end = text.rfind('.', search_start, end)
            if sentence_end > start:
                end = sentence_end + 1
        
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append((chunk_text, start, end))
        
        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def truncate_snippet(text: str, max_length: int = 200) -> str:
    """Truncate text to max_length with ellipsis."""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - 3] + "..."

def calculate_relevance_score(query: str, text: str) -> float:
    """Simple relevance scoring based on word overlap."""
    if not query or not text:
        return 0.0
    
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    
    if not query_words:
        return 0.0
    
    overlap = len(query_words.intersection(text_words))
    return overlap / len(query_words)

def safe_json_serialize(obj: Any) -> str:
    """Safely serialize object to JSON string."""
    import json
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return str(obj)

def format_timing_report(timings: Dict[str, float]) -> str:
    """Format timing report for logging."""
    return " | ".join([f"{key}: {value:.2f}ms" for key, value in timings.items()])
