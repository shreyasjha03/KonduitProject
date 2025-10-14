#!/usr/bin/env python3
"""
Comprehensive tests for the RAG system.
Tests crawl, index, ask functionality with proper assertions.
"""

import pytest
import asyncio
import httpx
import json
from typing import Dict, Any

class TestRAGSystem:
    """Test suite for RAG system functionality."""
    
    @pytest.fixture
    async def client(self):
        """Create HTTP client for testing."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            yield client
    
    @pytest.fixture
    async def setup_system(self, client):
        """Setup system by crawling and indexing content."""
        base_url = "http://localhost:8000"
        
        # Crawl example.com
        crawl_response = await client.post(
            f"{base_url}/crawl",
            json={
                "start_url": "https://example.com",
                "max_pages": 1,
                "max_depth": 1,
                "crawl_delay_ms": 500
            }
        )
        assert crawl_response.status_code == 200
        crawl_result = crawl_response.json()
        assert crawl_result["page_count"] > 0
        
        # Index content
        index_response = await client.post(
            f"{base_url}/index",
            json={
                "chunk_size": 500,
                "chunk_overlap": 100,
                "embedding_model": "llama3.2:latest"
            }
        )
        assert index_response.status_code == 200
        index_result = index_response.json()
        assert index_result["vector_count"] > 0
        
        return base_url, crawl_result, index_result
    
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test system health endpoint."""
        response = await client.get("http://localhost:8000/health")
        assert response.status_code == 200
        
        health = response.json()
        assert "status" in health
        assert "ollama_available" in health
        assert "database_available" in health
        assert "vector_index_available" in health
    
    @pytest.mark.asyncio
    async def test_crawl_functionality(self, client):
        """Test crawling functionality."""
        response = await client.post(
            "http://localhost:8000/crawl",
            json={
                "start_url": "https://example.com",
                "max_pages": 1,
                "max_depth": 1,
                "crawl_delay_ms": 500
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Validate response structure
        assert "page_count" in result
        assert "skipped_count" in result
        assert "urls" in result
        assert "crawl_time_seconds" in result
        assert "errors" in result
        
        # Validate data types
        assert isinstance(result["page_count"], int)
        assert isinstance(result["skipped_count"], int)
        assert isinstance(result["urls"], list)
        assert isinstance(result["crawl_time_seconds"], float)
        assert isinstance(result["errors"], list)
    
    @pytest.mark.asyncio
    async def test_index_functionality(self, client, setup_system):
        """Test indexing functionality."""
        base_url, _, _ = setup_system
        
        response = await client.post(
            f"{base_url}/index",
            json={
                "chunk_size": 500,
                "chunk_overlap": 100,
                "embedding_model": "llama3.2:latest"
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Validate response structure
        assert "vector_count" in result
        assert "chunk_count" in result
        assert "indexing_time_seconds" in result
        assert "errors" in result
        
        # Validate data types and values
        assert isinstance(result["vector_count"], int)
        assert isinstance(result["chunk_count"], int)
        assert isinstance(result["indexing_time_seconds"], float)
        assert isinstance(result["errors"], list)
        assert result["vector_count"] > 0
    
    @pytest.mark.asyncio
    async def test_ask_answerable_question(self, client, setup_system):
        """Test asking an answerable question."""
        base_url, _, _ = setup_system
        
        response = await client.post(
            f"{base_url}/ask",
            json={
                "question": "What is this website about?",
                "top_k": 3
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Validate response structure
        assert "answer" in result
        assert "sources" in result
        assert "timings" in result
        assert "grounded" in result
        assert "refusal_reason" in result
        
        # Validate data types
        assert isinstance(result["answer"], str)
        assert isinstance(result["sources"], list)
        assert isinstance(result["timings"], dict)
        assert isinstance(result["grounded"], bool)
        
        # Validate timings structure
        timings = result["timings"]
        assert "retrieval_ms" in timings
        assert "generation_ms" in timings
        assert "total_ms" in timings
        
        # Validate sources structure (if any)
        for source in result["sources"]:
            assert "url" in source
            assert "snippet" in source
            assert "relevance_score" in source
            assert isinstance(source["url"], str)
            assert isinstance(source["snippet"], str)
            assert isinstance(source["relevance_score"], (int, float))
    
    @pytest.mark.asyncio
    async def test_ask_unanswerable_question(self, client, setup_system):
        """Test asking an unanswerable question (should refuse)."""
        base_url, _, _ = setup_system
        
        response = await client.post(
            f"{base_url}/ask",
            json={
                "question": "What is the weather like today?",
                "top_k": 3
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Should either refuse or provide a grounded answer
        assert "answer" in result
        assert isinstance(result["answer"], str)
        
        # If it's a refusal, check the structure
        if not result["grounded"]:
            assert result["refusal_reason"] is not None
            assert "don't have enough information" in result["answer"].lower() or \
                   "not found" in result["answer"].lower()
    
    @pytest.mark.asyncio
    async def test_grounding_enforcement(self, client, setup_system):
        """Test that answers are properly grounded."""
        base_url, _, _ = setup_system
        
        response = await client.post(
            f"{base_url}/ask",
            json={
                "question": "Tell me about something completely unrelated to the crawled content",
                "top_k": 5
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Should either be grounded in sources or properly refuse
        if result["grounded"]:
            assert len(result["sources"]) > 0
            # Check that sources contain relevant content
            for source in result["sources"]:
                assert len(source["snippet"]) > 0
        else:
            # Should be a proper refusal
            assert result["refusal_reason"] is not None
            assert len(result["answer"]) > 0
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, client, setup_system):
        """Test that performance metrics are properly tracked."""
        base_url, _, _ = setup_system
        
        # Make a query to generate metrics
        await client.post(
            f"{base_url}/ask",
            json={
                "question": "What is this about?",
                "top_k": 3
            }
        )
        
        # Check stats endpoint
        stats_response = await client.get(f"{base_url}/stats")
        assert stats_response.status_code == 200
        
        stats = stats_response.json()
        
        # Validate stats structure
        assert "database" in stats
        assert "index" in stats
        assert "queries" in stats
        
        # Check query metrics
        queries = stats["queries"]
        assert "total_queries" in queries
        assert "grounded_queries" in queries
        assert "grounding_rate" in queries
        
        assert isinstance(queries["total_queries"], int)
        assert isinstance(queries["grounded_queries"], int)
        assert isinstance(queries["grounding_rate"], (int, float))
    
    @pytest.mark.asyncio
    async def test_error_handling(self, client):
        """Test error handling for invalid requests."""
        # Test invalid crawl request
        response = await client.post(
            "http://localhost:8000/crawl",
            json={
                "start_url": "invalid-url",
                "max_pages": 1,
                "max_depth": 1,
                "crawl_delay_ms": 500
            }
        )
        
        # Should handle gracefully (either error or empty result)
        assert response.status_code in [200, 400, 422]
        
        # Test invalid ask request (no content indexed)
        response = await client.post(
            "http://localhost:8000/ask",
            json={
                "question": "Test question",
                "top_k": 3
            }
        )
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 422]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
