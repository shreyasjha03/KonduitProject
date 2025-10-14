"""Tests for the QA engine and retrieval modules."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from app.qa_engine import QARAGEngine
from app.retriever import DocumentRetriever

class TestQARAGEngine:
    """Test cases for QARAGEngine."""
    
    def test_qa_engine_initialization(self):
        """Test QA engine initialization."""
        qa_engine = QARAGEngine(
            ollama_base_url="http://localhost:11434",
            chat_model="llama3.2"
        )
        assert qa_engine.ollama_base_url == "http://localhost:11434"
        assert qa_engine.chat_model == "llama3.2"
    
    def test_filter_high_quality_chunks(self):
        """Test chunk filtering."""
        qa_engine = QARAGEngine()
        
        # Test with high-quality chunks
        high_quality_chunks = [
            {"combined_score": 0.8, "chunk_text": "Relevant content"},
            {"combined_score": 0.6, "chunk_text": "Somewhat relevant content"},
            {"combined_score": 0.3, "chunk_text": "Less relevant content"}
        ]
        
        filtered = qa_engine._filter_high_quality_chunks(high_quality_chunks, min_combined_score=0.5)
        assert len(filtered) == 2
        
        # Test with no high-quality chunks
        low_quality_chunks = [
            {"combined_score": 0.1, "chunk_text": "Low quality content"},
            {"combined_score": 0.2, "chunk_text": "Another low quality content"}
        ]
        
        filtered = qa_engine._filter_high_quality_chunks(low_quality_chunks, min_combined_score=0.5)
        assert len(filtered) == 2  # Should return top 3 if none meet threshold
    
    def test_construct_grounded_prompt(self):
        """Test grounded prompt construction."""
        qa_engine = QARAGEngine()
        
        question = "What is the main topic?"
        chunks = [
            {
                "url": "https://example.com/page1",
                "chunk_text": "This page discusses artificial intelligence and machine learning concepts."
            },
            {
                "url": "https://example.com/page2", 
                "chunk_text": "Machine learning is a subset of AI that focuses on algorithms."
            }
        ]
        
        prompt = qa_engine._construct_grounded_prompt(question, chunks)
        
        # Check that prompt contains key elements
        assert question in prompt
        assert "https://example.com/page1" in prompt
        assert "artificial intelligence" in prompt
        assert "SAFETY INSTRUCTIONS" in prompt
        assert "Answer ONLY based on the information provided" in prompt
    
    def test_validate_grounding(self):
        """Test response grounding validation."""
        qa_engine = QARAGEngine()
        
        chunks = [{"url": "https://example.com", "chunk_text": "Some content"}]
        
        # Test grounded response with refusal
        grounded_response = "I don't have enough information in the crawled content to answer this question."
        grounded, reason = qa_engine._validate_grounding(grounded_response, chunks)
        assert grounded is True
        assert reason is None
        
        # Test potentially ungrounded response
        ungrounded_response = "I know that this is generally true because everyone knows it."
        grounded, reason = qa_engine._validate_grounding(ungrounded_response, chunks)
        assert grounded is False
        assert "hallucination indicator" in reason
        
        # Test short response
        short_response = "Yes."
        grounded, reason = qa_engine._validate_grounding(short_response, chunks)
        assert grounded is False
        assert "too short" in reason
    
    def test_extract_sources_from_response(self):
        """Test source extraction."""
        qa_engine = QARAGEngine()
        
        response = "Based on the content from example.com, AI is discussed."
        chunks = [
            {
                "url": "https://example.com/page1",
                "snippet": "AI content snippet",
                "combined_score": 0.8
            },
            {
                "url": "https://example.com/page2",
                "snippet": "ML content snippet", 
                "combined_score": 0.6
            }
        ]
        
        sources = qa_engine.extract_sources_from_response(response, chunks)
        
        assert len(sources) == 2
        assert sources[0]["url"] == "https://example.com/page1"
        assert sources[0]["relevance_score"] == 0.8
        assert sources[1]["url"] == "https://example.com/page2"
        assert sources[1]["relevance_score"] == 0.6
    
    @pytest.mark.asyncio
    async def test_check_ollama_availability(self):
        """Test Ollama availability check."""
        qa_engine = QARAGEngine()
        
        # Mock successful response
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [{"name": "llama3.2"}, {"name": "other-model"}]
            }
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            available = await qa_engine.check_ollama_availability()
            assert available is True
        
        # Mock failed response
        with patch('httpx.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get.side_effect = Exception("Connection failed")
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            available = await qa_engine.check_ollama_availability()
            assert available is False

class TestDocumentRetriever:
    """Test cases for DocumentRetriever."""
    
    def test_retriever_initialization(self):
        """Test retriever initialization."""
        retriever = DocumentRetriever(
            ollama_base_url="http://localhost:11434",
            embedding_model="llama3.2"
        )
        assert retriever.ollama_base_url == "http://localhost:11434"
        assert retriever.embedding_model == "llama3.2"
    
    def test_get_chunk_data(self):
        """Test chunk data retrieval."""
        retriever = DocumentRetriever()
        
        # Mock database manager
        with patch('app.retriever.db_manager') as mock_db:
            mock_chunk = {
                "chunk_id": 1,
                "chunk_text": "This is test content about AI",
                "url": "https://example.com",
                "title": "AI Article"
            }
            mock_db.get_text_chunks.return_value = [mock_chunk]
            
            search_result = {
                "chunk_id": 1,
                "page_title": "AI Article",
                "similarity_score": 0.8
            }
            
            chunk_data = retriever._get_chunk_data(search_result)
            
            assert chunk_data is not None
            assert chunk_data["chunk_text"] == "This is test content about AI"
            assert chunk_data["url"] == "https://example.com"
            assert chunk_data["similarity_score"] == 0.8
            assert "relevance_score" in chunk_data
    
    def test_rerank_results(self):
        """Test result reranking."""
        retriever = DocumentRetriever()
        
        question = "What is artificial intelligence?"
        chunks = [
            {
                "chunk_text": "Artificial intelligence is a field of computer science",
                "similarity_score": 0.5,
                "relevance_score": 0.8
            },
            {
                "chunk_text": "Machine learning is a subset of AI",
                "similarity_score": 0.3,
                "relevance_score": 0.9
            },
            {
                "chunk_text": "This page is about cooking recipes",
                "similarity_score": 0.7,
                "relevance_score": 0.1
            }
        ]
        
        reranked = retriever.rerank_results(question, chunks, max_results=2)
        
        assert len(reranked) == 2
        # Second chunk should rank higher due to better relevance score
        assert reranked[0]["chunk_text"] == "Machine learning is a subset of AI"
        assert "combined_score" in reranked[0]
    
    def test_filter_low_quality_chunks(self):
        """Test low quality chunk filtering."""
        retriever = DocumentRetriever()
        
        chunks = [
            {"combined_score": 0.8, "chunk_text": "High quality content"},
            {"combined_score": 0.3, "chunk_text": "Medium quality content"},
            {"combined_score": 0.05, "chunk_text": "Low quality content"}
        ]
        
        filtered = retriever.filter_low_quality_chunks(chunks, min_score=0.2)
        assert len(filtered) == 2
        assert all(chunk["combined_score"] >= 0.2 for chunk in filtered)
    
    def test_get_retrieval_stats(self):
        """Test retrieval statistics."""
        retriever = DocumentRetriever()
        
        chunks = [
            {
                "similarity_score": 0.8,
                "relevance_score": 0.6,
                "combined_score": 0.7,
                "url": "https://example.com/page1",
                "chunk_text": "Content 1"
            },
            {
                "similarity_score": 0.6,
                "relevance_score": 0.8,
                "combined_score": 0.7,
                "url": "https://example.com/page2",
                "chunk_text": "Content 2"
            }
        ]
        
        stats = retriever.get_retrieval_stats(chunks)
        
        assert stats["chunk_count"] == 2
        assert stats["avg_similarity_score"] == 0.7
        assert stats["avg_relevance_score"] == 0.7
        assert stats["avg_combined_score"] == 0.7
        assert stats["unique_urls"] == 2
        assert stats["total_text_length"] == len("Content 1") + len("Content 2")

if __name__ == "__main__":
    pytest.main([__file__])
