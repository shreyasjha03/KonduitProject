"""Tests for the indexer module."""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock

from app.indexer import VectorIndexer

class TestVectorIndexer:
    """Test cases for VectorIndexer."""
    
    def test_indexer_initialization(self):
        """Test indexer initialization."""
        indexer = VectorIndexer(
            ollama_base_url="http://localhost:11434",
            embedding_model="llama3.2"
        )
        assert indexer.ollama_base_url == "http://localhost:11434"
        assert indexer.embedding_model == "llama3.2"
        assert indexer.dimension == 4096
        assert indexer.index is None
    
    def test_chunk_page_text(self):
        """Test text chunking functionality."""
        indexer = VectorIndexer()
        
        # Test empty text
        chunks, positions = indexer._chunk_page_text("", 800, 100)
        assert len(chunks) == 0
        assert len(positions) == 0
        
        # Test short text
        short_text = "This is a short text that doesn't need chunking."
        chunks, positions = indexer._chunk_page_text(short_text, 800, 100)
        assert len(chunks) == 1
        assert chunks[0] == short_text
        assert positions[0][0] == short_text
        assert positions[0][1] == 0
        assert positions[0][2] == len(short_text)
        
        # Test long text
        long_text = "This is a long text. " * 100  # ~2000 characters
        chunks, positions = indexer._chunk_page_text(long_text, 200, 50)
        assert len(chunks) > 1
        assert len(positions) == len(chunks)
        
        # Verify positions are correct
        for i, (chunk, start, end) in enumerate(positions):
            assert chunk == long_text[start:end]
            if i > 0:
                # Check overlap
                prev_end = positions[i-1][2]
                assert start < prev_end
    
    @pytest.mark.asyncio
    async def test_generate_embeddings(self):
        """Test embedding generation."""
        indexer = VectorIndexer()
        
        # Mock HTTP response
        mock_embedding = np.random.rand(4096).tolist()
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"embedding": mock_embedding}
            
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # Test single text
            texts = ["This is a test text"]
            embeddings, errors = await indexer._generate_embeddings(texts)
            
            assert len(embeddings) == 1
            assert len(errors) == 0
            assert embeddings.shape == (1, 4096)
            np.testing.assert_array_equal(embeddings[0], mock_embedding)
    
    def test_create_faiss_index(self):
        """Test FAISS index creation."""
        indexer = VectorIndexer()
        
        # Create dummy embeddings
        embeddings = np.random.rand(10, 4096).astype('float32')
        
        # Create index
        index = indexer._create_faiss_index(embeddings)
        
        assert index.ntotal == 10
        assert index.d == 4096
    
    def test_faiss_search(self):
        """Test FAISS search functionality."""
        indexer = VectorIndexer()
        
        # Create dummy data
        embeddings = np.random.rand(5, 4096).astype('float32')
        index = indexer._create_faiss_index(embeddings)
        
        # Create query embedding
        query = np.random.rand(1, 4096).astype('float32')
        
        # Search
        scores, indices = index.search(query, k=3)
        
        assert len(scores[0]) == 3
        assert len(indices[0]) == 3
        assert all(0 <= idx < 5 for idx in indices[0])
    
    def test_save_and_load_index(self):
        """Test saving and loading index."""
        with tempfile.TemporaryDirectory() as temp_dir:
            indexer = VectorIndexer()
            indexer.index_path = os.path.join(temp_dir, "test_index.faiss")
            indexer.embeddings_path = os.path.join(temp_dir, "test_embeddings.npy")
            indexer.chunks_path = os.path.join(temp_dir, "test_chunks.pkl")
            
            # Create dummy data
            embeddings = np.random.rand(5, 4096).astype('float32')
            index = indexer._create_faiss_index(embeddings)
            chunk_metadata = [{"chunk_id": i, "text": f"chunk_{i}"} for i in range(5)]
            
            # Save
            indexer._save_index(index, embeddings, chunk_metadata)
            
            # Verify files exist
            assert os.path.exists(indexer.index_path)
            assert os.path.exists(indexer.embeddings_path)
            assert os.path.exists(indexer.chunks_path)
            
            # Load
            indexer.index = None
            indexer.embeddings = None
            indexer.chunk_metadata = []
            
            success = indexer.load_index()
            assert success is True
            assert indexer.index.ntotal == 5
            assert indexer.embeddings.shape == (5, 4096)
            assert len(indexer.chunk_metadata) == 5
    
    def test_get_index_stats(self):
        """Test index statistics."""
        indexer = VectorIndexer()
        
        # Test with no index loaded
        stats = indexer.get_index_stats()
        assert stats['index_loaded'] is False
        assert stats['vector_count'] == 0
        assert stats['dimension'] == 4096
        
        # Test with index loaded
        embeddings = np.random.rand(3, 4096).astype('float32')
        index = indexer._create_faiss_index(embeddings)
        indexer.index = index
        indexer.chunk_metadata = [{"id": i} for i in range(3)]
        
        stats = indexer.get_index_stats()
        assert stats['index_loaded'] is True
        assert stats['vector_count'] == 3
        assert stats['chunk_metadata_count'] == 3

if __name__ == "__main__":
    pytest.main([__file__])
