"""Top-k retrieval logic for finding relevant chunks."""

import asyncio
import httpx
import numpy as np
from typing import List, Dict, Optional, Tuple
import time

# from .indexer import indexer  # Will be passed as parameter
from .db import db_manager
from .utils import calculate_relevance_score, truncate_snippet, simple_timing_decorator
from .logger import logger

class DocumentRetriever:
    """Handles retrieval of relevant documents from the vector index."""
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434", 
                 embedding_model: str = "llama3.2:latest", indexer=None):
        self.ollama_base_url = ollama_base_url
        self.embedding_model = embedding_model
        self.indexer = indexer
    
    @simple_timing_decorator
    async def retrieve(self, question: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k relevant chunks for a question.
        
        Returns:
            List of retrieved chunks
        """
        
        # Generate query embedding
        query_embedding = await self._get_query_embedding(question)
        if query_embedding is None:
            logger.error("Failed to generate query embedding")
            return []
        
        # Search vector index
        if self.indexer is None:
            logger.error("Indexer not initialized")
            return []
        search_results = self.indexer.search(query_embedding, top_k)
        
        # Get full chunk data from database
        retrieved_chunks = []
        for result in search_results:
            chunk_data = self._get_chunk_data(result)
            if chunk_data:
                retrieved_chunks.append(chunk_data)
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        
        return retrieved_chunks
    
    async def _get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Generate embedding for the query using Ollama."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_base_url}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": query
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return np.array(data['embedding'])
                else:
                    logger.error(f"Ollama API error for embeddings: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return None
    
    def _get_chunk_data(self, search_result: Dict) -> Optional[Dict]:
        """Get full chunk data including text and metadata."""
        try:
            chunk_id = search_result['chunk_id']
            chunks = db_manager.get_text_chunks([chunk_id])
            
            if not chunks:
                return None
            
            chunk = chunks[0]
            
            # Calculate relevance score based on text similarity
            relevance_score = calculate_relevance_score(
                search_result.get('page_title', ''), 
                chunk['chunk_text']
            )
            
            return {
                'chunk_id': chunk_id,
                'chunk_text': chunk['chunk_text'],
                'url': chunk['url'],
                'title': chunk['title'],
                'similarity_score': search_result.get('similarity_score', 0.0),
                'relevance_score': relevance_score,
                'snippet': truncate_snippet(chunk['chunk_text'], 150)
            }
            
        except Exception as e:
            logger.error(f"Error getting chunk data: {e}")
            return None
    
    def rerank_results(self, question: str, chunks: List[Dict], 
                      max_results: int = 5) -> List[Dict]:
        """Rerank results based on text similarity and other factors."""
        if not chunks:
            return []
        
        # Calculate combined scores
        for chunk in chunks:
            # Combine similarity score with relevance score
            similarity_score = chunk.get('similarity_score', 0.0)
            relevance_score = chunk.get('relevance_score', 0.0)
            
            # Normalize scores (similarity is distance, so lower is better)
            normalized_similarity = max(0, 1 - (similarity_score / 10))  # Rough normalization
            
            # Weighted combination
            combined_score = 0.7 * normalized_similarity + 0.3 * relevance_score
            chunk['combined_score'] = combined_score
        
        # Sort by combined score
        chunks.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Return top results
        return chunks[:max_results]
    
    def filter_low_quality_chunks(self, chunks: List[Dict], 
                                 min_score: float = 0.1) -> List[Dict]:
        """Filter out low-quality chunks based on scores."""
        filtered_chunks = []
        
        for chunk in chunks:
            combined_score = chunk.get('combined_score', 0.0)
            if combined_score >= min_score:
                filtered_chunks.append(chunk)
        
        logger.info(f"Filtered {len(chunks)} chunks to {len(filtered_chunks)} high-quality chunks")
        return filtered_chunks
    
    def get_retrieval_stats(self, chunks: List[Dict]) -> Dict[str, any]:
        """Get statistics about the retrieval results."""
        if not chunks:
            return {
                'chunk_count': 0,
                'avg_similarity_score': 0.0,
                'avg_relevance_score': 0.0,
                'avg_combined_score': 0.0,
                'unique_urls': 0,
                'total_text_length': 0
            }
        
        similarity_scores = [chunk.get('similarity_score', 0.0) for chunk in chunks]
        relevance_scores = [chunk.get('relevance_score', 0.0) for chunk in chunks]
        combined_scores = [chunk.get('combined_score', 0.0) for chunk in chunks]
        
        unique_urls = len(set(chunk.get('url', '') for chunk in chunks))
        total_text_length = sum(len(chunk.get('chunk_text', '')) for chunk in chunks)
        
        return {
            'chunk_count': len(chunks),
            'avg_similarity_score': sum(similarity_scores) / len(similarity_scores),
            'avg_relevance_score': sum(relevance_scores) / len(relevance_scores),
            'avg_combined_score': sum(combined_scores) / len(combined_scores),
            'unique_urls': unique_urls,
            'total_text_length': total_text_length
        }

# Global retriever instance
retriever = DocumentRetriever()
