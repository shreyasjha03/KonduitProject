"""Text chunking, embedding generation, and FAISS vector indexing."""

import asyncio
import httpx
import numpy as np
import faiss
import pickle
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import time

from .db import db_manager
from .utils import chunk_text, timing_decorator
from .logger import logger, obs_logger

class VectorIndexer:
    """Handles text chunking, embedding generation, and FAISS indexing."""
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434", 
                 embedding_model: str = "llama3.2:latest"):
        self.ollama_base_url = ollama_base_url
        self.embedding_model = embedding_model
        self.index_path = Path("data/index.faiss")
        self.embeddings_path = Path("data/embeddings.npy")
        self.chunks_path = Path("data/chunks.pkl")
        self.index: Optional[faiss.Index] = None
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_metadata: List[Dict] = []
        self.dimension = 4096  # Llama 3.2 embedding dimension
    
    async def index_all_content(self, chunk_size: int = 800, chunk_overlap: int = 100) -> Dict:
        """Index all crawled content with chunking and embeddings."""
        start_time = time.time()
        
        # Get all crawled pages
        pages = db_manager.get_crawled_pages()
        if not pages:
            logger.warning("No crawled pages found. Run crawl first.")
            return {
                'vector_count': 0,
                'chunk_count': 0,
                'indexing_time_seconds': 0,
                'errors': ["No crawled pages found"]
            }
        
        logger.info(f"Starting indexing of {len(pages)} pages")
        
        # Chunk all pages
        all_chunks = []
        chunk_metadata = []
        embedding_id = 0
        
        for page in pages:
            page_chunks, positions = self._chunk_page_text(
                page['clean_text'], chunk_size, chunk_overlap
            )
            
            # Store chunks in database
            chunk_ids = db_manager.add_text_chunks(page['id'], positions)
            
            for i, (chunk_text, chunk_id) in enumerate(zip(page_chunks, chunk_ids)):
                all_chunks.append(chunk_text)
                chunk_metadata.append({
                    'chunk_id': chunk_id,
                    'page_id': page['id'],
                    'page_url': page['url'],
                    'page_title': page['title'],
                    'chunk_index': i,
                    'embedding_id': embedding_id
                })
                embedding_id += 1
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
        
        # Generate embeddings
        embedding_result, _ = await self._generate_embeddings(all_chunks)
        embeddings, embedding_errors = embedding_result
        
        if not embeddings.size:
            logger.error("No embeddings generated")
            return {
                'vector_count': 0,
                'chunk_count': len(all_chunks),
                'indexing_time_seconds': time.time() - start_time,
                'errors': embedding_errors + ["No embeddings generated"]
            }
        
        # Create FAISS index
        index = self._create_faiss_index(embeddings)
        
        # Update chunk metadata with embedding IDs
        for i, metadata in enumerate(chunk_metadata):
            db_manager.update_chunk_embedding_id(metadata['chunk_id'], i)
            metadata['embedding_id'] = i
        
        # Save index and metadata
        self._save_index(index, embeddings, chunk_metadata)
        
        indexing_time = time.time() - start_time
        
        # Log metrics
        obs_logger.log_indexing_metrics(
            chunk_count=len(all_chunks),
            vector_count=len(embeddings),
            indexing_time=indexing_time,
            errors=embedding_errors
        )
        
        logger.info(f"Indexing completed: {len(embeddings)} vectors in {indexing_time:.2f}s")
        
        # Auto-reload the index after indexing
        try:
            self.load_index()
            logger.info("Index automatically reloaded after indexing")
        except Exception as e:
            logger.warning(f"Failed to auto-reload index: {e}")
        
        return {
            'vector_count': len(embeddings),
            'chunk_count': len(all_chunks),
            'indexing_time_seconds': indexing_time,
            'errors': embedding_errors
        }
    
    def _chunk_page_text(self, text: str, chunk_size: int, chunk_overlap: int) -> Tuple[List[str], List[Tuple[str, int, int]]]:
        """Chunk page text and return chunks with positions."""
        if not text or len(text.strip()) == 0:
            return [], []
        
        chunks_with_positions = chunk_text(text, chunk_size, chunk_overlap)
        chunks = [chunk for chunk, _, _ in chunks_with_positions]
        positions = [(chunk, start, end) for chunk, start, end in chunks_with_positions]
        
        return chunks, positions
    
    @timing_decorator
    async def _generate_embeddings(self, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Generate embeddings for texts using Ollama."""
        if not texts:
            return np.array([]), []
        
        embeddings = []
        errors = []
        
        # Process in batches to avoid overwhelming Ollama
        batch_size = 5
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                batch_embeddings = await self._get_embeddings_batch(batch)
                embeddings.extend(batch_embeddings)
                
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
            except Exception as e:
                error_msg = f"Failed to generate embeddings for batch {i//batch_size + 1}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                # Add zero embeddings for failed batch
                embeddings.extend([np.zeros(self.dimension) for _ in batch])
        
        return np.array(embeddings), errors
    
    async def _get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for a batch of texts."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            embeddings = []
            
            for text in texts:
                try:
                    response = await client.post(
                        f"{self.ollama_base_url}/api/embeddings",
                        json={
                            "model": self.embedding_model,
                            "prompt": text
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        embedding = np.array(data['embedding'])
                        embeddings.append(embedding)
                    else:
                        logger.error(f"Ollama API error: {response.status_code}")
                        embeddings.append(np.zeros(self.dimension))
                        
                except Exception as e:
                    logger.error(f"Error getting embedding: {e}")
                    embeddings.append(np.zeros(self.dimension))
            
            return embeddings
    
    def _create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create FAISS index from embeddings."""
        dimension = embeddings.shape[1]
        
        # Create index (L2 distance)
        index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        index.add(embeddings.astype('float32'))
        
        logger.info(f"Created FAISS index with {index.ntotal} vectors")
        
        return index
    
    def _save_index(self, index: faiss.Index, embeddings: np.ndarray, 
                   chunk_metadata: List[Dict]):
        """Save index, embeddings, and metadata to disk."""
        # Ensure data directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(index, str(self.index_path))
        
        # Save embeddings
        np.save(self.embeddings_path, embeddings)
        
        # Save chunk metadata
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(chunk_metadata, f)
        
        logger.info(f"Saved index to {self.index_path}")
    
    def load_index(self) -> bool:
        """Load existing index from disk."""
        try:
            if not self.index_path.exists():
                logger.info("No existing index found")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            
            # Load embeddings
            if self.embeddings_path.exists():
                self.embeddings = np.load(self.embeddings_path)
            
            # Load chunk metadata
            if self.chunks_path.exists():
                with open(self.chunks_path, 'rb') as f:
                    self.chunk_metadata = pickle.load(f)
            
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar chunks using FAISS."""
        if self.index is None:
            logger.error("Index not loaded. Call load_index() first.")
            return []
        
        # Ensure query embedding is the right shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search index
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Get chunk metadata for results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunk_metadata):
                metadata = self.chunk_metadata[idx].copy()
                metadata['similarity_score'] = float(score)
                results.append(metadata)
        
        return results
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector index."""
        if self.index is None:
            return {
                'index_loaded': False,
                'vector_count': 0,
                'dimension': self.dimension
            }
        
        return {
            'index_loaded': True,
            'vector_count': self.index.ntotal,
            'dimension': self.dimension,
            'chunk_metadata_count': len(self.chunk_metadata),
            'index_type': 'IndexFlatL2'
        }

# Global indexer instance
indexer = VectorIndexer()
