#!/usr/bin/env python3
"""Debug script to test the retriever without timing decorator."""

import asyncio
import sys
import os
import numpy as np

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.retriever import DocumentRetriever
from app.indexer import VectorIndexer
from app.db import db_manager

async def test_retriever_simple():
    """Test the retriever functionality without timing decorator."""
    print("🔍 Testing retriever (simple)...")
    
    # Create instances
    indexer = VectorIndexer()
    retriever = DocumentRetriever(indexer=indexer)
    
    # Load index
    index_loaded = indexer.load_index()
    print(f"Index loaded: {index_loaded}")
    
    if not index_loaded:
        print("❌ No index found")
        return
    
    # Get query embedding
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "llama3.2:latest", "prompt": "What is this website about?"}
        )
        if response.status_code == 200:
            data = response.json()
            query_embedding = np.array(data['embedding'])
        else:
            print("❌ Failed to get embedding")
            return
    
    print(f"Query embedding shape: {query_embedding.shape}")
    
    # Test search directly
    search_results = indexer.search(query_embedding, top_k=3)
    print(f"Search results: {len(search_results)}")
    
    for i, result in enumerate(search_results):
        print(f"Search result {i+1}: {result}")
        
        # Test _get_chunk_data directly
        chunk_data = retriever._get_chunk_data(result)
        print(f"  Chunk data type: {type(chunk_data)}")
        print(f"  Chunk data: {chunk_data}")

if __name__ == "__main__":
    asyncio.run(test_retriever_simple())
