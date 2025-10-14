#!/usr/bin/env python3
"""Debug script to test the indexer search."""

import sys
import os
import numpy as np

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.indexer import VectorIndexer
from app.db import db_manager

def test_search():
    """Test the indexer search functionality."""
    print("🔍 Testing indexer search...")
    
    # Create indexer
    indexer = VectorIndexer()
    
    # Load index
    index_loaded = indexer.load_index()
    print(f"Index loaded: {index_loaded}")
    
    if not index_loaded:
        print("❌ No index found")
        return
    
    # Get a real embedding from Ollama
    import httpx
    import asyncio
    
    async def get_embedding():
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/embeddings",
                json={"model": "llama3.2:latest", "prompt": "test query"}
            )
            if response.status_code == 200:
                data = response.json()
                return np.array(data['embedding'])
            return None
    
    query_embedding = asyncio.run(get_embedding())
    if query_embedding is None:
        print("❌ Failed to get embedding from Ollama")
        return
    print(f"Query embedding shape: {query_embedding.shape}")
    
    # Test search
    try:
        search_results = indexer.search(query_embedding, top_k=3)
        print(f"✅ Search successful: {len(search_results)} results")
        
        for i, result in enumerate(search_results):
            print(f"Result {i+1}:")
            print(f"  Type: {type(result)}")
            print(f"  Content: {result}")
            
    except Exception as e:
        print(f"❌ Search failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_search()
