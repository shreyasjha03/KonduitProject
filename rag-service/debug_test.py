#!/usr/bin/env python3
"""Debug script to test the retriever."""

import asyncio
import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.retriever import DocumentRetriever
from app.indexer import VectorIndexer
from app.db import db_manager

async def test_retriever():
    """Test the retriever functionality."""
    print("🔍 Testing retriever...")
    
    # Create instances
    indexer = VectorIndexer()
    retriever = DocumentRetriever(indexer=indexer)
    
    # Load index
    index_loaded = indexer.load_index()
    print(f"Index loaded: {index_loaded}")
    
    if not index_loaded:
        print("❌ No index found")
        return
    
    # Test retrieval
    try:
        result, timing = await retriever.retrieve("What is this website about?", top_k=3)
        print(f"✅ Retrieval successful: {len(result)} chunks, {timing:.2f}ms")
        
        for i, chunk in enumerate(result):
            print(f"Chunk {i+1}:")
            print(f"  Type: {type(chunk)}")
            print(f"  Content: {chunk}")
            if isinstance(chunk, dict):
                print(f"  URL: {chunk.get('url', 'No URL')}")
                print(f"  Text length: {len(chunk.get('chunk_text', ''))}")
            else:
                print(f"  Not a dict: {chunk}")
            
    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_retriever())
