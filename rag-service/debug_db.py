#!/usr/bin/env python3
"""Debug script to test the database."""

import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.db import db_manager

def test_database():
    """Test the database functionality."""
    print("🔍 Testing database...")
    
    try:
        # Get all chunks
        chunks = db_manager.get_text_chunks([3])  # We know chunk 3 exists
        print(f"✅ Database query successful: {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}:")
            print(f"  Type: {type(chunk)}")
            print(f"  Keys: {list(chunk.keys()) if isinstance(chunk, dict) else 'Not a dict'}")
            print(f"  Content: {chunk}")
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_database()
