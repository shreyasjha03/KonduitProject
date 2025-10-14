"""SQLite database operations for URL-document mapping and metadata."""

import sqlite3
import json
import pickle
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import hashlib

from .logger import logger

class DatabaseManager:
    """Manages SQLite database for crawled pages and metadata."""
    
    def __init__(self, db_path: str = "data/meta.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Crawled pages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS crawled_pages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE NOT NULL,
                    title TEXT,
                    content_hash TEXT,
                    clean_text TEXT,
                    domain TEXT,
                    depth INTEGER,
                    crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status_code INTEGER,
                    content_length INTEGER,
                    text_length INTEGER
                )
            """)
            
            # Text chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS text_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    page_id INTEGER,
                    chunk_index INTEGER,
                    chunk_text TEXT,
                    chunk_start INTEGER,
                    chunk_end INTEGER,
                    embedding_id INTEGER,
                    FOREIGN KEY (page_id) REFERENCES crawled_pages (id)
                )
            """)
            
            # Query logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT,
                    answer TEXT,
                    sources TEXT,
                    timings TEXT,
                    grounded BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pages_url ON crawled_pages(url)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pages_domain ON crawled_pages(domain)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_page_id ON text_chunks(page_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_embedding_id ON text_chunks(embedding_id)")
            
            conn.commit()
    
    def add_crawled_page(self, url: str, title: str, content: str, clean_text: str,
                        domain: str, depth: int, status_code: int) -> int:
        """Add a crawled page to the database."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO crawled_pages 
                    (url, title, content_hash, clean_text, domain, depth, status_code, content_length, text_length)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (url, title, content_hash, clean_text, domain, depth, status_code, 
                      len(content), len(clean_text)))
                
                page_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Added page {page_id}: {url}")
                return page_id
                
            except sqlite3.IntegrityError:
                logger.warning(f"Page already exists: {url}")
                # Get existing page ID
                cursor.execute("SELECT id FROM crawled_pages WHERE url = ?", (url,))
                result = cursor.fetchone()
                return result[0] if result else None
    
    def get_crawled_pages(self, domain: Optional[str] = None) -> List[Dict]:
        """Get all crawled pages, optionally filtered by domain."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if domain:
                cursor.execute("SELECT * FROM crawled_pages WHERE domain = ?", (domain,))
            else:
                cursor.execute("SELECT * FROM crawled_pages")
            
            pages = cursor.fetchall()
            return [dict(page) for page in pages]
    
    def get_page_by_url(self, url: str) -> Optional[Dict]:
        """Get a page by its URL."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM crawled_pages WHERE url = ?", (url,))
            result = cursor.fetchone()
            return dict(result) if result else None
    
    def add_text_chunks(self, page_id: int, chunks: List[Tuple[str, int, int]]) -> List[int]:
        """Add text chunks for a page.
        
        Args:
            page_id: ID of the crawled page
            chunks: List of (chunk_text, start_pos, end_pos) tuples
            
        Returns:
            List of chunk IDs
        """
        chunk_ids = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for i, (chunk_text, start_pos, end_pos) in enumerate(chunks):
                cursor.execute("""
                    INSERT INTO text_chunks 
                    (page_id, chunk_index, chunk_text, chunk_start, chunk_end)
                    VALUES (?, ?, ?, ?, ?)
                """, (page_id, i, chunk_text, start_pos, end_pos))
                
                chunk_id = cursor.lastrowid
                chunk_ids.append(chunk_id)
            
            conn.commit()
            logger.info(f"Added {len(chunks)} chunks for page {page_id}")
        
        return chunk_ids
    
    def update_chunk_embedding_id(self, chunk_id: int, embedding_id: int):
        """Update chunk with its FAISS embedding ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE text_chunks 
                SET embedding_id = ? 
                WHERE id = ?
            """, (embedding_id, chunk_id))
            conn.commit()
    
    def get_text_chunks(self, chunk_ids: List[int]) -> List[Dict]:
        """Get text chunks by their IDs."""
        if not chunk_ids:
            return []
        
        placeholders = ','.join(['?'] * len(chunk_ids))
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT tc.*, cp.url, cp.title 
                FROM text_chunks tc
                JOIN crawled_pages cp ON tc.page_id = cp.id
                WHERE tc.id IN ({placeholders})
            """, chunk_ids)
            
            chunks = cursor.fetchall()
            return [dict(chunk) for chunk in chunks]
    
    def get_all_chunks_with_embeddings(self) -> List[Dict]:
        """Get all chunks that have embedding IDs."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT tc.*, cp.url, cp.title 
                FROM text_chunks tc
                JOIN crawled_pages cp ON tc.page_id = cp.id
                WHERE tc.embedding_id IS NOT NULL
                ORDER BY tc.embedding_id
            """)
            
            chunks = cursor.fetchall()
            return [dict(chunk) for chunk in chunks]
    
    def log_query(self, question: str, answer: str, sources: List[Dict], 
                  timings: Dict[str, float], grounded: bool):
        """Log a query for observability."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO query_logs 
                (question, answer, sources, timings, grounded)
                VALUES (?, ?, ?, ?, ?)
            """, (question, answer, json.dumps(sources), json.dumps(timings), grounded))
            conn.commit()
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total queries
            cursor.execute("SELECT COUNT(*) FROM query_logs")
            total_queries = cursor.fetchone()[0]
            
            # Grounded queries
            cursor.execute("SELECT COUNT(*) FROM query_logs WHERE grounded = 1")
            grounded_queries = cursor.fetchone()[0]
            
            # Average response times (last 100 queries)
            cursor.execute("""
                SELECT timings FROM query_logs 
                ORDER BY created_at DESC 
                LIMIT 100
            """)
            recent_timings = cursor.fetchall()
            
            avg_timings = {"retrieval_ms": 0, "generation_ms": 0, "total_ms": 0}
            if recent_timings:
                total_retrieval = 0
                total_generation = 0
                total_total = 0
                count = 0
                
                for (timings_json,) in recent_timings:
                    try:
                        timings = json.loads(timings_json)
                        total_retrieval += timings.get("retrieval_ms", 0)
                        total_generation += timings.get("generation_ms", 0)
                        total_total += timings.get("total_ms", 0)
                        count += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
                
                if count > 0:
                    avg_timings = {
                        "retrieval_ms": total_retrieval / count,
                        "generation_ms": total_generation / count,
                        "total_ms": total_total / count
                    }
            
            return {
                "total_queries": total_queries,
                "grounded_queries": grounded_queries,
                "grounding_rate": grounded_queries / total_queries if total_queries > 0 else 0,
                "avg_timings": avg_timings
            }
    
    def get_crawl_stats(self) -> Dict[str, Any]:
        """Get crawling statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total pages
            cursor.execute("SELECT COUNT(*) FROM crawled_pages")
            total_pages = cursor.fetchone()[0]
            
            # Pages by domain
            cursor.execute("""
                SELECT domain, COUNT(*) as count 
                FROM crawled_pages 
                GROUP BY domain
            """)
            pages_by_domain = dict(cursor.fetchall())
            
            # Total chunks
            cursor.execute("SELECT COUNT(*) FROM text_chunks")
            total_chunks = cursor.fetchone()[0]
            
            # Chunks with embeddings
            cursor.execute("SELECT COUNT(*) FROM text_chunks WHERE embedding_id IS NOT NULL")
            chunks_with_embeddings = cursor.fetchone()[0]
            
            return {
                "total_pages": total_pages,
                "pages_by_domain": pages_by_domain,
                "total_chunks": total_chunks,
                "chunks_with_embeddings": chunks_with_embeddings,
                "indexing_complete": chunks_with_embeddings == total_chunks if total_chunks > 0 else True
            }

# Global database manager instance
db_manager = DatabaseManager()
