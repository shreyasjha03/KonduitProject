"""FastAPI main application with all endpoints."""

import asyncio
import time
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .schemas import (
    CrawlRequest, CrawlResponse, IndexRequest, IndexResponse,
    AskRequest, AskResponse, HealthResponse, SourceSnippet, Timings
)
from .crawler import PoliteCrawler
from .indexer import VectorIndexer
from .retriever import DocumentRetriever
from .qa_engine import QARAGEngine
from .db import db_manager
from .logger import logger, obs_logger

# Global instances
crawler = PoliteCrawler()
indexer = VectorIndexer(embedding_model="llama3.2:latest")
retriever = DocumentRetriever(indexer=indexer, embedding_model="llama3.2:latest")
qa_engine = QARAGEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting RAG Service...")
    
    # Load existing index if available
    index_loaded = indexer.load_index()
    if index_loaded:
        logger.info("Loaded existing vector index")
    else:
        logger.info("No existing index found - will need to crawl and index")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Service...")

# Create FastAPI app
app = FastAPI(
    title="RAG Web Crawler Service",
    description="""
    A complete Retrieval-Augmented Generation (RAG) service that:
    
    * **Crawls websites** with robots.txt respect and domain boundaries
    * **Indexes content** with chunking and vector embeddings
    * **Provides grounded Q&A** with source citations and proper refusals
    * **Tracks performance** with comprehensive metrics and observability
    
    ## Key Features:
    - Polite crawling with configurable delays
    - FAISS vector indexing with embeddings
    - Context-only answering with safety guardrails
    - Source citations and relevance scoring
    - Performance metrics and error tracking
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system status."""
    timestamp = time.time()
    
    # Check Ollama availability
    ollama_available = await qa_engine.check_ollama_availability()
    
    # Check database availability
    try:
        db_stats = db_manager.get_crawl_stats()
        database_available = True
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        database_available = False
        db_stats = {}
    
    # Check vector index availability
    index_stats = indexer.get_index_stats()
    vector_index_available = index_stats.get('index_loaded', False)
    
    # Get query stats
    try:
        query_stats = obs_logger.get_query_stats()
    except Exception:
        query_stats = {}
    
    status = "healthy" if all([ollama_available, database_available]) else "degraded"
    
    return HealthResponse(
        status=status,
        timestamp=timestamp,
        ollama_available=ollama_available,
        database_available=database_available,
        vector_index_available=vector_index_available,
        stats={
            **db_stats,
            **index_stats,
            **query_stats
        }
    )

@app.post("/crawl", response_model=CrawlResponse)
async def crawl_website(request: CrawlRequest):
    """Crawl a website starting from the given URL."""
    start_time = time.time()
    
    try:
        # Update crawler settings
        crawler.crawl_delay_ms = request.crawl_delay_ms / 1000.0
        
        # Start crawling
        async with crawler:
            page_count, skipped_count, urls, errors = await crawler.crawl_site(
                start_url=str(request.start_url),
                max_pages=request.max_pages,
                max_depth=request.max_depth
            )
        
        crawl_time = time.time() - start_time
        
        logger.info(f"Crawl completed: {page_count} pages in {crawl_time:.2f}s")
        
        return CrawlResponse(
            page_count=page_count,
            skipped_count=skipped_count,
            urls=urls,
            crawl_time_seconds=crawl_time,
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"Crawl failed: {e}")
        raise HTTPException(status_code=500, detail=f"Crawl failed: {str(e)}")

@app.post("/index", response_model=IndexResponse)
async def index_content(request: IndexRequest):
    """Index crawled content with chunking and embeddings."""
    start_time = time.time()
    
    try:
        # Update indexer settings
        indexer.embedding_model = request.embedding_model
        
        # Start indexing
        result = await indexer.index_all_content(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        indexing_time = time.time() - start_time
        
        logger.info(f"Indexing completed: {result['vector_count']} vectors in {indexing_time:.2f}s")
        
        return IndexResponse(
            vector_count=result['vector_count'],
            chunk_count=result['chunk_count'],
            indexing_time_seconds=indexing_time,
            errors=result['errors']
        )
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Ask a question and get a grounded answer with sources."""
    total_start_time = time.time()
    
    try:
        # Retrieve relevant chunks
        start_retrieval = time.time()
        retrieved_chunks = await retriever.retrieve(
            question=request.question,
            top_k=request.top_k
        )
        retrieval_time = (time.time() - start_retrieval) * 1000
        
        # Rerank results
        ranked_chunks = retriever.rerank_results(
            question=request.question,
            chunks=retrieved_chunks,
            max_results=request.top_k
        )
        
        # Generate answer
        start_generation = time.time()
        answer, grounded, refusal_reason = await qa_engine.generate_answer(
            question=request.question,
            retrieved_chunks=ranked_chunks,
            top_k=request.top_k
        )
        generation_time = (time.time() - start_generation) * 1000
        
        # Extract sources
        sources = qa_engine.extract_sources_from_response(answer, ranked_chunks)
        
        total_time = (time.time() - total_start_time) * 1000
        
        timings = Timings(
            retrieval_ms=retrieval_time,
            generation_ms=generation_time,
            total_ms=total_time
        )
        
        # Log query
        db_manager.log_query(
            question=request.question,
            answer=answer,
            sources=sources,
            timings={
                "retrieval_ms": retrieval_time,
                "generation_ms": generation_time,
                "total_ms": total_time
            },
            grounded=grounded
        )
        
        # Log metrics
        obs_logger.log_query_metrics(
            question=request.question,
            answer=answer,
            sources=sources,
            timings={
                "retrieval_ms": retrieval_time,
                "generation_ms": generation_time,
                "total_ms": total_time
            },
            grounded=grounded,
            refusal_reason=refusal_reason
        )
        
        # Create source snippets
        source_snippets = [
            SourceSnippet(
                url=source['url'],
                snippet=source['snippet'],
                relevance_score=source['relevance_score']
            )
            for source in sources
        ]
        
        logger.info(f"Query processed: {total_time:.2f}ms total (grounded: {grounded})")
        
        return AskResponse(
            answer=answer,
            sources=source_snippets,
            timings=timings,
            grounded=grounded,
            refusal_reason=refusal_reason
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get system statistics and metrics."""
    try:
        # Get database stats
        db_stats = db_manager.get_crawl_stats()
        query_stats = db_manager.get_query_stats()
        
        # Get index stats
        index_stats = indexer.get_index_stats()
        
        # Get observability stats
        obs_stats = obs_logger.get_query_stats()
        
        return {
            "database": db_stats,
            "index": index_stats,
            "queries": query_stats,
            "observability": obs_stats
        }
        
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stats failed: {str(e)}")

@app.get("/pages")
async def get_crawled_pages():
    """Get list of all crawled pages."""
    try:
        pages = db_manager.get_crawled_pages()
        return {"pages": pages}
        
    except Exception as e:
        logger.error(f"Failed to get pages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get pages: {str(e)}")

@app.delete("/reset")
async def reset_all_data():
    """Reset all crawled data and indexes (for testing)."""
    try:
        # Clear database
        with db_manager.engine.connect() as conn:
            conn.execute("DELETE FROM query_logs")
            conn.execute("DELETE FROM text_chunks")
            conn.execute("DELETE FROM crawled_pages")
            conn.commit()
        
        # Clear index files
        import os
        import shutil
        data_dir = Path("data")
        if data_dir.exists():
            shutil.rmtree(data_dir)
            data_dir.mkdir(exist_ok=True)
        
        logger.info("All data reset successfully")
        return {"message": "All data reset successfully"}
        
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
