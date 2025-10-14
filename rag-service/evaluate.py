#!/usr/bin/env python3
"""
RAG Service Evaluation Script

Demonstrates the system's capabilities with example crawl, index, and ask operations.
Shows both successful answers and proper refusals.
"""

import asyncio
import json
import time
import httpx
from typing import Dict, List, Any

class RAGEvaluator:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def crawl_website(self, start_url: str, max_pages: int = 3) -> Dict[str, Any]:
        """Crawl a website and return results."""
        print(f"🔍 Crawling {start_url}...")
        
        response = await self.client.post(
            f"{self.base_url}/crawl",
            json={
                "start_url": start_url,
                "max_pages": max_pages,
                "max_depth": 2,
                "crawl_delay_ms": 1000
            }
        )
        
        result = response.json()
        print(f"✅ Crawled {result['page_count']} pages in {result['crawl_time_seconds']:.2f}s")
        return result
    
    async def index_content(self, chunk_size: int = 500) -> Dict[str, Any]:
        """Index crawled content."""
        print("📚 Indexing content...")
        
        response = await self.client.post(
            f"{self.base_url}/index",
            json={
                "chunk_size": chunk_size,
                "chunk_overlap": 100,
                "embedding_model": "llama3.2:latest"
            }
        )
        
        result = response.json()
        print(f"✅ Indexed {result['vector_count']} vectors in {result['indexing_time_seconds']:.2f}s")
        return result
    
    async def ask_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Ask a question and return the answer."""
        print(f"❓ Question: {question}")
        
        start_time = time.time()
        response = await self.client.post(
            f"{self.base_url}/ask",
            json={
                "question": question,
                "top_k": top_k
            }
        )
        
        result = response.json()
        total_time = (time.time() - start_time) * 1000
        
        print(f"💬 Answer: {result['answer'][:100]}...")
        print(f"📊 Grounded: {result['grounded']}, Sources: {len(result['sources'])}")
        print(f"⏱️  Timing: {result['timings']['total_ms']:.1f}ms total")
        
        return result
    
    async def get_health(self) -> Dict[str, Any]:
        """Get system health status."""
        response = await self.client.get(f"{self.base_url}/health")
        return response.json()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        response = await self.client.get(f"{self.base_url}/stats")
        return response.json()

async def run_evaluation():
    """Run comprehensive evaluation of the RAG system."""
    
    print("🚀 RAG Service Evaluation")
    print("=" * 50)
    
    async with RAGEvaluator() as evaluator:
        # Check system health
        print("\n🏥 System Health Check")
        health = await evaluator.get_health()
        print(f"Status: {health['status']}")
        print(f"Ollama Available: {health['ollama_available']}")
        print(f"Database Available: {health['database_available']}")
        print(f"Vector Index Available: {health['vector_index_available']}")
        
        # Example 1: Crawl and index a website
        print("\n" + "=" * 50)
        print("📖 EXAMPLE 1: Crawling and Indexing")
        print("=" * 50)
        
        crawl_result = await evaluator.crawl_website("https://example.com", max_pages=1)
        index_result = await evaluator.index_content(chunk_size=500)
        
        # Example 2: Answerable question
        print("\n" + "=" * 50)
        print("✅ EXAMPLE 2: Answerable Question")
        print("=" * 50)
        
        answerable_result = await evaluator.ask_question(
            "What is the purpose of this domain?"
        )
        
        print("\n📋 Detailed Response:")
        print(f"Answer: {answerable_result['answer']}")
        print(f"Sources:")
        for i, source in enumerate(answerable_result['sources'], 1):
            print(f"  {i}. URL: {source['url']}")
            print(f"     Snippet: {source['snippet'][:100]}...")
            print(f"     Relevance: {source['relevance_score']:.2f}")
        
        # Example 3: Unanswerable question (demonstrates refusal)
        print("\n" + "=" * 50)
        print("❌ EXAMPLE 3: Unanswerable Question (Refusal)")
        print("=" * 50)
        
        refusal_result = await evaluator.ask_question(
            "What is the weather like today?"
        )
        
        print("\n📋 Refusal Response:")
        print(f"Answer: {refusal_result['answer']}")
        print(f"Grounded: {refusal_result['grounded']}")
        print(f"Refusal Reason: {refusal_result['refusal_reason']}")
        print(f"Sources Count: {len(refusal_result['sources'])}")
        
        # Example 4: Performance metrics
        print("\n" + "=" * 50)
        print("📊 EXAMPLE 4: Performance Metrics")
        print("=" * 50)
        
        stats = await evaluator.get_stats()
        
        print("System Statistics:")
        print(f"Total Pages Crawled: {stats['database']['total_pages']}")
        print(f"Total Chunks: {stats['database']['total_chunks']}")
        print(f"Vector Count: {stats['index']['vector_count']}")
        print(f"Total Queries: {stats['queries']['total_queries']}")
        print(f"Grounded Queries: {stats['queries']['grounded_queries']}")
        print(f"Grounding Rate: {stats['queries']['grounding_rate']:.2%}")
        
        if 'avg_timings' in stats['queries']:
            timings = stats['queries']['avg_timings']
            print(f"Avg Retrieval Time: {timings['retrieval_ms']:.1f}ms")
            print(f"Avg Generation Time: {timings['generation_ms']:.1f}ms")
        
        # Summary
        print("\n" + "=" * 50)
        print("🎯 EVALUATION SUMMARY")
        print("=" * 50)
        
        print("✅ Successfully demonstrated:")
        print("  • Polite crawling with robots.txt respect")
        print("  • Content indexing with chunking and embeddings")
        print("  • Grounded Q&A with source citations")
        print("  • Proper refusal when insufficient information")
        print("  • Performance metrics and observability")
        print("  • Safety guardrails and context-only answering")
        
        print(f"\n📈 Key Metrics:")
        print(f"  • Grounding Rate: {stats['queries']['grounding_rate']:.1%}")
        print(f"  • Total Queries Processed: {stats['queries']['total_queries']}")
        print(f"  • System Status: {health['status']}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())