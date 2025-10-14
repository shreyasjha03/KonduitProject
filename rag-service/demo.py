#!/usr/bin/env python3
"""
Demo script showing complete RAG service workflow.
Demonstrates crawl → index → ask pipeline with examples.
"""

import requests
import time
import json
from typing import Dict, Any

class RAGDemo:
    """Demonstration of RAG service capabilities."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def run_complete_demo(self):
        """Run complete demonstration."""
        print("🎬 RAG Service Complete Demo")
        print("="*50)
        
        # Step 1: Health check
        self.check_health()
        
        # Step 2: Crawl a website
        crawl_result = self.demo_crawl()
        
        # Step 3: Index content
        index_result = self.demo_index()
        
        # Step 4: Ask questions
        self.demo_ask_questions()
        
        # Step 5: Show stats
        self.show_stats()
        
        print("\n🎉 Demo completed successfully!")
    
    def check_health(self):
        """Check service health."""
        print("\n1️⃣ Checking service health...")
        
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health = response.json()
                print(f"✅ Service status: {health['status']}")
                print(f"✅ Ollama available: {health['ollama_available']}")
                print(f"✅ Database available: {health['database_available']}")
                print(f"✅ Vector index available: {health['vector_index_available']}")
            else:
                print(f"❌ Health check failed: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ Cannot connect to service: {e}")
            raise
    
    def demo_crawl(self) -> Dict[str, Any]:
        """Demonstrate website crawling."""
        print("\n2️⃣ Crawling a website...")
        
        # Use a simple, reliable website for demo
        crawl_request = {
            "start_url": "https://httpbin.org",
            "max_pages": 10,
            "max_depth": 2,
            "crawl_delay_ms": 500
        }
        
        print(f"📡 Starting crawl of {crawl_request['start_url']}...")
        
        try:
            response = requests.post(f"{self.base_url}/crawl", json=crawl_request)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Crawled {result['page_count']} pages in {result['crawl_time_seconds']:.2f}s")
                print(f"📄 URLs crawled: {len(result['urls'])}")
                
                if result['errors']:
                    print(f"⚠️ Errors encountered: {len(result['errors'])}")
                
                return result
            else:
                print(f"❌ Crawl failed: HTTP {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"❌ Crawl error: {e}")
            return {}
    
    def demo_index(self) -> Dict[str, Any]:
        """Demonstrate content indexing."""
        print("\n3️⃣ Indexing crawled content...")
        
        index_request = {
            "chunk_size": 800,
            "chunk_overlap": 100,
            "embedding_model": "llama3.2"
        }
        
        print("🔄 Generating embeddings and building vector index...")
        
        try:
            response = requests.post(f"{self.base_url}/index", json=index_request)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Indexed {result['vector_count']} vectors in {result['indexing_time_seconds']:.2f}s")
                print(f"📦 Created {result['chunk_count']} text chunks")
                
                if result['errors']:
                    print(f"⚠️ Errors during indexing: {len(result['errors'])}")
                
                return result
            else:
                print(f"❌ Indexing failed: HTTP {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"❌ Indexing error: {e}")
            return {}
    
    def demo_ask_questions(self):
        """Demonstrate Q&A capabilities."""
        print("\n4️⃣ Asking questions...")
        
        # Test questions
        questions = [
            "What is this website about?",
            "What information is available here?",
            "Tell me about cats and dogs",  # Should be refused
            "What services does this site offer?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n❓ Question {i}: {question}")
            
            try:
                start_time = time.time()
                response = requests.post(f"{self.base_url}/ask", json={
                    "question": question,
                    "top_k": 3
                })
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Show answer
                    print(f"💬 Answer: {result['answer'][:200]}{'...' if len(result['answer']) > 200 else ''}")
                    
                    # Show grounding status
                    if result['grounded']:
                        print("✅ Answer is grounded in crawled content")
                    else:
                        print("⚠️ Answer not grounded - refused appropriately")
                        if result.get('refusal_reason'):
                            print(f"   Reason: {result['refusal_reason']}")
                    
                    # Show sources
                    sources = result.get('sources', [])
                    if sources:
                        print(f"📚 Sources ({len(sources)}):")
                        for j, source in enumerate(sources[:2], 1):  # Show top 2
                            print(f"   {j}. {source['url']} (score: {source['relevance_score']:.3f})")
                    
                    # Show timing
                    timings = result.get('timings', {})
                    print(f"⏱️ Timing: {timings.get('total_ms', 0):.1f}ms total "
                          f"({timings.get('retrieval_ms', 0):.1f}ms retrieval, "
                          f"{timings.get('generation_ms', 0):.1f}ms generation)")
                    
                else:
                    print(f"❌ Query failed: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Query error: {e}")
            
            time.sleep(1)  # Brief pause between questions
    
    def show_stats(self):
        """Show system statistics."""
        print("\n5️⃣ System Statistics...")
        
        try:
            response = requests.get(f"{self.base_url}/stats")
            
            if response.status_code == 200:
                stats = response.json()
                
                # Database stats
                db_stats = stats.get('database', {})
                print(f"📊 Database:")
                print(f"   - Total pages: {db_stats.get('total_pages', 0)}")
                print(f"   - Total chunks: {db_stats.get('total_chunks', 0)}")
                print(f"   - Indexed chunks: {db_stats.get('chunks_with_embeddings', 0)}")
                
                # Query stats
                query_stats = stats.get('queries', {})
                print(f"🔍 Queries:")
                print(f"   - Total queries: {query_stats.get('total_queries', 0)}")
                print(f"   - Grounded queries: {query_stats.get('grounded_queries', 0)}")
                print(f"   - Grounding rate: {query_stats.get('grounding_rate', 0):.1%}")
                
                # Performance stats
                avg_timings = query_stats.get('avg_timings', {})
                print(f"⚡ Performance:")
                print(f"   - Avg total latency: {avg_timings.get('total_ms', 0):.1f}ms")
                print(f"   - Avg retrieval: {avg_timings.get('retrieval_ms', 0):.1f}ms")
                print(f"   - Avg generation: {avg_timings.get('generation_ms', 0):.1f}ms")
                
            else:
                print(f"❌ Stats request failed: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Stats error: {e}")

def main():
    """Main demo function."""
    demo = RAGDemo()
    
    print("🚀 Starting RAG Service Demo...")
    print("Make sure the service is running: python -m app.main")
    print()
    
    try:
        demo.run_complete_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")

if __name__ == "__main__":
    main()
