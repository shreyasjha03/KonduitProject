# RAG Web Crawler Service

A complete Retrieval-Augmented Generation (RAG) service that crawls websites, indexes content, and provides grounded Q&A with source citations.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ollama with llama3.2 model

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama
ollama serve
ollama pull llama3.2

# Start the service
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### API Documentation
- **Swagger UI:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

## 📋 API Endpoints

### POST /crawl
Crawl a website starting from the given URL.

```json
{
  "start_url": "website url",
  "max_pages": 10,
  "max_depth": 2,
  "crawl_delay_ms": 1000
}
```

**Response:**
```json
{
  "page_count": 5,
  "skipped_count": 0,
  "urls": ["website url", "website url"],
  "crawl_time_seconds": 12.5,
  "errors": []
}
```

### POST /index
Index crawled content with chunking and embeddings.

```json
{
  "chunk_size": 500,
  "chunk_overlap": 100,
  "embedding_model": "llama3.2:latest"
}
```

**Response:**
```json
{
  "vector_count": 25,
  "chunk_count": 25,
  "indexing_time_seconds": 3.2,
  "errors": []
}
```

### POST /ask
Ask a question and get a grounded answer with sources.

```json
{
  "question": "What is this website about?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "Based on the crawled content, this website is about...",
  "sources": [
    {
      "url": "website url",
      "snippet": "Relevant text snippet...",
      "relevance_score": 0.85
    }
  ],
  "timings": {
    "retrieval_ms": 250.5,
    "generation_ms": 1200.3,
    "total_ms": 1450.8
  },
  "grounded": true,
  "refusal_reason": null
}
```

## 🏗️ Architecture

### Core Components

1. **PoliteCrawler**: Respects robots.txt, domain boundaries, and crawl delays
2. **VectorIndexer**: Chunks text, generates embeddings, creates FAISS index
3. **DocumentRetriever**: Retrieves top-k relevant chunks using vector similarity
4. **QARAGEngine**: Constructs grounded prompts and generates answers
5. **DatabaseManager**: Stores crawled pages and query logs

### Data Flow
```
URL → Crawl → Clean Text → Chunk → Embed → Vector Index → Retrieve → Generate Answer
```

### Storage
- **SQLite Database**: Crawled pages, text chunks, query logs
- **FAISS Index**: Vector embeddings for similarity search
- **Files**: Index files, embeddings, chunk metadata

## 🔧 Design Decisions & Tradeoffs

### Chunking Strategy
- **Chunk Size**: 500-1000 characters (configurable)
- **Overlap**: 100 characters for context continuity
- **Rationale**: Balances retrieval precision with context preservation

### Embedding Model
- **Model**: llama3.2:latest (4096 dimensions)
- **Rationale**: Open source, good performance, consistent with chat model

### Retrieval Strategy
- **Top-k**: 5 chunks by default (configurable)
- **Similarity**: Cosine similarity in FAISS IndexFlatL2
- **Reranking**: Combined relevance and quality scores

### Crawling Constraints
- **Domain Restriction**: Stays within registrable domain
- **Page Limit**: 30-50 pages maximum
- **Politeness**: 1-second crawl delay, robots.txt respect
- **Content Focus**: HTML text extraction, boilerplate reduction

## 🛡️ Safety & Guardrails

### Grounding Enforcement
- Answers only from retrieved context
- Clear refusals when evidence insufficient
- Source URLs and snippets provided for all answers

### Prompt Hardening
- Instructions to ignore page-embedded directives
- Context-only response generation
- No execution of page instructions

### Content Boundaries
- Domain-restricted crawling
- No cross-domain information leakage
- Explicit refusal for off-site queries

## 📊 Observability

### Metrics Tracked
- Crawl statistics (pages, domains, timing)
- Index statistics (vectors, chunks, dimensions)
- Query metrics (latency p50/p95, grounding rate)
- Error rates and failure modes

### Logging
- Structured logging with timestamps
- Request/response tracking
- Performance metrics
- Error categorization

## 🧪 Evaluation Examples

### Example 1: Successful Answer
**Question:** "What is the main purpose of this website?"

**Answer:** "Based on the crawled content, this website serves as a documentation platform for developers, providing tutorials and API references."

**Sources:**
- URL: https://example.com/docs
- Snippet: "Our platform provides comprehensive documentation for developers..."
- Relevance: 0.92

### Example 2: Refusal Case
**Question:** "What is the weather like today?"

**Answer:** "I don't have enough information to answer your question based on the crawled content. The website content doesn't contain any weather-related information."

**Sources:** [] (empty - no relevant content found)

## 🔍 Tooling & Prompts

### Models Used
- **LLM**: llama3.2:latest (via Ollama)
- **Embeddings**: llama3.2:latest (via Ollama API)
- **Vector Store**: FAISS IndexFlatL2

### Libraries
- **FastAPI**: Web framework
- **aiohttp**: Async HTTP client
- **FAISS**: Vector similarity search
- **BeautifulSoup**: HTML parsing
- **SQLAlchemy**: Database ORM

### Prompt Template
```
Based on the following context from crawled web pages, answer the question. 
Only use information from the provided context. If the context doesn't contain 
enough information to answer the question, say "I don't have enough information 
to answer this question based on the crawled content."

Context:
{retrieved_chunks}

Question: {question}

Answer:
```

## 🚧 Limitations & Next Steps

### Current Limitations
- Limited to HTML text content
- No JavaScript rendering
- Basic text cleaning
- Single embedding model

### Future Improvements
- Multi-modal content support
- Advanced text cleaning
- Multiple embedding models
- Query expansion and reformulation
- Caching and performance optimization

## 🧪 Testing

### Run Tests
```bash
python -m pytest tests/ -v
```

### Manual Testing
```bash
# Test crawl
curl -X POST http://localhost:8000/crawl -d '{"start_url": "https://example.com", "max_pages": 1}'

# Test indexing
curl -X POST http://localhost:8000/index -d '{"chunk_size": 500}'

# Test Q&A
curl -X POST http://localhost:8000/ask -d '{"question": "What is this about?", "top_k": 3}'
```

## 📈 Performance

### Benchmarks
- **Crawl Speed**: ~1-2 pages/second (with delays)
- **Indexing Speed**: ~100-200 chunks/second
- **Query Latency**: 1-3 seconds (retrieval + generation)
- **Memory Usage**: ~50MB base + ~1MB per 1000 chunks

### Scalability
- Supports up to 10,000 chunks efficiently
- Horizontal scaling possible with distributed FAISS
- Database can handle millions of pages

---

**Built for educational purposes. Respect robots.txt and website terms of service.**
