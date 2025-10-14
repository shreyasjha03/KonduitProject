"""Logging and observability setup."""

import logging
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Create logs directory
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / "rag_service.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("rag_service")

class ObservabilityLogger:
    """Enhanced logging for observability and metrics."""
    
    def __init__(self):
        self.metrics_file = logs_dir / "metrics.jsonl"
        self.metrics_file.touch()
    
    def log_crawl_metrics(self, start_url: str, page_count: int, skipped_count: int, 
                         crawl_time: float, errors: list):
        """Log crawling metrics."""
        metrics = {
            "event": "crawl_completed",
            "timestamp": datetime.utcnow().isoformat(),
            "start_url": start_url,
            "page_count": page_count,
            "skipped_count": skipped_count,
            "crawl_time_seconds": crawl_time,
            "pages_per_second": page_count / crawl_time if crawl_time > 0 else 0,
            "error_count": len(errors),
            "errors": errors
        }
        self._write_metrics(metrics)
        logger.info(f"Crawl completed: {page_count} pages in {crawl_time:.2f}s")
    
    def log_indexing_metrics(self, chunk_count: int, vector_count: int, 
                           indexing_time: float, errors: list):
        """Log indexing metrics."""
        metrics = {
            "event": "indexing_completed",
            "timestamp": datetime.utcnow().isoformat(),
            "chunk_count": chunk_count,
            "vector_count": vector_count,
            "indexing_time_seconds": indexing_time,
            "chunks_per_second": chunk_count / indexing_time if indexing_time > 0 else 0,
            "error_count": len(errors),
            "errors": errors
        }
        self._write_metrics(metrics)
        logger.info(f"Indexing completed: {chunk_count} chunks, {vector_count} vectors in {indexing_time:.2f}s")
    
    def log_query_metrics(self, question: str, answer: str, sources: list, 
                         timings: Dict[str, float], grounded: bool, 
                         refusal_reason: Optional[str] = None):
        """Log query metrics."""
        metrics = {
            "event": "query_processed",
            "timestamp": datetime.utcnow().isoformat(),
            "question": question[:100] + "..." if len(question) > 100 else question,
            "answer_length": len(answer),
            "source_count": len(sources),
            "grounded": grounded,
            "refusal_reason": refusal_reason,
            "timings": timings,
            "retrieval_latency_ms": timings.get("retrieval_ms", 0),
            "generation_latency_ms": timings.get("generation_ms", 0),
            "total_latency_ms": timings.get("total_ms", 0)
        }
        self._write_metrics(metrics)
        
        status = "GROUNDED" if grounded else "REFUSED"
        logger.info(f"Query {status}: {timings.get('total_ms', 0):.2f}ms total, "
                   f"{timings.get('retrieval_ms', 0):.2f}ms retrieval, "
                   f"{timings.get('generation_ms', 0):.2f}ms generation")
    
    def log_error(self, error_type: str, message: str, context: Dict[str, Any] = None):
        """Log error events."""
        metrics = {
            "event": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": error_type,
            "message": message,
            "context": context or {}
        }
        self._write_metrics(metrics)
        logger.error(f"{error_type}: {message}")
    
    def log_performance_metrics(self, operation: str, duration_ms: float, 
                              additional_metrics: Dict[str, Any] = None):
        """Log performance metrics."""
        metrics = {
            "event": "performance",
            "timestamp": datetime.utcnow().isoformat(),
            "operation": operation,
            "duration_ms": duration_ms,
            **(additional_metrics or {})
        }
        self._write_metrics(metrics)
        
        if duration_ms > 1000:  # Log slow operations
            logger.warning(f"Slow operation: {operation} took {duration_ms:.2f}ms")
    
    def _write_metrics(self, metrics: Dict[str, Any]):
        """Write metrics to JSONL file."""
        try:
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        except Exception as e:
            logger.error(f"Failed to write metrics: {e}")
    
    def get_recent_metrics(self, limit: int = 100) -> list:
        """Get recent metrics for analysis."""
        try:
            with open(self.metrics_file, "r") as f:
                lines = f.readlines()
                recent_lines = lines[-limit:] if len(lines) > limit else lines
                return [json.loads(line.strip()) for line in recent_lines if line.strip()]
        except Exception as e:
            logger.error(f"Failed to read metrics: {e}")
            return []
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query statistics from recent metrics."""
        metrics = self.get_recent_metrics(1000)
        query_metrics = [m for m in metrics if m.get("event") == "query_processed"]
        
        if not query_metrics:
            return {"total_queries": 0, "grounding_rate": 0}
        
        total_queries = len(query_metrics)
        grounded_queries = sum(1 for m in query_metrics if m.get("grounded", False))
        
        # Calculate latency statistics
        total_latencies = [m.get("total_latency_ms", 0) for m in query_metrics]
        retrieval_latencies = [m.get("retrieval_latency_ms", 0) for m in query_metrics]
        generation_latencies = [m.get("generation_latency_ms", 0) for m in query_metrics]
        
        return {
            "total_queries": total_queries,
            "grounded_queries": grounded_queries,
            "grounding_rate": grounded_queries / total_queries if total_queries > 0 else 0,
            "avg_total_latency_ms": sum(total_latencies) / len(total_latencies) if total_latencies else 0,
            "avg_retrieval_latency_ms": sum(retrieval_latencies) / len(retrieval_latencies) if retrieval_latencies else 0,
            "avg_generation_latency_ms": sum(generation_latencies) / len(generation_latencies) if generation_latencies else 0,
            "p95_total_latency_ms": sorted(total_latencies)[int(0.95 * len(total_latencies))] if total_latencies else 0,
            "p95_retrieval_latency_ms": sorted(retrieval_latencies)[int(0.95 * len(retrieval_latencies))] if retrieval_latencies else 0,
            "p95_generation_latency_ms": sorted(generation_latencies)[int(0.95 * len(generation_latencies))] if generation_latencies else 0
        }

# Global observability logger instance
obs_logger = ObservabilityLogger()
