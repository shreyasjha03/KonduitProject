"""Polite web crawler with robots.txt respect and domain restrictions."""

import asyncio
import aiohttp
import time
from typing import Set, List, Dict, Optional, Tuple, Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import hashlib

from .text_cleaner import TextExtractor
from .db import db_manager
from .utils import extract_domain, normalize_url, is_same_domain
from .logger import logger, obs_logger

class PoliteCrawler:
    """Polite web crawler that respects robots.txt and domain boundaries."""
    
    def __init__(self, crawl_delay_ms: int = 1000, max_concurrent: int = 3):
        self.crawl_delay_ms = crawl_delay_ms / 1000.0  # Convert to seconds
        self.max_concurrent = max_concurrent
        self.session: Optional[aiohttp.ClientSession] = None
        self.robots_cache: Dict[str, RobotFileParser] = {}
        self.crawled_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'RAG-Crawler/1.0 (+https://example.com/bot)'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def crawl_site(self, start_url: str, max_pages: int = 50, 
                        max_depth: int = 3) -> Tuple[int, int, List[str], List[str]]:
        """Crawl a website starting from the given URL.
        
        Returns:
            Tuple of (page_count, skipped_count, urls, errors)
        """
        start_time = time.time()
        target_domain = extract_domain(start_url)
        urls_to_crawl = [(start_url, 0)]  # (url, depth)
        crawled_count = 0
        skipped_count = 0
        errors = []
        
        logger.info(f"Starting crawl of {start_url} (domain: {target_domain})")
        
        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async with TextExtractor() as extractor:
            while urls_to_crawl and crawled_count < max_pages:
                # Process URLs in batches
                batch_size = min(self.max_concurrent, max_pages - crawled_count)
                batch = []
                
                for _ in range(batch_size):
                    if not urls_to_crawl:
                        break
                    batch.append(urls_to_crawl.pop(0))
                
                if not batch:
                    break
                
                # Process batch concurrently
                tasks = [
                    self._crawl_page_with_semaphore(
                        semaphore, extractor, url, depth, target_domain
                    )
                    for url, depth in batch
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        errors.append(str(result))
                        skipped_count += 1
                    else:
                        page_data, new_links = result
                        if page_data:
                            crawled_count += 1
                            
                            # Add new links to crawl queue
                            for link, link_depth in new_links:
                                if (link_depth < max_depth and 
                                    link not in self.crawled_urls and 
                                    link not in self.failed_urls):
                                    urls_to_crawl.append((link, link_depth))
                        else:
                            skipped_count += 1
                
                # Respect crawl delay
                if urls_to_crawl and crawled_count < max_pages:
                    await asyncio.sleep(self.crawl_delay_ms)
        
        crawl_time = time.time() - start_time
        
        # Log metrics
        obs_logger.log_crawl_metrics(
            start_url=start_url,
            page_count=crawled_count,
            skipped_count=skipped_count,
            crawl_time=crawl_time,
            errors=errors
        )
        
        logger.info(f"Crawl completed: {crawled_count} pages in {crawl_time:.2f}s")
        
        return crawled_count, skipped_count, list(self.crawled_urls), errors
    
    async def _crawl_page_with_semaphore(self, semaphore: asyncio.Semaphore,
                                       extractor: TextExtractor, url: str, 
                                       depth: int, target_domain: str) -> Tuple[Optional[Dict], List[Tuple[str, int]]]:
        """Crawl a single page with semaphore control."""
        async with semaphore:
            return await self._crawl_single_page(extractor, url, depth, target_domain)
    
    async def _crawl_single_page(self, extractor: TextExtractor, url: str, 
                               depth: int, target_domain: str) -> Tuple[Optional[Dict], List[Tuple[str, int]]]:
        """Crawl a single page."""
        normalized_url = normalize_url(url)
        
        # Check if already crawled
        if normalized_url in self.crawled_urls:
            return None, []
        
        # Check domain restriction
        if not is_same_domain(url, f"https://{target_domain}"):
            logger.warning(f"Skipping cross-domain URL: {url}")
            return None, []
        
        # Check robots.txt
        if not await self._can_crawl_url(url):
            logger.info(f"Robots.txt disallows crawling: {url}")
            return None, []
        
        try:
            # Extract text using Playwright
            result = await extractor.extract_text_with_playwright(url)
            
            if not result['success']:
                logger.warning(f"Failed to extract content from {url}: {result.get('error', 'Unknown error')}")
                self.failed_urls.add(normalized_url)
                return None, []
            
            # Store in database
            page_id = db_manager.add_crawled_page(
                url=url,
                title=result['title'],
                content="",  # We don't store raw HTML
                clean_text=result['text'],
                domain=target_domain,
                depth=depth,
                status_code=result.get('status_code', 200)
            )
            
            # Mark as crawled
            self.crawled_urls.add(normalized_url)
            
            # Prepare new links for crawling
            new_links = []
            for link in result['links']:
                new_links.append((link, depth + 1))
            
            logger.info(f"Crawled page {page_id}: {url} ({len(result['text'])} chars)")
            
            return {
                'page_id': page_id,
                'url': url,
                'title': result['title'],
                'text': result['text'],
                'depth': depth
            }, new_links
            
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            self.failed_urls.add(normalized_url)
            return None, []
    
    async def _can_crawl_url(self, url: str) -> bool:
        """Check if URL can be crawled according to robots.txt."""
        try:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            
            # Check cache first
            if robots_url in self.robots_cache:
                robots = self.robots_cache[robots_url]
            else:
                # Fetch robots.txt
                robots = RobotFileParser()
                robots.set_url(robots_url)
                
                try:
                    robots.read()
                    self.robots_cache[robots_url] = robots
                except Exception as e:
                    logger.warning(f"Could not fetch robots.txt for {parsed.netloc}: {e}")
                    # Allow crawling if robots.txt is not accessible
                    return True
            
            # Check if our user agent can crawl this URL
            user_agent = 'RAG-Crawler'
            can_crawl = robots.can_fetch(user_agent, url)
            
            # Also check with wildcard user agent
            if not can_crawl:
                can_crawl = robots.can_fetch('*', url)
            
            return can_crawl
            
        except Exception as e:
            logger.error(f"Error checking robots.txt for {url}: {e}")
            # Default to allowing crawl if there's an error
            return True
    
    def get_robots_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about robots.txt cache."""
        return {
            "cached_robots_files": len(self.robots_cache),
            "crawled_urls": len(self.crawled_urls),
            "failed_urls": len(self.failed_urls)
        }

# Global crawler instance
crawler = PoliteCrawler()
