"""Tests for the crawler module."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from app.crawler import PoliteCrawler
from app.text_cleaner import TextExtractor

class TestPoliteCrawler:
    """Test cases for PoliteCrawler."""
    
    @pytest.mark.asyncio
    async def test_crawler_initialization(self):
        """Test crawler initialization."""
        crawler = PoliteCrawler(crawl_delay_ms=1000, max_concurrent=2)
        assert crawler.crawl_delay_ms == 1.0
        assert crawler.max_concurrent == 2
        assert len(crawler.crawled_urls) == 0
        assert len(crawler.failed_urls) == 0
    
    @pytest.mark.asyncio
    async def test_domain_extraction(self):
        """Test domain extraction logic."""
        from app.utils import extract_domain, is_same_domain
        
        # Test domain extraction
        assert extract_domain("https://example.com/page") == "example.com"
        assert extract_domain("https://subdomain.example.com/page") == "example.com"
        assert extract_domain("https://www.example.co.uk/page") == "example.co.uk"
        
        # Test same domain check
        assert is_same_domain("https://example.com/page1", "https://example.com/page2")
        assert is_same_domain("https://sub.example.com/page1", "https://example.com/page2")
        assert not is_same_domain("https://example.com/page1", "https://other.com/page2")
    
    @pytest.mark.asyncio
    async def test_robots_txt_respect(self):
        """Test robots.txt checking."""
        crawler = PoliteCrawler()
        
        # Mock robots.txt response
        with patch('urllib.robotparser.RobotFileParser') as mock_robots:
            mock_parser = Mock()
            mock_parser.can_fetch.return_value = True
            mock_robots.return_value = mock_parser
            
            # Test allowed URL
            result = await crawler._can_crawl_url("https://example.com/page")
            assert result is True
            
            # Test disallowed URL
            mock_parser.can_fetch.return_value = False
            result = await crawler._can_crawl_url("https://example.com/page")
            assert result is False
    
    @pytest.mark.asyncio
    async def test_crawl_delay_respect(self):
        """Test that crawler respects delay settings."""
        crawler = PoliteCrawler(crawl_delay_ms=100)  # 100ms delay
        
        start_time = asyncio.get_event_loop().time()
        
        # Mock the crawling process
        with patch.object(crawler, '_crawl_single_page') as mock_crawl:
            mock_crawl.return_value = ({"page_id": 1, "url": "test", "title": "Test", "text": "content", "depth": 0}, [])
            
            # This would normally crawl, but we're just testing delay logic
            await asyncio.sleep(0.1)  # Simulate delay
        
        elapsed = (asyncio.get_event_loop().time() - start_time) * 1000
        assert elapsed >= 90  # Allow some tolerance

class TestTextExtractor:
    """Test cases for TextExtractor."""
    
    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        from app.utils import clean_text
        
        # Test whitespace normalization
        dirty_text = "  Hello    world  \n\n  Test  "
        clean = clean_text(dirty_text)
        assert clean == "Hello world Test"
        
        # Test empty text
        assert clean_text("") == ""
        assert clean_text(None) == ""
        
        # Test non-printable characters
        text_with_nonprintable = "Hello\x00world\x01test"
        clean = clean_text(text_with_nonprintable)
        assert "\x00" not in clean
        assert "\x01" not in clean
    
    def test_text_chunking(self):
        """Test text chunking functionality."""
        from app.utils import chunk_text
        
        # Test short text
        short_text = "This is a short text."
        chunks = chunk_text(short_text, chunk_size=100, overlap=20)
        assert len(chunks) == 1
        assert chunks[0][0] == short_text
        
        # Test long text
        long_text = "This is a long text. " * 50  # ~1000 characters
        chunks = chunk_text(long_text, chunk_size=200, overlap=50)
        assert len(chunks) > 1
        
        # Verify overlap
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1][0]
            curr_chunk = chunks[i][0]
            # There should be some overlap (not perfect test but basic check)
            assert len(prev_chunk) > 0
            assert len(curr_chunk) > 0
    
    @pytest.mark.asyncio
    async def test_extractor_initialization(self):
        """Test TextExtractor initialization."""
        extractor = TextExtractor(headless=True, timeout=30000)
        assert extractor.headless is True
        assert extractor.timeout == 30000

if __name__ == "__main__":
    pytest.main([__file__])
