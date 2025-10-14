"""HTML to clean text extraction using BeautifulSoup and Chromium."""

import asyncio
import re
from typing import Optional, Dict, Any
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page, Browser
import time

from .logger import logger
from .utils import clean_text, extract_domain

class TextExtractor:
    """Extracts clean text from HTML pages using Chromium for JS-rendered content."""
    
    def __init__(self, headless: bool = True, timeout: int = 30000):
        self.headless = headless
        self.timeout = timeout
        self.browser: Optional[Browser] = None
        self.playwright = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    async def extract_text_with_playwright(self, url: str) -> Dict[str, Any]:
        """Extract text using Playwright for JS-rendered content."""
        if not self.browser:
            raise RuntimeError("Browser not initialized. Use async context manager.")
        
        page = await self.browser.new_page()
        try:
            # Set user agent
            await page.set_extra_http_headers({
                'User-Agent': 'RAG-Crawler/1.0 (+https://example.com/bot)'
            })
            
            # Navigate to page
            response = await page.goto(url, timeout=self.timeout, wait_until='networkidle')
            
            if not response or response.status >= 400:
                return {
                    'success': False,
                    'error': f'HTTP {response.status if response else "No response"}',
                    'title': '',
                    'text': '',
                    'links': []
                }
            
            # Wait for content to load
            await page.wait_for_timeout(2000)
            
            # Get page title
            title = await page.title()
            
            # Extract main content using multiple strategies
            text = await self._extract_main_content(page)
            
            # Extract links
            links = await self._extract_links(page, url)
            
            # Clean the extracted text
            clean_text_content = clean_text(text)
            
            logger.info(f"Extracted {len(clean_text_content)} chars from {url}")
            
            return {
                'success': True,
                'title': title,
                'text': clean_text_content,
                'links': links,
                'status_code': response.status
            }
            
        except Exception as e:
            logger.error(f"Playwright extraction failed for {url}: {e}")
            return {
                'success': False,
                'error': str(e),
                'title': '',
                'text': '',
                'links': []
            }
        finally:
            await page.close()
    
    async def _extract_main_content(self, page: Page) -> str:
        """Extract main content using multiple CSS selectors."""
        content_selectors = [
            'main',
            'article',
            '[role="main"]',
            '.content',
            '.main-content',
            '#content',
            '#main',
            '.post',
            '.entry',
            'body'
        ]
        
        for selector in content_selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    text = await element.inner_text()
                    if len(text.strip()) > 100:  # Minimum content length
                        return text
            except Exception:
                continue
        
        # Fallback: get all text from body
        try:
            body = await page.query_selector('body')
            return await body.inner_text() if body else ""
        except Exception:
            return ""
    
    async def _extract_links(self, page: Page, base_url: str) -> list:
        """Extract all links from the page."""
        try:
            links = await page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links.map(link => ({
                        href: link.href,
                        text: link.textContent.trim()
                    })).filter(link => link.href && link.href.startsWith('http'));
                }
            """)
            
            # Filter links to same domain
            base_domain = extract_domain(base_url)
            same_domain_links = []
            
            for link in links:
                try:
                    link_domain = extract_domain(link['href'])
                    if link_domain == base_domain:
                        same_domain_links.append(link['href'])
                except Exception:
                    continue
            
            return same_domain_links
            
        except Exception as e:
            logger.error(f"Failed to extract links: {e}")
            return []
    
    def extract_text_with_bs4(self, html: str, url: str) -> Dict[str, Any]:
        """Extract text using BeautifulSoup (fallback method)."""
        try:
            soup = BeautifulSoup(html, 'lxml')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get title
            title_elem = soup.find('title')
            title = title_elem.get_text().strip() if title_elem else ""
            
            # Try to find main content
            main_content = None
            content_selectors = [
                'main', 'article', '[role="main"]', '.content', 
                '.main-content', '#content', '#main', '.post', '.entry'
            ]
            
            for selector in content_selectors:
                element = soup.select_one(selector)
                if element and len(element.get_text().strip()) > 100:
                    main_content = element
                    break
            
            # Fallback to body
            if not main_content:
                main_content = soup.find('body')
            
            # Extract text
            if main_content:
                text = main_content.get_text()
            else:
                text = soup.get_text()
            
            # Clean text
            clean_text_content = clean_text(text)
            
            # Extract links
            links = []
            base_domain = extract_domain(url)
            
            for link in soup.find_all('a', href=True):
                try:
                    href = urljoin(url, link['href'])
                    link_domain = extract_domain(href)
                    if link_domain == base_domain and href.startswith('http'):
                        links.append(href)
                except Exception:
                    continue
            
            return {
                'success': True,
                'title': title,
                'text': clean_text_content,
                'links': links
            }
            
        except Exception as e:
            logger.error(f"BeautifulSoup extraction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'title': '',
                'text': '',
                'links': []
            }

# Global text extractor instance
text_extractor = TextExtractor()
