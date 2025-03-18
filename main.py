import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from asyncio import Semaphore
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from urllib.parse import urljoin, urlparse

# Third-party imports
import aiofiles
import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Configuration class
class Config:
    """Configuration for the DocsiteToMD scraper."""
    
    def __init__(
        self,
        base_dir: str = "docs_output",
        max_concurrent: int = 3,
        rate_limit: int = 5,
        batch_size: int = 10,
        request_timeout: int = 30,
        retry_attempts: int = 5,
        retry_backoff_factor: int = 1,
        retry_max_wait: int = 30,
        chunk_size: int = 500000,  # 500KB chunks for processing large HTML
        gc_threshold: int = 1000,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        user_agent: str = "Mozilla/5.0 (compatible; DocsiteScraperBot/1.0)",
        processing_timeout: int = 300,  # 5 minutes timeout for processing
        domain_rate_limit: int = 3,     # Maximum concurrent requests per domain
        request_delay: float = 1.0,     # Delay between requests to the same domain
    ):
        self.base_dir = Path(base_dir)
        self.max_concurrent = max_concurrent
        self.rate_limit = rate_limit
        self.batch_size = batch_size
        self.request_timeout = request_timeout
        self.retry_attempts = retry_attempts
        self.retry_backoff_factor = retry_backoff_factor
        self.retry_max_wait = retry_max_wait
        self.chunk_size = chunk_size
        self.gc_threshold = gc_threshold
        self.log_level = log_level
        self.log_file = log_file
        self.user_agent = user_agent
        self.processing_timeout = processing_timeout
        self.domain_rate_limit = domain_rate_limit
        self.request_delay = request_delay

# Logging setup
class LoggingSetup:
    """Configure logging with proper handlers and formatters."""
    
    @staticmethod
    def setup(config: Config) -> logging.Logger:
        logger = logging.getLogger("docsitetomd")
        logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers will filter
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(getattr(logging, config.log_level))
        console_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
        console_handler.setFormatter(logging.Formatter(console_format, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(console_handler)
        
        # File handler (if log_file is specified)
        if config.log_file:
            log_dir = Path(config.log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(config.log_file)
            file_handler.setLevel(logging.DEBUG)
            file_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
            file_handler.setFormatter(logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S"))
            logger.addHandler(file_handler)
        
        return logger

# URL and domain validation
class URLValidator:
    """Validate URLs and domains."""
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL for security and correctness."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            return False
    
    @staticmethod
    def is_same_domain(url: str, allowed_domain: str) -> bool:
        """Check if URL belongs to allowed domain."""
        try:
            parsed_url = urlparse(url)
            return parsed_url.netloc == allowed_domain or parsed_url.netloc.endswith(f".{allowed_domain}")
        except Exception:
            return False
    
    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalize URL by removing fragments and query parameters."""
        try:
            parsed_url = urlparse(url)
            clean_url = parsed_url._replace(fragment="", query="").geturl()
            return clean_url
        except Exception:
            return url

# Resource management
class ResourceManager:
    """Manage resources like connections and sessions."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.session = None
        self.conn = None
        
    async def initialize(self):
        """Initialize resources."""
        self.logger.debug("Initializing resources")
        self.conn = aiohttp.TCPConnector(
            limit=self.config.max_concurrent,
            ttl_dns_cache=300,
            use_dns_cache=True,
            limit_per_host=self.config.domain_rate_limit,
            force_close=False,
        )
        
        timeout = aiohttp.ClientTimeout(
            total=self.config.request_timeout,
            connect=10,
            sock_read=10,
            sock_connect=10
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=self.conn,
            headers={
                "Connection": "keep-alive",
                "User-Agent": self.config.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        self.logger.debug("Resources initialized")
        
    async def cleanup(self):
        """Clean up resources."""
        self.logger.debug("Cleaning up resources")
        if self.session:
            await self.session.close()
        # Allow time for connections to close properly
        await asyncio.sleep(0.25)
        self.logger.debug("Resources cleaned up")

# URL Queue management
class URLQueue:
    """Manage URL queue with prioritization and deduplication."""
    
    def __init__(self, logger: logging.Logger):
        self.queue = asyncio.Queue()
        self.in_queue = set()
        self.logger = logger
        self._size = 0  # Track size manually to avoid race conditions
        self._lock = asyncio.Lock()  # Lock for thread-safe operations
        
    async def add(self, url: str, domain: str, path_parts: Optional[List[str]] = None):
        """Add URL to queue if not already present."""
        async with self._lock:
            if url not in self.in_queue:
                self.in_queue.add(url)
                await self.queue.put((url, domain, path_parts))
                self._size += 1
                self.logger.debug(f"Added URL to queue: {url}, queue size: {self._size}")
            
    async def get(self):
        """Get next URL from queue."""
        item = await self.queue.get()
        async with self._lock:
            self._size -= 1
        return item
        
    def task_done(self):
        """Mark task as done."""
        self.queue.task_done()
        
    async def join(self):
        """Wait for all tasks to complete."""
        await self.queue.join()
        
    def empty(self):
        """Check if queue is empty."""
        return self.queue.empty()
    
    def size(self):
        """Get queue size."""
        return self._size

# Progress tracking
class ProgressTracker:
    """Track and report progress."""
    
    def __init__(self, total: int = 0, log_interval: int = 10, logger: logging.Logger = None):
        self.total = total
        self.processed = 0
        self.failed = 0
        self.log_interval = log_interval
        self.start_time = datetime.now()
        self.logger = logger
        self.last_log_time = time.time()
        
    def update(self, success: bool = True):
        """Update progress."""
        self.processed += 1
        if not success:
            self.failed += 1
            
        # Log at regular intervals or when a certain number of items are processed
        current_time = time.time()
        if self.processed % self.log_interval == 0 or (current_time - self.last_log_time) > 10:
            self._log_progress()
            self.last_log_time = current_time
            
    def _log_progress(self):
        """Log progress."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.processed / elapsed if elapsed > 0 else 0
        
        if self.total > 0:
            percent = (self.processed / self.total) * 100
            eta = (self.total - self.processed) / rate if rate > 0 else 0
            self.logger.info(
                f"Progress: {self.processed}/{self.total} ({percent:.1f}%) URLs processed, "
                f"{self.failed} failed, {rate:.2f} URLs/s, ETA: {eta:.0f}s"
            )
        else:
            self.logger.info(
                f"Progress: {self.processed} URLs processed, {self.failed} failed, "
                f"{rate:.2f} URLs/s"
            )

# Worker pool management
class WorkerPool:
    """Manage worker tasks."""
    
    def __init__(self, worker_count: int, worker_func, name: str = "worker", logger: logging.Logger = None):
        self.worker_count = worker_count
        self.worker_func = worker_func
        self.name = name
        self.workers = []
        self.logger = logger
        
    async def start(self):
        """Start workers."""
        self.logger.debug(f"Starting {self.worker_count} {self.name} workers")
        self.workers = [
            asyncio.create_task(self.worker_func(), name=f"{self.name}-{i}")
            for i in range(self.worker_count)
        ]
        self.logger.info(f"Created {len(self.workers)} {self.name} worker tasks")
        
        # Даем воркерам время на инициализацию
        await asyncio.sleep(0.5)  # Увеличенное время ожидания для полной инициализации воркеров
        
        # Проверяем, не завершились ли воркеры преждевременно
        active_workers = sum(1 for w in self.workers if not w.done())
        if active_workers < len(self.workers):
            self.logger.warning(f"Only {active_workers}/{len(self.workers)} {self.name} workers started successfully")
            
            # Проверяем ошибки в завершившихся воркерах
            for i, worker in enumerate(self.workers):
                if worker.done():
                    try:
                        exc = worker.exception()
                        if exc:
                            self.logger.error(f"{self.name}-{i} failed with error: {str(exc)}")
                    except asyncio.InvalidStateError:
                        # Task was cancelled
                        self.logger.warning(f"{self.name}-{i} was cancelled")
            
            # Пытаемся перезапустить отказавшие воркеры
            if active_workers < len(self.workers):
                self.logger.info(f"Attempting to restart failed {self.name} workers")
                for i, worker in enumerate(self.workers):
                    if worker.done():
                        self.logger.debug(f"Restarting {self.name}-{i}")
                        self.workers[i] = asyncio.create_task(self.worker_func(), name=f"{self.name}-{i}")
                
                # Проверяем снова после перезапуска
                await asyncio.sleep(0.5)
                active_workers = sum(1 for w in self.workers if not w.done())
                self.logger.info(f"After restart: {active_workers}/{len(self.workers)} {self.name} workers active")
        
    async def stop(self):
        """Stop workers."""
        for worker in self.workers:
            if not worker.done():
                worker.cancel()
            
        if self.workers:
            # Wait for all workers to complete
            await asyncio.gather(*self.workers, return_exceptions=True)
            
            # Check for errors
            for i, worker in enumerate(self.workers):
                if worker.done() and not worker.cancelled():
                    try:
                        exc = worker.exception()
                        if exc:
                            self.logger.error(f"{self.name}-{i} exited with error: {str(exc)}")
                    except asyncio.InvalidStateError:
                        pass
            
            self.workers = []
            self.logger.info(f"Stopped {self.name} workers")

# Main scraper class
class DocsiteToMD:
    """Main class for converting documentation sites to Markdown."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.logger = LoggingSetup.setup(self.config)
        self.resource_manager = ResourceManager(self.config, self.logger)
        self.url_queue = URLQueue(self.logger)
        self.markdown_queue = asyncio.Queue()
        self.visited_urls = set()
        self.failed_urls = set()
        self.processing = True  # Устанавливаем True по умолчанию, чтобы воркеры не завершались раньше времени
        self.progress = ProgressTracker(logger=self.logger)
        self.domain_limiters = {}
        self.domain_last_request = {}  # Track last request time per domain
        self._processed_docs = set()
        
        # Create semaphores
        self.semaphore = Semaphore(self.config.max_concurrent)
        self.rate_limit = Semaphore(self.config.rate_limit)
        
        # Status tracking
        self.last_activity_time = time.time()
        self.status_task = None
        
        # Queue tracking
        self.markdown_items_added = 0
        self.markdown_items_processed = 0
        
    async def initialize(self):
        """Initialize the scraper."""
        self.logger.info("Starting initialization")
        
        # Create directories
        self.base_dir = self.config.base_dir
        await self._ensure_directory(self.base_dir)
        
        self.sites_dir = self.base_dir / "sites"
        await self._ensure_directory(self.sites_dir)
        
        self.combined_dir = self.base_dir / "combined"
        await self._ensure_directory(self.combined_dir)
        
        # Initialize resources
        await self.resource_manager.initialize()
        
        # Start worker pools
        self.url_workers = WorkerPool(
            worker_count=min(5, self.config.max_concurrent),  # Ограничиваем до 5 воркеров для стабильности
            worker_func=self._url_worker,
            name="url-worker",
            logger=self.logger
        )
        
        self.markdown_workers = WorkerPool(
            worker_count=min(3, max(1, self.config.max_concurrent // 2)),  # Ограничиваем до 3 воркеров для обработки MD
            worker_func=self._markdown_worker,
            name="markdown-worker",
            logger=self.logger
        )
        
        # Запускаем воркеры с улучшенным логированием ошибок
        self.logger.debug("About to start URL workers")
        await self.url_workers.start()
        self.logger.debug("URL workers started, about to start markdown workers")
        await self.markdown_workers.start()
        self.logger.debug("Markdown workers started")
        
        # Verify workers started properly
        active_url_workers = sum(1 for w in self.url_workers.workers if not w.done())
        active_md_workers = sum(1 for w in self.markdown_workers.workers if not w.done())
        
        self.logger.info(f"Workers started - URL: {active_url_workers}/{len(self.url_workers.workers)}, Markdown: {active_md_workers}/{len(self.markdown_workers.workers)}")
        
        if active_url_workers == 0 or active_md_workers == 0:
            self.logger.error("Failed to start all workers, some may have exited prematurely")
        
        # Start status monitoring
        self.status_task = asyncio.create_task(self._monitor_status())
        
        self.logger.info("Initialization completed")
    
    async def _monitor_status(self):
        """Monitor scraper status and detect hangs."""
        try:
            while self.processing:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                current_time = time.time()
                if current_time - self.last_activity_time > 60:  # No activity for 1 minute
                    queue_size = self.url_queue.size()
                    markdown_size = self.markdown_queue.qsize()
                    self.logger.warning(
                        f"No activity for 60 seconds. URL queue: {queue_size}, "
                        f"Markdown queue: {markdown_size}, "
                        f"Visited URLs: {len(self.visited_urls)}, "
                        f"Failed URLs: {len(self.failed_urls)}, "
                        f"Markdown items added: {self.markdown_items_added}, "
                        f"Markdown items processed: {self.markdown_items_processed}"
                    )
                    
                    # If both queues are empty but we're still processing, something might be wrong
                    if queue_size == 0 and markdown_size == 0 and self.processing:
                        self.logger.warning("Both queues are empty but processing is still active. Forcing completion.")
                        self.processing = False
                        break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in status monitor: {str(e)}", exc_info=True)
    
    async def _ensure_directory(self, path: Path):
        """Ensure directory exists."""
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {path}")
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize filename to be safe for all filesystems."""
        return (
            "".join(c if c.isalnum() or c in "-_ " else "_" for c in name)
            .lower()
            .replace(" ", "_")
        )
    
    async def _get_domain_dir(self, domain: str) -> Path:
        """Create and return domain-specific directory."""
        domain_dir = self.sites_dir / self._sanitize_filename(domain)
        await self._ensure_directory(domain_dir)
        return domain_dir
    
    async def _create_directory_structure(self, domain: str, path_parts: List[str]) -> Path:
        """Create directory structure for a document."""
        try:
            current_path = await self._get_domain_dir(domain)
            for part in path_parts:
                if part:  # Skip empty parts
                    part = self._sanitize_filename(part)
                    current_path = current_path / part
                    await self._ensure_directory(current_path)
            return current_path
        except Exception as e:
            self.logger.error(f"Error creating directory structure for {domain}/{'/'.join(path_parts)}: {str(e)}", exc_info=True)
            # Fallback to domain directory
            return await self._get_domain_dir(domain)
    
    def _get_domain_limiter(self, domain: str) -> Semaphore:
        """Get or create a rate limiter for a specific domain."""
        if domain not in self.domain_limiters:
            # Use a more conservative limit for concurrent requests per domain
            self.domain_limiters[domain] = Semaphore(self.config.domain_rate_limit)
        return self.domain_limiters[domain]
    
    async def _respect_rate_limits(self, domain: str):
        """Respect rate limits for a specific domain."""
        # Check when the last request to this domain was made
        last_request_time = self.domain_last_request.get(domain, 0)
        current_time = time.time()
        
        # If we've made a request recently, wait before making another
        if current_time - last_request_time < self.config.request_delay:
            delay = self.config.request_delay - (current_time - last_request_time)
            # Add a small random jitter to avoid thundering herd
            delay += random.uniform(0.1, 0.5)
            self.logger.debug(f"Rate limiting for domain {domain}, waiting {delay:.2f}s")
            await asyncio.sleep(delay)
        
        # Update the last request time
        self.domain_last_request[domain] = time.time()
    
    async def _make_request(self, url: str) -> Optional[str]:
        """Make HTTP request with proper error handling."""
        self.logger.debug(f"Making request to: {url}")
        domain = urlparse(url).netloc
        
        # Respect rate limits
        await self._respect_rate_limits(domain)
        
        for attempt in range(self.config.retry_attempts):
            try:
                async with self.resource_manager.session.get(url, allow_redirects=True) as response:
                    self.logger.debug(f"Response status: {response.status}, URL: {url}")
                    
                    if response.status == 429:  # Rate limit
                        retry_after = int(response.headers.get("Retry-After", 30))
                        self.logger.warning(f"Rate limited for {url}, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    elif response.status >= 400:
                        wait_time = min(
                            self.config.retry_backoff_factor * (2 ** attempt) + random.uniform(0, 1),
                            self.config.retry_max_wait
                        )
                        self.logger.warning(f"Error response {response.status} for {url}, waiting {wait_time:.2f}s")
                        await asyncio.sleep(wait_time)
                        continue
                        
                    response.raise_for_status()
                    
                    # Check content type
                    content_type = response.headers.get("Content-Type", "")
                    if not (content_type.startswith("text/html") or "text/plain" in content_type):
                        self.logger.warning(f"Skipping non-HTML content: {url} (Content-Type: {content_type})")
                        return None
                    
                    # Process HTML content in a streaming fashion
                    html_chunks = []
                    async for chunk in response.content.iter_chunked(self.config.chunk_size):
                        html_chunks.append(chunk.decode('utf-8', errors='replace'))
                    
                    # Update last activity time
                    self.last_activity_time = time.time()
                    
                    return "".join(html_chunks)
                    
            except aiohttp.ClientError as e:
                wait_time = min(
                    self.config.retry_backoff_factor * (2 ** attempt) + random.uniform(0, 1),
                    self.config.retry_max_wait
                )
                self.logger.warning(f"Request failed for {url} (attempt {attempt+1}/{self.config.retry_attempts}): {str(e)}")
                self.logger.debug(f"Waiting {wait_time:.2f}s before retry")
                await asyncio.sleep(wait_time)
            except asyncio.TimeoutError:
                self.logger.warning(f"Request timed out for {url} (attempt {attempt+1}/{self.config.retry_attempts})")
                await asyncio.sleep(5)
            except Exception as e:
                self.logger.error(f"Unexpected error for {url}: {str(e)}", exc_info=True)
                break
                
        self.logger.error(f"Failed to fetch {url} after {self.config.retry_attempts} attempts")
        self.failed_urls.add(url)
        return None
    
    async def _process_html_content(self, html_content: str, url: str, allowed_domain: str, path_parts: Optional[List[str]] = None) -> Optional[Tuple[Dict, BeautifulSoup]]:
        """Process HTML content and extract useful information."""
        try:
            self.logger.debug(f"Processing content for URL: {url}")
            
            # Process in chunks if content is large
            if len(html_content) > 1_000_000:  # 1MB
                chunks = [
                    html_content[i : i + 1_000_000]
                    for i in range(0, len(html_content), 1_000_000)
                ]
                soup = BeautifulSoup("", "html.parser")  # Create empty soup
                for chunk in chunks:
                    chunk_soup = BeautifulSoup(chunk, "html.parser")
                    soup.append(chunk_soup)
            else:
                soup = BeautifulSoup(html_content, "html.parser")
            
            # Find main content
            main_content = await self._find_main_content(soup)
            
            if not main_content:
                self.logger.warning(f"No main content found for: {url}")
                return None
            
            self.logger.debug(f"Found main content for: {url}")
            
            # Extract path parts
            parsed_url = urlparse(url)
            url_path = parsed_url.path.strip("/")
            current_path_parts = url_path.split("/") if url_path else ["index"]
            
            current_path_parts = [part for part in current_path_parts if part]
            if not current_path_parts:
                current_path_parts = ["index"]
            
            if path_parts:
                current_path_parts = path_parts + current_path_parts
            
            # Extract content
            title = await self._extract_title(soup, main_content)
            content = await self._extract_content(main_content)
            headers = await self._extract_headers(main_content)
            
            self.logger.debug(
                f"Extracted content - Title: {title}, Headers count: {len(headers)}, Content length: {len(content)}"
            )
            
            doc_data = {
                "url": url,
                "title": title,
                "content": content,
                "headers": headers,
                "path_parts": current_path_parts,
                "domain": allowed_domain,
            }
            
            # Update last activity time
            self.last_activity_time = time.time()
            
            return doc_data, soup
            
        except Exception as e:
            self.logger.error(f"Error processing content for {url}: {str(e)}", exc_info=True)
            return None
    
    async def _find_main_content(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """Find the main content container with fallback mechanisms."""
        # Try different content selectors in order of specificity
        selectors = [
            # Common content containers
            "main", "article", "div.content", "div.main", "div#content", "div#main",
            # Documentation-specific selectors
            "div.documentation", "div.docs", "div.doc-content", 
            # Common CMS selectors
            "div.entry-content", "div.post-content",
            # Fallback to body
            "body"
        ]
        
        for selector in selectors:
            try:
                element = soup.select_one(selector)
                if element and len(element.get_text(strip=True)) > 100:  # Ensure it has substantial content
                    self.logger.debug(f"Found main content using selector: {selector}")
                    return element
            except Exception:
                continue
        
        # If no suitable container found, try to find the largest text block
        paragraphs = soup.find_all('p')
        if paragraphs:
            # Find the div containing the most paragraphs
            parent_counts = {}
            for p in paragraphs:
                parent = p.parent
                if parent:
                    parent_id = id(parent)
                    parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
            
            if parent_counts:
                max_parent_id = max(parent_counts, key=parent_counts.get)
                for p in paragraphs:
                    if id(p.parent) == max_parent_id:
                        return p.parent
        
        # If still no content found, just return the body
        body = soup.find('body')
        if body:
            return body
            
        self.logger.warning("No main content container found")
        return None
    
    async def _extract_title(self, soup: BeautifulSoup, main_content: BeautifulSoup) -> str:
        """Extract title from the document."""
        # Try to find title in main content first
        title = main_content.find("h1")
        if title:
            return title.get_text().strip()
        
        # Try document title
        if soup.title:
            return soup.title.string.strip()
        
        # Fallback
        return "Untitled Document"
    
    async def _extract_headers(self, main_content: BeautifulSoup) -> List[str]:
        """Extract headers from the document."""
        headers = main_content.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        return [h.get_text().strip() for h in headers]
    
    def _detect_code_language(self, element: BeautifulSoup) -> str:
        """Detect the language of a code block."""
        # Check for class-based language indicators
        if element.get("class"):
            for cls in element.get("class"):
                if cls.startswith("language-"):
                    return cls.replace("language-", "")
        
        # Check for common language indicators in the content
        code = element.get_text().lower()
        if "def " in code and "return" in code:
            return "python"
        elif "{" in code and "}" in code and ("function" in code or "var " in code or "const " in code):
            return "javascript"
        elif "<html" in code or "<div" in code:
            return "html"
        elif "select " in code and "from " in code:
            return "sql"
        
        # Default
        return ""
    
    def _process_table(self, table: BeautifulSoup) -> str:
        """Process a table element into markdown format."""
        rows = table.find_all("tr")
        if not rows:
            return ""
        
        table_content = []
        
        # Process header row
        headers = rows[0].find_all(["th", "td"])
        if headers:
            table_content.append("| " + " | ".join(h.get_text().strip() for h in headers) + " |")
            table_content.append("|" + "|".join(["---" for _ in headers]) + "|")
        
        # Process data rows
        for row in rows[1:]:
            cols = row.find_all("td")
            if cols:
                table_content.append("| " + " | ".join(c.get_text().strip() for c in cols) + " |")
        
        return "\n".join(table_content)
    
    async def _extract_content(self, main_content: BeautifulSoup) -> str:
        """Extract content with improved formatting preservation."""
        if not main_content:
            return ""
            
        content_parts = []
        
        # Process headings first to maintain hierarchy
        headings = main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        heading_elements = {h: h.get_text().strip() for h in headings}
        
        # Process elements in document order
        for element in main_content.descendants:
            # Skip already processed headings
            if element in heading_elements:
                heading_level = int(element.name[1])
                content_parts.append(f"{'#' * heading_level} {heading_elements[element]}\n\n")
                continue
                
            if element.name == "pre":
                # Handle code blocks
                code = element.get_text().strip()
                lang = self._detect_code_language(element)
                content_parts.append(f"```{lang}\n{code}\n```\n\n")
                
            elif element.name == "code" and element.parent.name != "pre":
                # Inline code
                code = element.get_text().strip()
                if len(code.split("\n")) > 1:
                    content_parts.append(f"```\n{code}\n```\n\n")
                else:
                    content_parts.append(f"`{code}`")
                    
            elif element.name == "p":
                text = element.get_text().strip()
                if text:
                    content_parts.append(f"{text}\n\n")
                    
            elif element.name == "li":
                # Only process li elements directly, not their children
                if element.parent and element.parent.name in ["ul", "ol"]:
                    prefix = "* " if element.parent.name == "ul" else f"{list(element.parent.children).index(element) + 1}. "
                    text = element.get_text().strip()
                    if text:
                        content_parts.append(f"{prefix}{text}\n")
                    
            elif element.name == "table":
                table_content = self._process_table(element)
                if table_content:
                    content_parts.append(table_content + "\n\n")
                    
            elif element.name == "a" and not any(p in content_parts for p in element.parents):
                href = element.get("href")
                text = element.get_text().strip()
                if href and text:
                    content_parts.append(f"[{text}]({href})")
                    
            elif element.name == "img":
                src = element.get("src")
                alt = element.get("alt", "")
                if src:
                    content_parts.append(f"![{alt}]({src})\n\n")
        
        return "".join(content_parts)
    
    async def _extract_links(self, soup: BeautifulSoup, base_url: str, allowed_domain: str) -> Set[str]:
        """Extract and filter links with improved handling."""
        links = set()
        
        for a in soup.find_all("a", href=True):
            href = a.get("href")
            
            # Skip empty links
            if not href:
                continue
                
            # Skip fragment identifiers and javascript links
            if href.startswith("#") or href.startswith("javascript:"):
                continue
                
            # Skip mail links
            if href.startswith("mailto:"):
                continue
                
            # Normalize URL
            try:
                full_url = urljoin(base_url, href)
                parsed_url = urlparse(full_url)
                
                # Skip fragments and queries
                full_url = parsed_url._replace(fragment="").geturl()
                if "?" in full_url:
                    full_url = full_url.split("?")[0]
                    
                # Validate domain
                if allowed_domain not in parsed_url.netloc:
                    continue
                    
                # Skip common non-documentation paths
                skip_patterns = [
                    "/login", "/logout", "/signup", "/register", 
                    "/account", "/profile", "/settings",
                    "/search", "/download", "/print",
                    "/share", "/comment", "/feedback"
                ]
                if any(pattern in parsed_url.path.lower() for pattern in skip_patterns):
                    continue
                    
                # Skip common file extensions
                skip_extensions = [".pdf", ".zip", ".tar", ".gz", ".jpg", ".jpeg", ".png", ".gif", ".svg"]
                if any(parsed_url.path.lower().endswith(ext) for ext in skip_extensions):
                    continue
                    
                links.add(full_url)
                
            except Exception as e:
                self.logger.debug(f"Error processing link {href}: {str(e)}")
                continue
                
        return links
    
    async def _url_worker(self):
        """Worker to process URLs from the queue."""
        self.logger.debug("URL worker started")
        
        # Диагностическая информация о состоянии
        self.logger.debug(f"URL worker state - processing: {self.processing}, queue size: {self.url_queue.size()}")
        
        try:
            while self.processing:
                try:
                    # Проверяем, есть ли элементы в очереди перед тем, как пытаться получить URL
                    if self.url_queue.size() == 0:
                        self.logger.debug("URL queue is empty, waiting...")
                        await asyncio.sleep(1)  # Ждем немного и проверяем снова вместо блокировки
                        continue
                    
                    # Получаем URL из очереди с таймаутом
                    try:
                        url_data = await asyncio.wait_for(self.url_queue.get(), timeout=5.0)
                        if url_data is None:
                            self.logger.debug("Received None in URL queue, worker exiting")
                            self.url_queue.task_done()
                            break
                        self.logger.debug(f"Got URL from queue: {url_data[0] if url_data else None}")
                    except asyncio.TimeoutError:
                        # Ничего нет в очереди, продолжаем ожидание
                        self.logger.debug("Timeout waiting for URL from queue, continuing...")
                        continue
                    
                    # Обрабатываем URL
                    try:
                        url, domain, path_parts = url_data
                        await self._process_url_safe(url, domain, path_parts)
                    except Exception as e:
                        self.logger.error(f"Error processing URL {url_data[0] if url_data else 'unknown'}: {str(e)}", exc_info=True)
                        if url_data and url_data[0]:
                            self.failed_urls.add(url_data[0])
                    finally:
                        # Всегда отмечаем задачу как выполненную, даже при ошибке
                        self.url_queue.task_done()
                        self.logger.debug(f"Marked URL task as done: {url_data[0] if url_data else None}")
                        
                except asyncio.CancelledError:
                    self.logger.debug("URL worker received cancellation signal")
                    raise
                except Exception as e:
                    self.logger.error(f"Unexpected error in URL worker: {str(e)}", exc_info=True)
                    # Продолжаем цикл, чтобы воркер оставался активным даже при ошибке
                    await asyncio.sleep(1)  # Небольшая пауза перед продолжением
                    continue
        except asyncio.CancelledError:
            self.logger.debug("URL worker cancelled")
        except Exception as e:
            self.logger.error(f"Fatal error in URL worker: {str(e)}", exc_info=True)
        self.logger.debug("URL worker exited")
    
    async def _process_url_safe(self, url: str, allowed_domain: str, path_parts: Optional[List[str]] = None):
        """Process URL with proper exception handling and recovery."""
        if url in self.visited_urls or not self.processing:
            return None
            
        self.visited_urls.add(url)
        
        try:
            # Validate URL
            if not URLValidator.validate_url(url):
                self.logger.warning(f"Invalid URL format: {url}")
                return None
                
            # Check domain
            if not URLValidator.is_same_domain(url, allowed_domain):
                self.logger.warning(f"URL not in allowed domain: {url}")
                return None
                
            # Apply rate limiting
            domain_limiter = self._get_domain_limiter(allowed_domain)
            
            # Set timeout for URL processing
            try:
                async with asyncio.timeout(self.config.processing_timeout // 2):  # Use half of total timeout
                    async with domain_limiter, self.semaphore:
                        # Add a small delay between requests to be polite
                        await asyncio.sleep(random.uniform(0.5, 1.5))
                        
                        # Fetch content
                        content = await self._make_request(url)
                        if not content:
                            return None
                        
                        # Process content
                        result = await self._process_html_content(content, url, allowed_domain, path_parts)
                        if not result:
                            return None
                        
                        doc_data, soup = result
                        
                        # Create a unique key for this document
                        doc_key = f"{doc_data['domain']}/{'/'.join(doc_data['path_parts'])}"
                        
                        # Only add to queue if we haven't processed this document before
                        if doc_key not in self._processed_docs:
                            self._processed_docs.add(doc_key)
                            await self.markdown_queue.put(doc_data)
                            self.markdown_items_added += 1  # Track items added to markdown queue
                            self.logger.debug(f"Added document to markdown queue for URL: {url}")
                        else:
                            self.logger.debug(f"Skipping duplicate document: {doc_key}")
                        
                        # Process links
                        links = await self._extract_links(soup, url, allowed_domain)
                        for link in links:
                            if self.processing and link not in self.visited_urls:
                                await self.url_queue.add(link, allowed_domain, None)
                        
                        # Update progress
                        self.progress.update(True)
                        
                        return doc_data
            except asyncio.TimeoutError:
                self.logger.error(f"Processing timed out for URL: {url}")
                self.failed_urls.add(url)
                self.progress.update(False)
                return None
        except Exception as e:
            self.logger.error(f"Error processing {url}: {str(e)}", exc_info=True)
            self.failed_urls.add(url)
            self.progress.update(False)
            return None
    
    async def _markdown_worker(self):
        """Worker to process markdown generation."""
        self.logger.debug("Markdown worker started")
        
        # Диагностическая информация о состоянии
        self.logger.debug(f"Markdown worker state - processing: {self.processing}, queue size: {self.markdown_queue.qsize()}")
        
        try:
            while self.processing:
                try:
                    # Check if there are items in the queue
                    if self.markdown_queue.qsize() == 0:
                        self.logger.debug("Markdown queue is empty, waiting...")
                        await asyncio.sleep(1)
                        continue
                    
                    # Get document from queue with timeout
                    try:
                        doc_data = await asyncio.wait_for(self.markdown_queue.get(), timeout=5.0)
                        if doc_data is None:
                            self.logger.debug("Received None in markdown queue, worker exiting")
                            self.markdown_queue.task_done()
                            break
                        self.logger.debug(f"Got document from markdown queue for URL: {doc_data.get('url', 'unknown')}")
                    except asyncio.TimeoutError:
                        # No document available, continue waiting
                        self.logger.debug("Timeout waiting for document from markdown queue, continuing...")
                        continue
                    
                    # Process the document
                    try:
                        # Generate markdown
                        markdown_content = await self.generate_markdown(doc_data)
                        
                        # Save markdown
                        await self.save_markdown(
                            doc_data["title"],
                            markdown_content,
                            doc_data["path_parts"],
                            doc_data["domain"],
                        )
                        
                        # Update tracking counters
                        self.markdown_items_processed += 1
                        
                        # Update last activity time
                        self.last_activity_time = time.time()
                        
                        self.logger.debug(f"Successfully processed markdown for: {doc_data.get('title', 'unknown')}")
                    except Exception as e:
                        self.logger.error(f"Error processing markdown for {doc_data.get('title', 'unknown')}: {str(e)}", exc_info=True)
                    finally:
                        # Always mark the task as done, even if processing failed
                        self.markdown_queue.task_done()
                        self.logger.debug(f"Marked markdown task as done: {doc_data.get('title', 'unknown')}")
                        
                except asyncio.CancelledError:
                    self.logger.debug("Markdown worker received cancellation signal")
                    raise
                except Exception as e:
                    self.logger.error(f"Unexpected error in markdown worker: {str(e)}", exc_info=True)
                    # Continue the loop to keep the worker alive
                    await asyncio.sleep(1)  # Небольшая пауза перед продолжением
                    continue
        except asyncio.CancelledError:
            self.logger.debug("Markdown worker cancelled")
        except Exception as e:
            self.logger.error(f"Fatal error in markdown worker: {str(e)}", exc_info=True)
        self.logger.debug("Markdown worker exited")
    
    async def generate_markdown(self, doc_data: Dict) -> str:
        """Generate well-formatted markdown content."""
        # Get the raw content
        content = doc_data["content"]
        title = doc_data["title"]
        headers = doc_data["headers"]
        url = doc_data["url"]
        
        # Clean up the content
        content = re.sub(r"```(\w*)\s*\n\s*```", "", content)  # Remove empty code blocks
        
        # Format JSON blocks nicely
        def format_json_blocks(match):
            try:
                json_str = match.group(1)
                parsed = json.loads(json_str)
                return f"```json\n{json.dumps(parsed, indent=2)}\n```"
            except:
                return match.group(0)
        
        content = re.sub(
            r"```json\n(.*?)\n```", format_json_blocks, content, flags=re.DOTALL
        )
        
        # Create markdown structure
        markdown_lines = []
        
        # Add title
        markdown_lines.append(f"# {title}")
        markdown_lines.append("")
        
        # Generate TOC if there are headers
        if headers:
            markdown_lines.append("## Table of Contents")
            
            # Create a mapping of header to its sanitized anchor
            toc_entries = []
            for header in headers:
                # Create GitHub-style anchor: lowercase, spaces to dashes, remove punctuation
                anchor = re.sub(r'[^\w\s-]', '', header.lower())
                anchor = re.sub(r'[\s]+', '-', anchor).strip('-')
                toc_entries.append(f"- [{header}](#{anchor})")
            
            markdown_lines.extend(toc_entries)
            markdown_lines.append("")
        
        # Add main content
        markdown_lines.append(content)
        
        # Add reference link
        markdown_lines.append(f"\n\n[Original Documentation]({url})")
        
        return "\n".join(markdown_lines)
    
    async def save_markdown(self, title: str, content: str, path_parts: List[str], domain: str):
        """Save markdown content to file."""
        if not title:
            title = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create directory structure
        doc_dir = await self._create_directory_structure(domain, path_parts[:-1])
        doc_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        
        safe_title = self._sanitize_filename(title)
        filename = doc_dir / f"{safe_title}.md"
        
        # Minimal cleanup to preserve content
        content = re.sub(r"(?i)\[Original Documentation\].*?\n*", "", content)
        content = re.sub(r"Was this page helpful\?.*?$", "", content, flags=re.DOTALL)
        content = re.sub(r"^```markdown\s*|\s*```\s*$", "", content)  # Remove outer markdown code blocks
        content = "\n".join(line for line in content.splitlines() if line.strip())
        
        # Add frontmatter
        final_content = f"""---
title: {title}
date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
path: {'/'.join(path_parts)}
domain: {domain}
---

{content}

[Original Documentation](https://{domain}/{'/'.join(path_parts)})
"""
        
        # Save to file
        try:
            async with aiofiles.open(filename, "w", encoding="utf-8") as f:
                await f.write(final_content)
                self.logger.info(f"Saved markdown file: {filename}")
        except Exception as e:
            self.logger.error(f"Error saving file {filename}: {str(e)}", exc_info=True)
    
    async def process_url(self, url: str, allowed_domain: str, path_parts: Optional[List[str]] = None):
        """Start URL processing by adding to queue."""
        self.processing = True
        self.progress.start_time = datetime.now()
        self.last_activity_time = time.time()
        
        # Reset counters
        self.markdown_items_added = 0
        self.markdown_items_processed = 0
        
        try:
            self.logger.info(f"Starting to process URL: {url}")
            
            # Validate URL
            if not URLValidator.validate_url(url):
                self.logger.error(f"Invalid URL: {url}")
                return
            
            # Обрабатываем исходный URL напрямую для быстрого старта
            self.logger.info(f"Processing initial URL directly: {url}")
            content = await self._make_request(url)
            if not content:
                self.logger.error(f"Failed to fetch initial content from {url}")
                self.failed_urls.add(url)
                return
                
            result = await self._process_html_content(content, url, allowed_domain, path_parts)
            if not result:
                self.logger.error(f"Failed to process content from {url}")
                self.failed_urls.add(url)
                return
                
            doc_data, soup = result
            
            # Create a unique key for this document
            doc_key = f"{doc_data['domain']}/{'/'.join(doc_data['path_parts'])}"
            
            # Add to processed docs
            self._processed_docs.add(doc_key)
            
            # Добавляем в очередь markdown вместо прямой обработки
            self.logger.info(f"Adding initial document to markdown queue: {doc_data.get('title', 'unknown')}")
            await self.markdown_queue.put(doc_data)
            self.markdown_items_added += 1
            
            # Add links to the queue
            links = await self._extract_links(soup, url, allowed_domain)
            self.logger.info(f"Adding {len(links)} links to the queue")
            
            links_added = 0
            for link in links:
                if link not in self.visited_urls:
                    await self.url_queue.add(link, allowed_domain, None)
                    links_added += 1
            self.logger.info(f"Added {links_added} links to the queue")
                    
            self.visited_urls.add(url)
            self.progress.update(True)
            
            # Проверяем статус воркеров
            active_url_workers = sum(1 for w in self.url_workers.workers if not w.done())
            active_md_workers = sum(1 for w in self.markdown_workers.workers if not w.done())
            self.logger.info(f"Worker status - URL: {active_url_workers}/{len(self.url_workers.workers)}, Markdown: {active_md_workers}/{len(self.markdown_workers.workers)}")
            
            # Прекращаем обработку, если нет активных воркеров
            if active_url_workers == 0 and active_md_workers == 0:
                self.logger.error("No workers are running, cannot continue processing")
                return
            
            # Мониторим процесс обработки с таймаутом
            timeout = self.config.processing_timeout
            self.logger.info(f"Waiting up to {timeout} seconds for processing to complete")
            
            # Мониторим статус воркеров и очередей
            start_time = time.time()
            last_status_time = start_time
            while time.time() - start_time < timeout and self.processing:
                current_time = time.time()
                
                # Логируем статус каждые 10 секунд
                if current_time - last_status_time >= 10:
                    url_queue_size = self.url_queue.size()
                    markdown_queue_size = self.markdown_queue.qsize()
                    active_url_workers = sum(1 for w in self.url_workers.workers if not w.done())
                    active_md_workers = sum(1 for w in self.markdown_workers.workers if not w.done())
                    
                    self.logger.info(
                        f"Progress - URLs: {len(self.visited_urls)}, Failed: {len(self.failed_urls)}, "
                        f"Queue sizes - URL: {url_queue_size}, Markdown: {markdown_queue_size}, "
                        f"Workers - URL: {active_url_workers}/{len(self.url_workers.workers)}, "
                        f"Markdown: {active_md_workers}/{len(self.markdown_workers.workers)}"
                    )
                    
                    # Если обе очереди пусты и мы обработали все элементы, завершаем обработку
                    if url_queue_size == 0 and markdown_queue_size == 0 and len(self.visited_urls) > 1:
                        self.logger.info("Both queues empty, checking if processing is complete")
                        if self.markdown_items_processed >= self.markdown_items_added:
                            self.logger.info("All items processed, finishing")
                            break
                    
                    # Перезапускаем отказавшие воркеры при необходимости
                    if active_url_workers < len(self.url_workers.workers) or active_md_workers < len(self.markdown_workers.workers):
                        self.logger.warning("Some workers have stopped, attempting to restart")
                        if active_url_workers < len(self.url_workers.workers):
                            await self.url_workers.stop()
                            await self.url_workers.start()
                        if active_md_workers < len(self.markdown_workers.workers):
                            await self.markdown_workers.stop()
                            await self.markdown_workers.start()
                    
                    last_status_time = current_time
                
                await asyncio.sleep(1)
            
            # Итоговый отчет о статусе
            self.logger.info(
                f"Final status - URLs visited: {len(self.visited_urls)}, Failed: {len(self.failed_urls)}, "
                f"Markdown items added: {self.markdown_items_added}, Processed: {self.markdown_items_processed}"
            )
            
        except Exception as e:
            self.logger.error(f"Error in process_url: {str(e)}", exc_info=True)
        finally:
            self.processing = False
            
            # Cancel status monitoring
            if self.status_task and not self.status_task.done():
                self.status_task.cancel()
                try:
                    await self.status_task
                except asyncio.CancelledError:
                    pass
    
    async def generate_combined_docs(self):
        """Generate combined documentation for all scraped content."""
        try:
            self.logger.info("Starting to generate combined documentation")
            
            # Group docs by domain
            docs_by_domain = {}
            for root, _, files in os.walk(self.sites_dir):
                for file in files:
                    if file.endswith(".md"):
                        file_path = Path(root) / file
                        try:
                            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                                content = await f.read()
                                domain = None
                                # Extract domain from frontmatter
                                for line in content.split("\n"):
                                    if line.startswith("domain: "):
                                        domain = line.split("domain: ")[1].strip()
                                        break
                                if domain:
                                    if domain not in docs_by_domain:
                                        docs_by_domain[domain] = []
                                    docs_by_domain[domain].append(
                                        {
                                            "path": file_path,
                                            "content": content,
                                            "title": file_path.stem,
                                        }
                                    )
                        except Exception as e:
                            self.logger.error(f"Error reading file {file_path}: {str(e)}", exc_info=True)
                            continue
            
            # Generate combined docs for each domain
            for domain, docs in docs_by_domain.items():
                try:
                    self.logger.info(f"Generating combined documentation for domain: {domain}")
                    # Sort docs by path for consistent ordering
                    docs.sort(key=lambda x: str(x["path"]))
                    
                    combined_content = await self._generate_domain_content(domain, docs)
                    
                    # Write combined file
                    safe_domain = self._sanitize_filename(domain)
                    combined_file = self.combined_dir / f"{safe_domain}_complete.md"
                    async with aiofiles.open(combined_file, "w", encoding="utf-8") as f:
                        await f.write(combined_content)
                    self.logger.info(f"Generated combined documentation: {combined_file}")
                except Exception as e:
                    self.logger.error(f"Error generating combined docs for {domain}: {str(e)}", exc_info=True)
                    continue
            
            # Generate master index
            try:
                await self._generate_master_index(docs_by_domain)
                self.logger.info("Generated master index")
            except Exception as e:
                self.logger.error(f"Error generating master index: {str(e)}", exc_info=True)
            
        except Exception as e:
            self.logger.error(f"Error in generate_combined_docs: {str(e)}", exc_info=True)
    
    async def _generate_domain_content(self, domain: str, docs: List[Dict]) -> str:
        """Generate combined content for a domain."""
        combined_content = f"""---
title: {domain} Complete Documentation
date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
domain: {domain}
---

# {domain} Documentation

## Table of Contents\n"""
        
        # Add table of contents
        for doc in docs:
            title = doc["title"].replace("_", " ").title()
            combined_content += f"- [{title}](#{self._sanitize_filename(title)})\n"
        
        # Add each document's content
        for doc in docs:
            combined_content += f"\n\n{doc['content']}"
        
        return combined_content
    
    async def _generate_master_index(self, docs_by_domain: Dict[str, List[Dict]]):
        """Generate master index file."""
        master_index = ["# Documentation Index\n\n"]
        master_index.extend(
            f"- [{domain}]({self._sanitize_filename(domain)}_complete.md)"
            for domain in sorted(docs_by_domain.keys())
        )
        
        index_file = self.combined_dir / "index.md"
        async with aiofiles.open(index_file, "w", encoding="utf-8") as f:
            await f.write("\n".join(master_index))
        self.logger.debug(f"Generated master index: {index_file}")
    
    async def cleanup(self):
        """Clean up resources."""
        self.logger.info("Cleaning up resources...")
        
        # Stop processing
        self.processing = False
        
        # Cancel status monitoring
        if self.status_task and not self.status_task.done():
            self.status_task.cancel()
            try:
                await self.status_task
            except asyncio.CancelledError:
                pass
        
        # Stop workers
        await self.url_workers.stop()
        
        # Send poison pills to markdown workers
        for _ in range(len(self.markdown_workers.workers)):
            await self.markdown_queue.put(None)
        
        await self.markdown_workers.stop()
        
        # Clean up resources
        await self.resource_manager.cleanup()
        
        # Log results
        self.logger.info(f"Final stats: {self.progress.processed} URLs processed, {len(self.failed_urls)} failed")
        self.logger.info(f"Markdown stats: {self.markdown_items_added} items added, {self.markdown_items_processed} items processed")
        
        if self.failed_urls:
            self.logger.warning(f"Failed URLs: {len(self.failed_urls)}")
            for url in list(self.failed_urls)[:10]:  # Show only first 10 failed URLs
                self.logger.warning(f"Failed: {url}")
            if len(self.failed_urls) > 10:
                self.logger.warning(f"... and {len(self.failed_urls) - 10} more")

# Main function
async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Documentation site scraper")
    parser.add_argument("--url", required=True, help="Starting URL to scrape")
    parser.add_argument("--output-dir", default="docs_output", help="Output directory")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Maximum concurrent requests")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--log-file", help="Log file path")
    parser.add_argument("--timeout", type=int, default=300, help="Processing timeout in seconds")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with extra logging")
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure log directory exists
    if args.log_file:
        log_dir = Path(args.log_file).parent
    else:
        log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = Config(
        base_dir=args.output_dir,
        max_concurrent=args.max_concurrent,
        log_level="DEBUG" if args.debug else args.log_level,
        log_file=args.log_file or f"logs/docparser_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        processing_timeout=args.timeout,
        domain_rate_limit=3,  # More conservative limit
        request_delay=1.0,    # 1 second delay between requests to same domain
    )
    
    scraper = None
    try:
        # Create and initialize scraper
        scraper = DocsiteToMD(config=config)
        await scraper.initialize()
        
        # Process URL
        domain = urlparse(args.url).netloc
        await scraper.process_url(args.url, domain)
        
        # Generate combined docs
        await scraper.generate_combined_docs()
        
        scraper.logger.info("Documentation conversion completed successfully")
    except Exception as e:
        if scraper:
            scraper.logger.error(f"Fatal error: {str(e)}", exc_info=True)
        else:
            print(f"Fatal error before scraper initialization: {str(e)}")
        raise
    finally:
        if scraper:
            await scraper.cleanup()
            scraper.logger.info("Cleanup completed")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Process interrupted by user")
    except Exception as e:
        print(f"Process failed: {str(e)}")
        sys.exit(1)
