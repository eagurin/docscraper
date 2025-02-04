# Standard library imports
import argparse
import asyncio
import json
import os
import re
import sys
from asyncio import Semaphore
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set
from urllib.parse import urljoin, urlparse

# Third-party imports
import aiofiles
import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",  # Only show INFO and above in console
    filter=lambda record: record["level"].name
    in ["INFO", "WARNING", "ERROR", "SUCCESS"],
)
logger.add(
    "/Users/laptop/dev/docscraper/logs/docparser_{time}.log",  # Use absolute path
    rotation="500 MB",
    retention="10 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",  # Keep all levels in file
    backtrace=True,
    diagnose=True,
    enqueue=True,  # Enable thread-safe logging
    filter=lambda record: True,  # Log everything to file
)

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
MAX_CONCURRENT = 3  # Limit concurrent operations


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=30))
async def retry_request(session, url: str):
    logger.debug(f"Making request to: {url}")  # Keep as DEBUG
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with session.get(url, timeout=timeout, allow_redirects=True) as response:
            logger.debug(f"Response status for {url}: {response.status}")
            if response.status == 429:  # Rate limit
                retry_after = int(response.headers.get("Retry-After", 30))
                await asyncio.sleep(retry_after)
                raise aiohttp.ClientError("Rate limited")
            response.raise_for_status()  # Raise exception for 4xx/5xx status
            content = await response.text()
            logger.debug(f"Retrieved content from {url} (length: {len(content)})")
            return content
    except aiohttp.ClientError as e:
        logger.error(f"Request failed for {url}: {str(e)}")
        if isinstance(e, aiohttp.ClientConnectorError):
            await asyncio.sleep(5)  # Wait before retry on connection error
        raise
    except asyncio.TimeoutError:
        logger.error(f"Request timed out for {url}")
        await asyncio.sleep(5)  # Wait before retry on timeout
        raise
    except Exception as e:
        logger.error(f"Unexpected error for {url}: {str(e)}")
        raise


class DocsiteToMD:
    def __init__(
        self, base_dir="docs_output", batch_size=10, rate_limit=5, max_concurrent=3
    ):
        self.client = AsyncOpenAI()
        self.base_dir = Path(base_dir)
        self.visited_urls = set()
        self.collected_docs = []
        self.semaphore = Semaphore(max_concurrent)
        self.rate_limit = Semaphore(rate_limit)
        self.openai_semaphore = Semaphore(5)
        self.max_concurrent = max_concurrent
        self.domain_limiters = {}
        self.session = None
        self.url_queue = asyncio.Queue()
        self.markdown_queue = asyncio.Queue()
        self.workers = []
        self.markdown_workers = []
        self.batch_size = batch_size
        self.processing = True
        self.processed_count = 0
        self.total_urls = 0
        self.failed_urls = set()
        self.gc_threshold = 1000  # GC after processing this many URLs
        self._processed_docs = set()  # Initialize set to track processed documents
        self.conn = aiohttp.TCPConnector(
            limit=max_concurrent,  # Use instance max_concurrent
            ttl_dns_cache=300,
            use_dns_cache=True,
            limit_per_host=5,  # Increased from 2
        )
        logger.info("DocsiteToMD initialized")

    async def initialize(self):
        """Initialize directories and HTTP session asynchronously"""
        logger.info("Starting initialization")
        try:
            await self._ensure_directory(self.base_dir)
            self.sites_dir = self.base_dir / "sites"
            await self._ensure_directory(self.sites_dir)
            self.combined_dir = self.base_dir / "combined"
            await self._ensure_directory(self.combined_dir)

            # Initialize aiohttp session with optimized settings
            timeout = aiohttp.ClientTimeout(
                total=30, connect=10, sock_read=10, sock_connect=10
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=self.conn,
                headers={
                    "Connection": "keep-alive",
                    "User-Agent": "Mozilla/5.0 (compatible; DocScraper/1.0)",
                },
            )

            # Start workers with optimal counts
            worker_count = min(self.max_concurrent, 20)  # Cap at 20 workers
            markdown_worker_count = min(
                worker_count // 2, 10
            )  # Cap at 10 markdown workers

            self.workers = [
                asyncio.create_task(self._worker()) for _ in range(worker_count)
            ]
            self.markdown_workers = [
                asyncio.create_task(self._markdown_worker())
                for _ in range(markdown_worker_count)
            ]

            await asyncio.sleep(0.1)
            logger.info(
                f"Started {len(self.workers)} URL workers and {len(self.markdown_workers)} markdown workers"
            )
            logger.success("Initialization completed")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    async def _ensure_directory(self, path: Path):
        """Ensure directory exists asynchronously"""
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

    async def _get_domain_dir(self, domain: str) -> Path:
        """Create and return domain-specific directory asynchronously"""
        domain_dir = self.sites_dir / self._sanitize_filename(domain)
        await self._ensure_directory(domain_dir)
        return domain_dir

    async def _create_directory_structure(
        self, domain: str, path_parts: List[str]
    ) -> Path:
        current_path = await self._get_domain_dir(domain)
        for part in path_parts:
            current_path = current_path / self._sanitize_filename(part)
            await self._ensure_directory(current_path)
        return current_path

    def _sanitize_filename(self, name: str) -> str:
        return (
            "".join(c if c.isalnum() or c in "-_ " else "_" for c in name)
            .lower()
            .replace(" ", "_")
        )

    async def _extract_title(self, soup, main_content):
        title = main_content.find("h1")
        if title:
            return title.get_text().strip()
        return soup.title.string if soup.title else "Untitled Document"

    async def _extract_content(self, main_content):
        content = []
        for element in main_content.descendants:
            if element.name == "pre":
                # Handle code blocks
                code = element.get_text().strip()
                lang = element.get("class", [""])[0] if element.get("class") else ""
                if lang.startswith("language-"):
                    lang = lang.replace("language-", "")
                elif "json" in str(element).lower():
                    lang = "json"
                elif any(
                    keyword in code.lower()
                    for keyword in ["function", "class", "def", "return"]
                ):
                    lang = "python"
                content.append(f"```{lang}\n{code}\n```\n")
            elif element.name == "code":
                # Inline code
                code = element.get_text().strip()
                if len(code.split("\n")) > 1:
                    content.append(f"```\n{code}\n```\n")
                else:
                    content.append(f"`{code}`")
            elif element.name in ["p", "li", "h1", "h2", "h3", "h4", "h5", "h6"]:
                text = element.get_text().strip()
                if text and not any(text in c for c in content):
                    content.append(text)
            elif element.name == "table":
                # Handle tables
                rows = element.find_all("tr")
                if rows:
                    table_content = []
                    headers = rows[0].find_all(["th", "td"])
                    if headers:
                        table_content.append(
                            "| "
                            + " | ".join(h.get_text().strip() for h in headers)
                            + " |"
                        )
                        table_content.append(
                            "|" + "|".join(["---" for _ in headers]) + "|"
                        )

                    for row in rows[1:]:
                        cols = row.find_all("td")
                        if cols:
                            table_content.append(
                                "| "
                                + " | ".join(c.get_text().strip() for c in cols)
                                + " |"
                            )
                    content.extend(table_content)

        return "\n\n".join(content)

    async def _extract_headers(self, main_content):
        headers = main_content.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        return [h.get_text().strip() for h in headers]

    async def _process_content(
        self, content: str, url: str, allowed_domain: str, path_parts: List[str] = None
    ):
        """Process HTML content asynchronously"""
        try:
            logger.debug(f"Processing content for URL: {url}")

            # Process in chunks if content is large
            if len(content) > 1_000_000:  # 1MB
                chunks = [
                    content[i : i + 1_000_000]
                    for i in range(0, len(content), 1_000_000)
                ]
                soup = BeautifulSoup("", "html.parser")  # Create empty soup
                for chunk in chunks:
                    chunk_soup = BeautifulSoup(chunk, "html.parser")
                    soup.append(chunk_soup)
            else:
                soup = BeautifulSoup(content, "html.parser")

            try:
                # Try different content selectors
                main_content = (
                    soup.find("main")
                    or soup.find("article")
                    or soup.find("div", class_="content")
                    or soup.find("div", class_="main")
                    or soup.find("div", id="content")
                    or soup.find("div", id="main")
                    or soup.find("body")  # Fallback to body if no other container found
                )

                if not main_content:
                    logger.error(f"No main content found for: {url}")
                    return None

                logger.debug(f"Found main content for: {url}")

                parsed_url = urlparse(url)
                url_path = parsed_url.path.strip("/")
                current_path_parts = url_path.split("/") if url_path else ["index"]

                current_path_parts = [part for part in current_path_parts if part]
                if not current_path_parts:
                    current_path_parts = ["index"]

                if path_parts:
                    current_path_parts = path_parts + current_path_parts

                title = await self._extract_title(soup, main_content)
                content = await self._extract_content(main_content)
                headers = await self._extract_headers(main_content)

                logger.debug(
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

                return doc_data, main_content

            except Exception as e:
                logger.error(f"Error processing main content for {url}: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Error processing content for {url}: {str(e)}")
            return None

    BATCH_SIZE = 10

    async def _process_batch(self, urls: List[tuple]):
        """Process a batch of URLs concurrently with optimized handling"""
        try:
            tasks = []
            for i in range(0, len(urls), self.BATCH_SIZE):
                batch = urls[i : i + self.BATCH_SIZE]
                tasks.extend(
                    [
                        asyncio.create_task(
                            self._process_single_url(url, domain, path_parts)
                        )
                        for url, domain, path_parts in batch
                        if url not in self.visited_urls
                    ]
                )
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            # Silently handle batch processing cancellation
            pass
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")

    async def _worker(self):
        """Worker to process URLs from the queue"""
        logger.debug("URL worker started")
        try:
            while True:
                try:
                    url_data = await self.url_queue.get()
                    if url_data is None:
                        break

                    url, domain, path_parts = url_data
                    logger.debug(f"Worker processing URL: {url}")
                    result = await self._process_single_url(url, domain, path_parts)
                    if result:
                        logger.debug(f"Successfully processed URL: {url}")
                        await self._update_progress()

                    self.url_queue.task_done()
                except Exception as e:
                    logger.error(f"Worker error: {str(e)}")
                    if url_data and url_data[0]:
                        self.failed_urls.add(url_data[0])
                    self.url_queue.task_done()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Fatal worker error: {str(e)}")

    def _get_domain_limiter(self, domain: str) -> Semaphore:
        if domain not in self.domain_limiters:
            self.domain_limiters[domain] = Semaphore(
                5
            )  # 5 concurrent requests per domain
        return self.domain_limiters[domain]

    async def _update_progress(self):
        """Update and log progress"""
        self.processed_count += 1
        if self.processed_count % 10 == 0:  # Log every 10 URLs
            logger.info(
                f"Progress: {self.processed_count}/{self.total_urls} URLs processed"
            )
            if self.processed_count % self.gc_threshold == 0:
                import gc

                gc.collect()  # Force garbage collection periodically

    async def _process_single_url(
        self, url: str, allowed_domain: str, path_parts: List[str] = None
    ):
        """Process a single URL with optimized handling"""
        if url in self.visited_urls or not self.processing:
            logger.debug(f"Skipping URL (already visited or processing stopped): {url}")
            return

        self.visited_urls.add(
            url
        )  # Mark as visited before processing to prevent duplicates
        logger.debug(f"Processing URL: {url}")

        domain_limiter = self._get_domain_limiter(allowed_domain)
        try:
            async with domain_limiter, self.semaphore:  # Limit both domain and total concurrent requests
                content = await retry_request(self.session, url)
                if not content:
                    logger.warning(f"No content retrieved for URL: {url}")
                    return

                result = await self._process_content(
                    content, url, allowed_domain, path_parts
                )
                if not result:
                    logger.warning(f"Content processing failed for URL: {url}")
                    return

                doc_data, main_content = result
                doc_data["domain"] = allowed_domain

                # Create a unique key for this document
                doc_key = f"{doc_data['domain']}/{'/'.join(doc_data['path_parts'])}"
                if not hasattr(self, "_processed_docs"):
                    self._processed_docs = set()

                # Only add to queue if we haven't processed this document before
                if doc_key not in self._processed_docs:
                    self._processed_docs.add(doc_key)
                    await self.markdown_queue.put(doc_data)
                    logger.debug(f"Added document to markdown queue for URL: {url}")
                else:
                    logger.debug(f"Skipping duplicate document: {doc_key}")

                # Process links in parallel
                links = await self._extract_links(content, url, allowed_domain)
                for link in links:
                    if self.processing and link not in self.visited_urls:
                        logger.debug(f"Adding URL to queue: {link}")
                        await self.url_queue.put((link, allowed_domain, None))
                    else:
                        logger.debug(
                            f"URL already visited or processing stopped: {link}"
                        )

        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}", exc_info=True)

    async def _extract_links(
        self, content: str, base_url: str, allowed_domain: str
    ) -> Set[str]:
        """Extract and filter links from content"""
        soup = BeautifulSoup(content, "html.parser")
        links = {
            urljoin(base_url, a.get("href"))
            for a in soup.find_all("a", href=True)
            if allowed_domain in urljoin(base_url, a.get("href"))
            and "#" not in a.get("href")
            and "?" not in a.get("href")
            and a.get("href") not in self.visited_urls
        }
        return links

    async def _markdown_worker(self):
        """Worker to process markdown generation"""
        processed_paths = set()  # Track processed paths
        while True:
            try:
                doc_data = await self.markdown_queue.get()
                if doc_data is None:  # Poison pill
                    break

                try:
                    # Create a unique key for this document
                    doc_key = f"{doc_data['domain']}/{'/'.join(doc_data['path_parts'])}"

                    # Only process if we haven't seen this document before
                    if doc_key not in processed_paths:
                        processed_paths.add(doc_key)
                        async with self.openai_semaphore:
                            markdown_content = await self.generate_markdown(doc_data)
                            await self.save_markdown(
                                doc_data["title"],
                                markdown_content,
                                doc_data["path_parts"],
                                doc_data["domain"],
                            )
                    else:
                        logger.debug(f"Skipping duplicate document: {doc_key}")
                except Exception as e:
                    logger.error(
                        f"Error processing markdown for {doc_data.get('title', 'unknown')}: {str(e)}"
                    )
            except Exception as e:
                logger.error(f"Markdown worker error: {str(e)}")
            finally:
                self.markdown_queue.task_done()

    async def process_url(
        self, url: str, allowed_domain: str, path_parts: List[str] = None
    ):
        """Start URL processing by adding to queue"""
        self.processing = True
        try:
            logger.debug(f"Starting to process URL: {url}")
            content = await retry_request(self.session, url)
            if not content:
                logger.error(f"Failed to fetch initial content from {url}")
                self.failed_urls.add(url)
                return

            # Extract initial links to get total count
            try:
                soup = BeautifulSoup(content, "html.parser")
                initial_links = {
                    urljoin(url, a.get("href"))
                    for a in soup.find_all("a", href=True)
                    if allowed_domain in urljoin(url, a.get("href"))
                }
                self.total_urls = len(initial_links) + 1  # +1 for initial URL
                logger.debug(f"Found {self.total_urls} total URLs to process")
            except Exception as e:
                logger.error(f"Error extracting initial links: {str(e)}")
                self.total_urls = 1  # Just process the initial URL

            # Start URL processing
            await self.url_queue.put((url, allowed_domain, path_parts))

            try:
                # Wait for all processing to complete
                await asyncio.wait_for(
                    asyncio.gather(self.url_queue.join(), self.markdown_queue.join()),
                    timeout=300,  # 5 minute timeout
                )
                logger.info(f"Completed processing for {url}")
            except asyncio.TimeoutError:
                logger.error("Processing timed out after 5 minutes")
            except Exception as e:
                logger.error(f"Error waiting for queues: {str(e)}")

        except Exception as e:
            logger.error(f"Error in main processing: {str(e)}", exc_info=True)
        finally:
            self.processing = False
            # Clean up workers
            for worker in self.workers:
                worker.cancel()
            for _ in range(len(self.markdown_workers)):
                await self.markdown_queue.put(None)
            await asyncio.gather(
                *self.workers, *self.markdown_workers, return_exceptions=True
            )
            self.workers = []
            self.markdown_workers = []

    async def _generate_domain_content(self, domain: str, docs: List[Dict]) -> str:
        """Generate combined content for a domain"""
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
        """Generate master index file"""
        master_index = ["# Documentation Index\n\n"]
        master_index.extend(
            f"- [{domain}]({self._sanitize_filename(domain)}_complete.md)"
            for domain in sorted(docs_by_domain.keys())
        )

        index_file = self.combined_dir / "index.md"
        async with aiofiles.open(index_file, "w", encoding="utf-8") as f:
            await f.write("\n".join(master_index))
        logger.debug(f"Generated master index: {index_file}")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_markdown(self, doc_data: Dict) -> str:
        # First, clean up the content to ensure proper markdown formatting
        content = doc_data["content"]

        # Ensure code blocks are properly formatted
        content = re.sub(
            r"```(\w*)\s*\n\s*```", "", content
        )  # Remove empty code blocks

        # Format JSON more nicely
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

        prompt = f"""
		Convert this documentation into a well-structured markdown format:
		Title: {doc_data['title']}
		Content: {content}
		Headers: {doc_data['headers']}
		URL: {doc_data['url']}
		
		Create:
		1. A clear title
		2. Table of contents based on headers
		3. Properly structured content with headers
		4. Preserve all code blocks and their language annotations
		5. Preserve table formatting
		6. Original URL as reference
		
		Important: Maintain all existing markdown formatting, especially code blocks and tables.
		"""

        try:
            response = await self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating markdown: {e}", exc_info=True)
            return content

    async def save_markdown(
        self, title: str, content: str, path_parts: List[str], domain: str
    ):
        if not title:
            title = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        doc_dir = await self._create_directory_structure(domain, path_parts[:-1])
        safe_title = self._sanitize_filename(title)
        filename = doc_dir / f"{safe_title}.md"

        # Minimal cleanup to preserve content
        content = re.sub(r"(?i)\[Original Documentation\].*?\n*", "", content)
        content = re.sub(r"Was this page helpful\?.*?$", "", content, flags=re.DOTALL)
        content = re.sub(
            r"^```markdown\s*|\s*```\s*$", "", content
        )  # Remove outer markdown code blocks
        content = "\n".join(line for line in content.splitlines() if line.strip())

        final_content = f"""---
title: {title}
date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
path: {'/'.join(path_parts)}
domain: {domain}
---

{content}

[Original Documentation](https://{domain}/{'/'.join(path_parts)})
"""

        async with aiofiles.open(filename, "w", encoding="utf-8") as f:
            await f.write(final_content)
            logger.info(f"Saved markdown file: {filename}")

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        try:
            # Cancel all workers
            for worker in self.workers + self.markdown_workers:
                worker.cancel()

            # Close session
            if self.session:
                await self.session.close()

            # Log results
            logger.info(f"Final stats: {self.processed_count} URLs processed")
            if self.failed_urls:
                logger.warning(f"Failed URLs: {len(self.failed_urls)}")
                for url in self.failed_urls:
                    logger.warning(f"Failed: {url}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def generate_combined_docs(self):
        """Generate combined documentation for all scraped content"""
        try:
            logger.debug("Starting to generate combined documentation")

            # Group docs by domain
            docs_by_domain = {}
            for root, _, files in os.walk(self.sites_dir):
                for file in files:
                    if file.endswith(".md"):
                        file_path = Path(root) / file
                        try:
                            async with aiofiles.open(
                                file_path, "r", encoding="utf-8"
                            ) as f:
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
                            logger.error(f"Error reading file {file_path}: {str(e)}")
                            continue

            # Generate combined docs for each domain
            for domain, docs in docs_by_domain.items():
                try:
                    logger.debug(
                        f"Generating combined documentation for domain: {domain}"
                    )
                    # Sort docs by path for consistent ordering
                    docs.sort(key=lambda x: str(x["path"]))

                    combined_content = await self._generate_domain_content(domain, docs)

                    # Write combined file
                    safe_domain = self._sanitize_filename(domain)
                    combined_file = self.combined_dir / f"{safe_domain}_complete.md"
                    async with aiofiles.open(combined_file, "w", encoding="utf-8") as f:
                        await f.write(combined_content)
                    logger.debug(f"Generated combined documentation: {combined_file}")
                except Exception as e:
                    logger.error(
                        f"Error generating combined docs for {domain}: {str(e)}"
                    )
                    continue

            # Generate master index
            try:
                await self._generate_master_index(docs_by_domain)
            except Exception as e:
                logger.error(f"Error generating master index: {str(e)}")

        except Exception as e:
            logger.error(f"Error in generate_combined_docs: {str(e)}")


async def main():
    parser = argparse.ArgumentParser(description="Documentation site scraper")
    parser.add_argument("--url", required=True, help="Starting URL to scrape")
    parser.add_argument("--output-dir", default="docs_output", help="Output directory")
    parser.add_argument(
        "--max-concurrent", type=int, default=3, help="Maximum concurrent requests"
    )
    parser.add_argument(
        "--wait-time", type=float, default=1.0, help="Wait time between requests"
    )
    parser.add_argument("--model", help="OpenAI model name (default: from env)")
    args = parser.parse_args()

    global MODEL_NAME, MAX_CONCURRENT
    if args.model:
        MODEL_NAME = args.model
    MAX_CONCURRENT = args.max_concurrent

    scraper = None
    try:
        logger.info("Starting DocsiteToMD")
        scraper = DocsiteToMD(
            base_dir=args.output_dir,
            max_concurrent=args.max_concurrent,
            rate_limit=args.max_concurrent,
            batch_size=10,
        )
        await scraper.initialize()

        domain = urlparse(args.url).netloc
        await scraper.process_url(args.url, domain)

        # Wait for queues to be empty
        await scraper.url_queue.join()
        await scraper.markdown_queue.join()

        # Generate combined docs
        await scraper.generate_combined_docs()

        logger.success("Documentation conversion completed successfully")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        raise
    finally:
        if scraper:
            await scraper.cleanup()
            logger.info("Cleanup completed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Process failed: {str(e)}", exc_info=True)
        sys.exit(1)
