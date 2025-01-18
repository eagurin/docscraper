import aiofiles
import argparse
import asyncio
import os
import sys
import functools
from typing import List, Dict, Set, Coroutine
from tenacity import retry, stop_after_attempt, wait_exponential
from urllib.parse import urlparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from browser_use import BrowserConfig, Browser
from browser_use.browser.context import BrowserContextConfig
from openai import AsyncOpenAI
from bs4 import BeautifulSoup
import re
from asyncio import Semaphore
from loguru import logger

load_dotenv()

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
	sys.stdout,
	colorize=True,
	format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
	level="INFO"
)
logger.add(
	"logs/docparser_{time}.log",
	rotation="500 MB",
	retention="10 days",
	format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
	level="DEBUG",
	backtrace=True,
	diagnose=True
)

MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4')
MAX_CONCURRENT = 3  # Limit concurrent operations

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def retry_request(func, *args, **kwargs):
	return await func(*args, **kwargs)

BROWSER_CONFIG = BrowserConfig(
	headless=True,
	disable_security=True,
	new_context_config=BrowserContextConfig(
		wait_for_network_idle_page_load_time=10.0  # Increased from 3.0
	)
)

class DocsiteToMD:
	def __init__(self, base_dir='docs_output'):
		self.browser = None
		self.client = AsyncOpenAI()
		self.base_dir = Path(base_dir)
		self.visited_urls = set()
		self.collected_docs = []
		self.semaphore = Semaphore(MAX_CONCURRENT)
		logger.info("DocsiteToMD initialized")

	async def initialize(self):
		"""Initialize directories asynchronously"""
		logger.info("Starting initialization")
		await self._ensure_directory(self.base_dir)
		self.sites_dir = self.base_dir / 'sites'
		await self._ensure_directory(self.sites_dir)
		self.combined_dir = self.base_dir / 'combined'
		await self._ensure_directory(self.combined_dir)
		await self.init_browser()
		logger.success("Initialization completed")

	async def _ensure_directory(self, path: Path):
		"""Ensure directory exists asynchronously"""
		if not path.exists():
			path.mkdir(parents=True, exist_ok=True)

	async def _get_domain_dir(self, domain: str) -> Path:
		"""Create and return domain-specific directory asynchronously"""
		domain_dir = self.sites_dir / self._sanitize_filename(domain)
		await self._ensure_directory(domain_dir)
		return domain_dir

	async def _create_directory_structure(self, domain: str, path_parts: List[str]) -> Path:
		current_path = await self._get_domain_dir(domain)
		for part in path_parts:
			current_path = current_path / self._sanitize_filename(part)
			await self._ensure_directory(current_path)
		return current_path

	def _sanitize_filename(self, name: str) -> str:
		return ''.join(c if c.isalnum() or c in '-_ ' else '_' for c in name).lower().replace(' ', '_')

	async def init_browser(self):
		try:
			logger.info("Initializing browser...")
			self.browser = Browser(BROWSER_CONFIG)
			self.context = await self.browser.new_context()
			logger.success("Browser initialized successfully")
		except Exception as e:
			logger.error(f"Error initializing browser: {str(e)}")
			raise

	async def safe_navigate(self, url: str) -> bool:
		try:
			await retry_request(self.context.create_new_tab)
			await asyncio.sleep(2)  # Increased from 1
			await retry_request(self.context.navigate_to, url)
			await asyncio.sleep(5)  # Increased from 2
			return True
		except Exception as e:
			logger.error(f"Navigation error for {url}: {str(e)}", exc_info=True)
			return False

	async def process_url(self, url: str, allowed_domain: str, path_parts: List[str] = None):
		if url in self.visited_urls:
			return

		async with self.semaphore:
			logger.info(f"Processing URL: {url}")
			self.visited_urls.add(url)
			
			for attempt in range(3):  # Try up to 3 times
				try:
					success = await self.safe_navigate(url)
					if not success:
						raise Exception("Navigation failed")

					content = await retry_request(self.context.get_page_html)
					if not content:
						raise Exception("No content received")

					result = await self._process_content(content, url, allowed_domain, path_parts)
					if result:
						doc_data, main_content = result
						
						logger.info(f"Generating markdown for: {doc_data['title']}")
						markdown_content = await self.generate_markdown(doc_data)
						await self.save_markdown(doc_data['title'], markdown_content, doc_data['path_parts'], allowed_domain)
						self.collected_docs.append(doc_data)

						page = await self.context.get_current_page()
						links = await page.evaluate(f"""() => {{
							return Array.from(document.querySelectorAll('a[href]'))
								.map(a => a.href)
								.filter(href => href.includes('{allowed_domain}') && !href.includes('#'));
						}}""")
						
						new_links = [link for link in links if link not in self.visited_urls][:3]
						if new_links:
							logger.debug(f"Found {len(new_links)} new links to process")
							await asyncio.gather(
								*(self.process_url(link, allowed_domain, path_parts) for link in new_links)
							)
					break  # Success, exit retry loop

				except Exception as e:
					logger.error(f"Error processing {url} (attempt {attempt + 1}/3): {str(e)}")
					if attempt < 2:  # If not the last attempt
						await asyncio.sleep(5 * (attempt + 1))  # Exponential backoff
						continue
				finally:
					try:
						await self.context.close_current_tab()
					except:
						pass





	async def _extract_title(self, soup, main_content):
		title = main_content.find('h1')
		if title:
			return title.get_text().strip()
		return soup.title.string if soup.title else 'Untitled Document'

	async def _extract_content(self, main_content):
		elements = main_content.find_all(['p', 'li', 'pre', 'code'])
		return ' '.join(el.get_text().strip() for el in elements if el.get_text().strip())

	async def _extract_headers(self, main_content):
		headers = main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
		return [h.get_text().strip() for h in headers]

	async def _process_content(self, content: str, url: str, allowed_domain: str, path_parts: List[str] = None):
		"""Process HTML content asynchronously"""
		logger.debug(f"Processing content for URL: {url}")
		soup = BeautifulSoup(content, 'html.parser')
		main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
		
		if not main_content:
			logger.warning(f"No main content found for: {url}")
			return None
			
		logger.info(f"Found main content for: {url}")
		url_path = url.replace(f"https://{allowed_domain}", "").strip("/")
		current_path_parts = url_path.split("/") if url_path else ["index"]
		if path_parts:
			current_path_parts = path_parts + current_path_parts
			logger.debug(f"Path parts for {url}: {current_path_parts}")

		doc_data = {
			'url': url,
			'title': await self._extract_title(soup, main_content),
			'content': await self._extract_content(main_content),
			'headers': await self._extract_headers(main_content),
			'path_parts': current_path_parts
		}
		logger.debug(f"Extracted data for {url}: title='{doc_data['title']}', headers_count={len(doc_data['headers'])}")
		return doc_data, main_content

	async def generate_markdown(self, doc_data: Dict) -> str:
		prompt = f"""
		Convert this documentation into a well-structured markdown format:
		Title: {doc_data['title']}
		Content: {doc_data['content'][:4000]}
		Headers: {doc_data['headers']}
		URL: {doc_data['url']}
		
		Create:
		1. A clear title
		2. Table of contents based on headers
		3. Properly structured content with headers
		4. Code blocks where appropriate
		5. Original URL as reference
		"""
		
		try:
			response = await self.client.chat.completions.create(
				model=MODEL_NAME,
				messages=[{"role": "user", "content": prompt}],
				temperature=0.7,
				max_tokens=2000
			)
			return response.choices[0].message.content
		except Exception as e:
			logger.error(f"Error generating markdown: {e}", exc_info=True)
			return f"# {doc_data['title']}\n\n{doc_data['content']}\n\n[Source]({doc_data['url']})"


	async def save_markdown(self, title: str, content: str, path_parts: List[str], domain: str):
		if not title:
			title = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
		
		doc_dir = await self._create_directory_structure(domain, path_parts[:-1])
		safe_title = self._sanitize_filename(title)
		filename = doc_dir / f"{safe_title}.md"
		
		final_content = f"""---
title: {title}
date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
path: {'/'.join(path_parts)}
domain: {domain}
---

{content}
"""
		async with aiofiles.open(filename, 'w', encoding='utf-8') as f:
			await f.write(final_content)
		logger.info(f"Saved markdown file: {filename}")


	async def generate_combined_docs(self):
		if not self.collected_docs:
			return

		docs_by_domain = {}
		for doc in self.collected_docs:
			domain = doc['url'].split('//')[1].split('/')[0]
			if domain not in docs_by_domain:
				docs_by_domain[domain] = []
			docs_by_domain[domain].append(doc)

		# Process domains concurrently
		async def process_domain(domain: str, docs: List[Dict]):
			sorted_docs = sorted(docs, key=lambda x: '/'.join(x['path_parts']))
			toc = [f"# {domain} Documentation\n\n## Table of Contents\n"]
			content = []
			
			for doc in sorted_docs:
				depth = len(doc['path_parts'])
				indent = "  " * (depth - 1)
				anchor = self._sanitize_filename(doc['title'])
				toc.append(f"{indent}- [{doc['title']}](#{anchor})")
				heading_level = '#' * min(depth + 1, 6)
				content.append(f"\n\n{heading_level} {doc['title']}\n")
				
				if doc.get('headers'):
					content.append("\n### Contents\n")
					content.extend(f"- {header}" for header in doc['headers'])
					content.append("\n")
				
				content.append(doc['content'])
				content.append(f"\n\n[Source]({doc['url']})")

			combined_content = '\n'.join(toc) + '\n\n' + '\n'.join(content)
			final_content = f"""---
title: {domain} Complete Documentation
date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
domain: {domain}
---

{combined_content}
"""
			safe_domain = self._sanitize_filename(domain)
			combined_file = self.combined_dir / f"{safe_domain}_complete.md"
			async with aiofiles.open(combined_file, 'w', encoding='utf-8') as f:
				await f.write(final_content)
			logger.info(f"Generated combined documentation for {domain}: {combined_file}")
			return domain

		# Process all domains concurrently
		processed_domains = await asyncio.gather(
			*(process_domain(domain, docs) for domain, docs in docs_by_domain.items())
		)

		# Generate master index
		master_index = ["# Documentation Index\n\n"]
		for domain in processed_domains:
			safe_domain = self._sanitize_filename(domain)
			master_index.append(f"- [{domain}]({safe_domain}_complete.md)")

		index_file = self.combined_dir / "index.md"
		async with aiofiles.open(index_file, 'w', encoding='utf-8') as f:
			await f.write('\n'.join(master_index))
		logger.info(f"Generated master index: {index_file}")


async def main():
	parser = argparse.ArgumentParser(description='Documentation site scraper')
	parser.add_argument('--url', required=True, help='Starting URL to scrape')
	parser.add_argument('--output-dir', default='docs_output', help='Output directory')
	parser.add_argument('--max-concurrent', type=int, default=3, help='Maximum concurrent requests')
	parser.add_argument('--wait-time', type=float, default=3.0, help='Wait time for page load')
	parser.add_argument('--model', help='OpenAI model name (default: from env)')
	args = parser.parse_args()

	global MODEL_NAME, MAX_CONCURRENT
	if args.model:
		MODEL_NAME = args.model
	MAX_CONCURRENT = args.max_concurrent

	BROWSER_CONFIG.new_context_config.wait_for_network_idle_page_load_time = args.wait_time
	
	try:
		logger.info("Starting DocsiteToMD")
		scraper = DocsiteToMD(base_dir=args.output_dir)
		await scraper.initialize()
		
		domain = urlparse(args.url).netloc
		await scraper.process_url(args.url, domain)
		await scraper.generate_combined_docs()
		logger.success("Documentation conversion completed successfully")
	except Exception as e:
		logger.error(f"Fatal error: {str(e)}", exc_info=True)
		raise
	finally:
		await scraper.browser.close()
		logger.info("Browser closed")

if __name__ == "__main__":
	asyncio.run(main())

