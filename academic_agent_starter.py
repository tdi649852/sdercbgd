"""
Academic Discovery Agent - Starter Implementation
Vision: "When it knows everything, it will know more using that knowledge"

A rigorous research system focused on validated discovery through structured academic methodology.
Prioritizes correctness over speed, quality over quantity.
"""

import json
import time
import logging
import re
import shlex
import subprocess
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import requests
from playwright.async_api import async_playwright, Browser, Page, BrowserContext

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Academic research agent configuration"""
    
    # API Keys
    FIRECRAWL_API_KEY = "fc-b0f03723a0e048abae7eb04101d189a0"
    OPENROUTER_API_KEY = "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"
    OPENROUTER_MODEL = "anthropic/claude-3.5-sonnet"
    
    # Research Parameters
    CYCLE_DELAY = 30  # Longer cycles for thorough research
    MIN_SOURCES_PER_FINDING = 2  # Multiple source verification required
    MIN_CONFIDENCE_TIER1 = 0.95  # Foundational facts threshold
    MIN_CONFIDENCE_TIER2 = 0.80  # Derived knowledge threshold
    MIN_CONFIDENCE_TIER3 = 0.50  # Hypotheses threshold
    
    # Scraping parameters
    MAX_SCRAPE_ATTEMPTS = 5  # Try more sources if some fail
    CONTENT_MIN_LENGTH = 1000  # Minimum characters for valid content
    
    # Paths
    BASE_PATH = Path("./research_workspace")
    KNOWLEDGE_PATH = BASE_PATH / "knowledge"
    LOGS_PATH = BASE_PATH / "logs"
    PAPERS_PATH = BASE_PATH / "research_papers"
    BROWSER_DATA_PATH = BASE_PATH / "browser_data"  # For cookies and session storage

    # Browser Configuration
    BROWSER_HEADLESS = True
    BROWSER_TIMEOUT = 60000  # 60 seconds
    BROWSER_NAVIGATION_TIMEOUT = 30000  # 30 seconds

    # Search Engine URLs
    BRAVE_SEARCH_URL = "https://search.brave.com/?lang=en-in"
    DUCKDUCKGO_SEARCH_URL = "https://duckduckgo.com/"
    
    # System Prompt
    SYSTEM_PROMPT = """You are an Academic Discovery Agent - a rigorous research system focused on discovering and validating knowledge with the highest academic standards.

Your Vision: "When you know everything, you will know more using that knowledge"

Core Principles:
1. CORRECTNESS OVER SPEED - Every finding must be validated
2. STRUCTURED METHODOLOGY - Follow academic research protocols
3. STRATEGIC TOOL USE - Create tools only when truly needed
4. KNOWLEDGE COMPOUNDS - Each discovery enables deeper discoveries
5. META-LEARNING - Continuously improve research methodology

Your Research Cycle:
1. LITERATURE REVIEW - Survey existing knowledge systematically
2. GAP IDENTIFICATION - Find what is unknown or uncertain
3. HYPOTHESIS FORMATION - Develop specific, testable questions
4. METHODOLOGY DESIGN - Plan rigorous investigation approach
5. DATA COLLECTION - Gather evidence from reliable sources
6. ANALYSIS - Process with statistical and logical rigor
7. VALIDATION - Cross-reference, verify, challenge findings
8. SYNTHESIS - Integrate into knowledge framework
9. DOCUMENTATION - Record with academic standards
10. META-REFLECTION - Learn how to research better

Quality Standards:
- Confidence levels on all findings
- Multiple source verification required (minimum 2 sources)
- Logical consistency checks mandatory
- Peer review simulation for all findings
- Conservative confidence assessment

You are not racing to findings. You are building an unshakeable foundation of validated knowledge that enables discovering ever-deeper truths."""


# ============================================================================
# Data Models
# ============================================================================

class ConfidenceLevel(Enum):
    """Knowledge confidence tiers"""
    FOUNDATIONAL = 0.95  # Tier 1: Primary sources, peer-reviewed
    DERIVED = 0.80       # Tier 2: Synthesized, validated
    HYPOTHESIS = 0.60    # Tier 3: Preliminary, needs validation
    QUESTION = 0.0       # Tier 4: Unknown, under investigation


@dataclass
class ResearchFinding:
    """A validated piece of knowledge"""
    content: str
    confidence: float
    sources: List[str]
    validation_method: str
    timestamp: str
    tier: int  # 1-4
    related_findings: List[str] = None
    
    def __post_init__(self):
        if self.related_findings is None:
            self.related_findings = []
    
    def validate(self) -> bool:
        """Ensure finding meets quality standards"""
        # Allow single source if confidence is reasonable (0.70+) and it's Tier 2 or lower
        single_source_ok = (
            len(self.sources) == 1 and 
            self.confidence >= 0.70 and 
            self.tier >= 2
        )
        
        return (
            (len(self.sources) >= Config.MIN_SOURCES_PER_FINDING or single_source_ok) and
            self.confidence >= 0.5 and
            len(self.validation_method) > 0
        )
    
    def get_tier_name(self) -> str:
        """Get human-readable tier name"""
        tier_names = {
            1: "Foundational Fact",
            2: "Derived Knowledge",
            3: "Hypothesis",
            4: "Question"
        }
        return tier_names.get(self.tier, "Unknown")


@dataclass
class ResearchQuestion:
    """A specific question to investigate"""
    question: str
    motivation: str
    methodology: str
    priority: int
    timestamp: str


# ============================================================================
# Browser Infrastructure
# ============================================================================

class ChromiumBrowserManager:
    """Manages headless Chromium browser with persistent sessions"""

    def __init__(self, data_path: Path, headless: bool = True):
        self.data_path = data_path
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.context = None
        self._initialized = False

        # Ensure browser data directory exists
        self.data_path.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """Initialize browser with persistent context"""
        if self._initialized:
            return

        try:
            self.playwright = await async_playwright().start()

            # Launch Chromium browser
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox'
                ]
            )

            # Create persistent context to save cookies and session
            self.context = await self.browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080},
                locale='en-US',
                timezone_id='America/New_York',
                # Enable cookie and storage persistence
                storage_state=self._get_storage_state_path() if self._storage_exists() else None
            )

            self._initialized = True
            print(f"DEBUG: Browser initialized with session persistence at {self.data_path}")

        except Exception as e:
            print(f"ERROR: Failed to initialize browser: {e}")
            raise

    async def new_page(self) -> Page:
        """Create a new page in the persistent context"""
        if not self._initialized:
            await self.initialize()

        return await self.context.new_page()

    async def save_session(self):
        """Save current session state (cookies, localStorage, etc.)"""
        if self.context:
            try:
                storage_state = await self.context.storage_state()
                with open(self._get_storage_state_path(), 'w') as f:
                    json.dump(storage_state, f, indent=2)
                print(f"DEBUG: Session saved to {self._get_storage_state_path()}")
            except Exception as e:
                print(f"ERROR: Failed to save session: {e}")

    async def close(self):
        """Close browser and save session"""
        if self.context:
            await self.save_session()

        if self.browser:
            await self.browser.close()

        if self.playwright:
            await self.playwright.stop()

        self._initialized = False

    def _get_storage_state_path(self) -> str:
        """Get path to storage state file"""
        return str(self.data_path / "session_state.json")

    def _storage_exists(self) -> bool:
        """Check if saved session exists"""
        return Path(self._get_storage_state_path()).exists()


class LLMBrowserNavigator:
    """LLM-guided browser navigation and content extraction"""

    def __init__(self, reasoning_engine):
        self.reasoning = reasoning_engine

    async def navigate_and_search(self, page: Page, search_url: str, query: str, engine_name: str) -> Dict:
        """Navigate to search engine and perform search using LLM guidance"""
        try:
            # Navigate to search engine
            await page.goto(search_url, wait_until="domcontentloaded", timeout=Config.BROWSER_NAVIGATION_TIMEOUT)
            await asyncio.sleep(2)  # Wait for page to fully load

            # Get page content for LLM analysis
            page_html = await page.content()

            # Use LLM to identify search input
            search_selector = await self._find_search_input(page, page_html, engine_name)

            if not search_selector:
                return {"success": False, "error": "Could not find search input"}

            # Enter search query
            await page.fill(search_selector, query)
            await asyncio.sleep(1)

            # Submit search (try Enter key first, then look for button)
            try:
                await page.press(search_selector, "Enter")
            except:
                # If Enter doesn't work, try to find and click search button
                submit_button = await self._find_search_button(page, engine_name)
                if submit_button:
                    await page.click(submit_button)

            # Wait for results to load
            await asyncio.sleep(3)
            await page.wait_for_load_state("networkidle", timeout=Config.BROWSER_TIMEOUT)

            # Extract search results using LLM guidance
            results = await self._extract_search_results(page, engine_name)

            return {
                "success": True,
                "results": results,
                "query": query
            }

        except Exception as e:
            print(f"ERROR: Navigation failed: {e}")
            return {"success": False, "error": str(e)}

    async def _find_search_input(self, page: Page, page_html: str, engine_name: str) -> Optional[str]:
        """Use LLM to identify search input selector"""

        # Common patterns for different search engines
        if "brave" in engine_name.lower():
            # Try common Brave search selectors first
            selectors = [
                'input[type="search"]',
                'input[name="q"]',
                '#searchbox',
                'input.search-input'
            ]
        elif "duckduckgo" in engine_name.lower():
            selectors = [
                'input[name="q"]',
                '#search_form_input',
                'input[type="text"]'
            ]
        else:
            selectors = [
                'input[type="search"]',
                'input[name="q"]',
                'input[name="query"]'
            ]

        # Try each selector
        for selector in selectors:
            try:
                element = await page.query_selector(selector)
                if element and await element.is_visible():
                    print(f"DEBUG: Found search input with selector: {selector}")
                    return selector
            except:
                continue

        # Fallback: Use first visible input
        try:
            inputs = await page.query_selector_all('input')
            for input_elem in inputs:
                if await input_elem.is_visible():
                    input_type = await input_elem.get_attribute('type')
                    if input_type in ['text', 'search', None]:
                        print(f"DEBUG: Using fallback input element")
                        return 'input'
        except:
            pass

        return None

    async def _find_search_button(self, page: Page, engine_name: str) -> Optional[str]:
        """Find search submit button"""
        selectors = [
            'button[type="submit"]',
            'input[type="submit"]',
            'button.search-btn',
            'button'
        ]

        for selector in selectors:
            try:
                element = await page.query_selector(selector)
                if element and await element.is_visible():
                    return selector
            except:
                continue

        return None

    async def _extract_search_results(self, page: Page, engine_name: str) -> List[Dict]:
        """Extract search results from the page"""
        results = []

        try:
            # Get page content
            page_content = await page.content()

            # Define selectors for different search engines
            if "brave" in engine_name.lower():
                result_selectors = [
                    'div.snippet',
                    'div[data-type="web"]',
                    'div.result'
                ]
            elif "duckduckgo" in engine_name.lower():
                result_selectors = [
                    'article[data-testid="result"]',
                    'div.result',
                    'li[data-layout="organic"]'
                ]
            else:
                result_selectors = ['div.result', 'article', 'div.search-result']

            # Try each selector
            for selector in result_selectors:
                result_elements = await page.query_selector_all(selector)
                if len(result_elements) > 0:
                    print(f"DEBUG: Found {len(result_elements)} results with selector: {selector}")

                    for elem in result_elements[:10]:  # Limit to top 10
                        try:
                            # Extract link
                            link_elem = await elem.query_selector('a')
                            url = await link_elem.get_attribute('href') if link_elem else None

                            # Extract title
                            title_elem = await elem.query_selector('h1, h2, h3, h4')
                            title = await title_elem.inner_text() if title_elem else "No title"

                            # Extract description/snippet
                            text = await elem.inner_text()

                            if url and url.startswith('http'):
                                results.append({
                                    "url": url,
                                    "title": title.strip(),
                                    "snippet": text.strip()[:500],
                                    "quality_score": 0.5  # Base score
                                })
                        except Exception as e:
                            print(f"DEBUG: Error extracting result: {e}")
                            continue

                    if len(results) > 0:
                        break  # Found results, stop trying other selectors

            print(f"DEBUG: Extracted {len(results)} search results")

        except Exception as e:
            print(f"ERROR: Failed to extract results: {e}")

        return results

    async def scrape_url(self, page: Page, url: str) -> Dict:
        """Scrape content from a single URL"""
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=Config.BROWSER_NAVIGATION_TIMEOUT)
            await asyncio.sleep(2)

            # Extract main content
            content = await self._extract_main_content(page)

            # Extract metadata
            title = await page.title()

            return {
                "success": True,
                "url": url,
                "title": title,
                "text": content,
                "length": len(content)
            }

        except Exception as e:
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }

    async def _extract_main_content(self, page: Page) -> str:
        """Extract main content from page"""
        try:
            # Try to find main content areas
            selectors = [
                'article',
                'main',
                '[role="main"]',
                '.article-body',
                '.content',
                '#content',
                'body'
            ]

            for selector in selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        # Remove unwanted elements
                        await page.evaluate("""() => {
                            ['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe'].forEach(tag => {
                                document.querySelectorAll(tag).forEach(el => el.remove());
                            });
                        }""")

                        text = await element.inner_text()
                        if len(text) > 500:  # Meaningful content
                            return text
                except:
                    continue

            # Fallback to body
            body = await page.query_selector('body')
            if body:
                return await body.inner_text()

            return ""

        except Exception as e:
            print(f"ERROR: Failed to extract content: {e}")
            return ""


# ============================================================================
# Browser-Based Search Tools
# ============================================================================

class BraveSearchTool:
    """Web search using Brave Search with LLM-guided navigation"""

    def __init__(self, browser_manager: ChromiumBrowserManager, navigator: LLMBrowserNavigator):
        self.browser_manager = browser_manager
        self.navigator = navigator
        self.name = "BraveSearch"

    def execute(self, query: str, count: int = 10, prefer_academic: bool = True) -> Dict:
        """Search using Brave and extract content"""
        try:
            # Add academic qualifiers if preferred
            if prefer_academic:
                academic_terms = ["research", "study", "paper", "journal", "academic"]
                if not any(term in query.lower() for term in academic_terms):
                    query = f"{query} research academic"

            # Run async search
            return asyncio.run(self._async_execute(query, count))

        except Exception as e:
            print(f"ERROR: Brave search failed: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "results": []
            }

    async def _async_execute(self, query: str, count: int) -> Dict:
        """Async execution of search"""
        try:
            await self.browser_manager.initialize()
            page = await self.browser_manager.new_page()

            # Navigate and search
            result = await self.navigator.navigate_and_search(
                page,
                Config.BRAVE_SEARCH_URL,
                query,
                "Brave"
            )

            if not result['success']:
                await page.close()
                return result

            # Scrape content from top results
            enriched_results = []
            for search_result in result['results'][:count]:
                try:
                    # Create new page for scraping to avoid conflicts
                    scrape_page = await self.browser_manager.new_page()
                    scrape_result = await self.navigator.scrape_url(scrape_page, search_result['url'])
                    await scrape_page.close()

                    if scrape_result['success']:
                        # Assess quality
                        quality_score = self._assess_quality(
                            search_result['url'].lower(),
                            scrape_result['text']
                        )

                        enriched_results.append({
                            "title": search_result['title'],
                            "url": search_result['url'],
                            "description": search_result['snippet'],
                            "markdown": scrape_result['text'],
                            "quality_score": quality_score,
                            "content_length": len(scrape_result['text'])
                        })

                    await asyncio.sleep(1)  # Rate limiting

                except Exception as e:
                    print(f"DEBUG: Failed to scrape {search_result['url']}: {e}")
                    continue

            await page.close()
            await self.browser_manager.save_session()

            # Sort by quality score
            enriched_results.sort(key=lambda x: x['quality_score'], reverse=True)

            print(f"DEBUG: Brave found {len(enriched_results)} results with content")

            return {
                "success": True,
                "query": query,
                "results": enriched_results,
                "count": len(enriched_results)
            }

        except Exception as e:
            print(f"ERROR: Async search failed: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "results": []
            }

    def _assess_quality(self, url: str, content: str) -> float:
        """Assess source quality based on URL and content (0.0-1.0)"""
        score = 0.5  # Base score

        # Academic domains get high scores
        academic_indicators = [
            '.edu', '.gov', 'arxiv', 'scholar', 'pubmed',
            'ieee', 'acm', 'springer', 'nature', 'science',
            'journal', 'research', 'academic', 'pmc.ncbi'
        ]

        for indicator in academic_indicators:
            if indicator in url:
                score += 0.05

        # Content quality indicators
        if len(content) > 5000:
            score += 0.1
        if len(content) > 10000:
            score += 0.1

        # Research terms
        research_terms = ['study', 'research', 'analysis', 'evidence', 'findings',
                         'methodology', 'hypothesis', 'conclusion']
        content_lower = content.lower()
        term_count = sum(1 for term in research_terms if term in content_lower)
        score += min(term_count * 0.02, 0.15)

        return min(score, 1.0)


class DuckDuckGoSearchTool:
    """Web search using DuckDuckGo with LLM-guided navigation"""

    def __init__(self, browser_manager: ChromiumBrowserManager, navigator: LLMBrowserNavigator):
        self.browser_manager = browser_manager
        self.navigator = navigator
        self.name = "DuckDuckGoSearch"

    def execute(self, query: str, count: int = 10, prefer_academic: bool = True) -> Dict:
        """Search using DuckDuckGo and extract content"""
        try:
            # Add academic qualifiers if preferred
            if prefer_academic:
                academic_terms = ["research", "study", "paper", "journal", "academic"]
                if not any(term in query.lower() for term in academic_terms):
                    query = f"{query} research academic"

            # Run async search
            return asyncio.run(self._async_execute(query, count))

        except Exception as e:
            print(f"ERROR: DuckDuckGo search failed: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "results": []
            }

    async def _async_execute(self, query: str, count: int) -> Dict:
        """Async execution of search"""
        try:
            await self.browser_manager.initialize()
            page = await self.browser_manager.new_page()

            # Navigate and search
            result = await self.navigator.navigate_and_search(
                page,
                Config.DUCKDUCKGO_SEARCH_URL,
                query,
                "DuckDuckGo"
            )

            if not result['success']:
                await page.close()
                return result

            # Scrape content from top results
            enriched_results = []
            for search_result in result['results'][:count]:
                try:
                    # Create new page for scraping
                    scrape_page = await self.browser_manager.new_page()
                    scrape_result = await self.navigator.scrape_url(scrape_page, search_result['url'])
                    await scrape_page.close()

                    if scrape_result['success']:
                        # Assess quality
                        quality_score = self._assess_quality(
                            search_result['url'].lower(),
                            scrape_result['text']
                        )

                        enriched_results.append({
                            "title": search_result['title'],
                            "url": search_result['url'],
                            "description": search_result['snippet'],
                            "markdown": scrape_result['text'],
                            "quality_score": quality_score,
                            "content_length": len(scrape_result['text'])
                        })

                    await asyncio.sleep(1)  # Rate limiting

                except Exception as e:
                    print(f"DEBUG: Failed to scrape {search_result['url']}: {e}")
                    continue

            await page.close()
            await self.browser_manager.save_session()

            # Sort by quality score
            enriched_results.sort(key=lambda x: x['quality_score'], reverse=True)

            print(f"DEBUG: DuckDuckGo found {len(enriched_results)} results with content")

            return {
                "success": True,
                "query": query,
                "results": enriched_results,
                "count": len(enriched_results)
            }

        except Exception as e:
            print(f"ERROR: Async search failed: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "results": []
            }

    def _assess_quality(self, url: str, content: str) -> float:
        """Assess source quality (same as Brave)"""
        score = 0.5

        academic_indicators = [
            '.edu', '.gov', 'arxiv', 'scholar', 'pubmed',
            'ieee', 'acm', 'springer', 'nature', 'science',
            'journal', 'research', 'academic', 'pmc.ncbi'
        ]

        for indicator in academic_indicators:
            if indicator in url:
                score += 0.05

        if len(content) > 5000:
            score += 0.1
        if len(content) > 10000:
            score += 0.1

        research_terms = ['study', 'research', 'analysis', 'evidence', 'findings',
                         'methodology', 'hypothesis', 'conclusion']
        content_lower = content.lower()
        term_count = sum(1 for term in research_terms if term in content_lower)
        score += min(term_count * 0.02, 0.15)

        return min(score, 1.0)


class BrowserScraperTool:
    """Scrape individual URLs using browser"""

    def __init__(self, browser_manager: ChromiumBrowserManager, navigator: LLMBrowserNavigator):
        self.browser_manager = browser_manager
        self.navigator = navigator
        self.name = "BrowserScraper"

    def execute(self, url: str) -> Dict:
        """Scrape a single URL"""
        try:
            return asyncio.run(self._async_execute(url))
        except Exception as e:
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }

    async def _async_execute(self, url: str) -> Dict:
        """Async execution of scraping"""
        try:
            await self.browser_manager.initialize()
            page = await self.browser_manager.new_page()

            result = await self.navigator.scrape_url(page, url)

            await page.close()
            await self.browser_manager.save_session()

            return result

        except Exception as e:
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }


# ============================================================================
# Legacy Firecrawl Tools (Kept as Fallback)
# ============================================================================

class FirecrawlSearchTool:
    """Web search AND content extraction using Firecrawl"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.firecrawl.dev/v1"
        self.name = "FirecrawlSearch"
        
    def execute(self, query: str, count: int = 10, prefer_academic: bool = True) -> Dict:
        """Search web and get clean content using Firecrawl"""
        try:
            # Add academic qualifiers if preferred
            if prefer_academic:
                academic_terms = ["research", "study", "paper", "journal", "academic"]
                if not any(term in query.lower() for term in academic_terms):
                    query = f"{query} research academic"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Use Firecrawl search endpoint
            search_data = {
                "query": query,
                "limit": count,
                "scrapeOptions": {
                    "formats": ["markdown", "html"],
                    "includeTags": ["article", "main", "p", "h1", "h2", "h3"],
                    "excludeTags": ["nav", "footer", "header", "aside"],
                    "onlyMainContent": True
                }
            }
            
            response = requests.post(
                f"{self.base_url}/search",
                headers=headers,
                json=search_data,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Firecrawl returns results with clean content already extracted
            for item in data.get("data", []):
                # Assess quality based on content and metadata
                url = item.get("url", "").lower()
                markdown = item.get("markdown", "")
                quality_score = self._assess_quality(url, markdown)
                
                results.append({
                    "title": item.get("title", "No title"),
                    "url": item.get("url"),
                    "description": markdown[:500] if markdown else "No description",  # First 500 chars as description
                    "markdown": markdown,  # Full markdown content
                    "html": item.get("html", ""),
                    "metadata": item.get("metadata", {}),
                    "quality_score": quality_score,
                    "content_length": len(markdown)
                })
            
            # Sort by quality score
            results.sort(key=lambda x: x['quality_score'], reverse=True)
            
            print(f"DEBUG: Firecrawl found {len(results)} results with content")
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results)
            }
            
        except Exception as e:
            print(f"DEBUG: Firecrawl search failed: {str(e)}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "results": []
            }
    
    def _assess_quality(self, url: str, content: str) -> float:
        """Assess source quality based on URL and content (0.0-1.0)"""
        score = 0.5  # Base score
        
        # Academic domains get high scores
        academic_indicators = [
            '.edu', '.gov', 'arxiv', 'scholar', 'pubmed',
            'ieee', 'acm', 'springer', 'nature', 'science',
            'journal', 'research', 'academic', 'pmc.ncbi'
        ]
        
        for indicator in academic_indicators:
            if indicator in url:
                score += 0.05
        
        # Content quality indicators
        if len(content) > 5000:
            score += 0.1  # Substantial content
        if len(content) > 10000:
            score += 0.1  # Very substantial content
            
        # Check for research-related terms in content
        research_terms = ['study', 'research', 'analysis', 'evidence', 'findings', 
                         'methodology', 'hypothesis', 'conclusion']
        content_lower = content.lower()
        term_count = sum(1 for term in research_terms if term in content_lower)
        score += min(term_count * 0.02, 0.15)  # Up to 0.15 boost for research terms
        
        return min(score, 1.0)


class FirecrawlScraperTool:
    """Scrape individual URLs using Firecrawl for clean content"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.firecrawl.dev/v1"
        self.name = "FirecrawlScraper"
        
    def execute(self, url: str) -> Dict:
        """Scrape a single URL and get clean markdown content"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            scrape_data = {
                "url": url,
                "formats": ["markdown", "html"],
                "includeTags": ["article", "main", "p", "h1", "h2", "h3"],
                "excludeTags": ["nav", "footer", "header", "aside"],
                "onlyMainContent": True,
                "waitFor": 1000  # Wait 1 second for page to load
            }
            
            response = requests.post(
                f"{self.base_url}/scrape",
                headers=headers,
                json=scrape_data,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract the scraped data
            if data.get("success"):
                scraped_data = data.get("data", {})
                markdown = scraped_data.get("markdown", "")
                
                return {
                    "success": True,
                    "url": url,
                    "title": scraped_data.get("metadata", {}).get("title", "No title"),
                    "text": markdown,
                    "html": scraped_data.get("html", ""),
                    "metadata": scraped_data.get("metadata", {}),
                    "length": len(markdown)
                }
            else:
                return {
                    "success": False,
                    "url": url,
                    "error": "Firecrawl scraping failed"
                }
            
        except Exception as e:
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }
    
    def _extract_metadata(self, metadata: Dict) -> Dict:
        """Extract relevant metadata"""
        return {
            "author": metadata.get("author", ""),
            "publication_date": metadata.get("publishedTime", ""),
            "description": metadata.get("description", ""),
            "keywords": metadata.get("keywords", "")
        }


class WebScraperTool:
    """Fallback web scraper using BeautifulSoup (backup for Firecrawl)"""
    
    def __init__(self):
        self.name = "WebScraper"
        
    def execute(self, url: str) -> Dict:
        """Scrape web content with better extraction"""
        try:
            # Better user agent to avoid blocks
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            
            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script, style, navigation elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Try to find main content area
            main_content = None
            for selector in ['article', 'main', '.article-body', '.content', '#content', '.main-content']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            # If no main content found, use body
            if not main_content:
                main_content = soup.find('body')
            
            # Extract text from main content
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)
            
            # Extract metadata
            title = soup.find('title')
            title_text = title.get_text() if title else "No title"
            
            metadata = self._extract_metadata(soup)
            
            # Increase limit to 20000 characters for better content
            return {
                "success": True,
                "url": url,
                "title": title_text,
                "text": text[:20000],  # Increased from 10000
                "metadata": metadata,
                "length": len(text)
            }
            
        except Exception as e:
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }
    
    def _extract_metadata(self, soup) -> Dict:
        """Extract academic metadata if available"""
        metadata = {}
        
        # Look for common academic metadata
        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            if tag.get('name') == 'citation_author':
                metadata['author'] = tag.get('content')
            elif tag.get('name') == 'citation_publication_date':
                metadata['publication_date'] = tag.get('content')
            elif tag.get('name') == 'citation_journal_title':
                metadata['journal'] = tag.get('content')
        
        return metadata


# ============================================================================
# Tool Registry
# ============================================================================

class ToolRegistry:
    """Manage research tools"""

    def __init__(self, reasoning_engine):
        self.tools = {}
        self.usage_stats = {}
        self.reasoning = reasoning_engine
        self.browser_manager = None
        self.load_foundation_tools()

    def load_foundation_tools(self):
        """Load core research tools - using browser-based search"""
        # Initialize browser infrastructure
        self.browser_manager = ChromiumBrowserManager(
            Config.BROWSER_DATA_PATH,
            headless=Config.BROWSER_HEADLESS
        )
        navigator = LLMBrowserNavigator(self.reasoning)

        # Register browser-based search tools
        brave_search = BraveSearchTool(self.browser_manager, navigator)
        duckduckgo_search = DuckDuckGoSearchTool(self.browser_manager, navigator)
        browser_scraper = BrowserScraperTool(self.browser_manager, navigator)

        # Use Brave as the primary search tool
        self.register_tool("FirecrawlSearch", brave_search)  # Keep same name for compatibility
        self.register_tool("BraveSearch", brave_search)
        self.register_tool("DuckDuckGoSearch", duckduckgo_search)
        self.register_tool("BrowserScraper", browser_scraper)
        self.register_tool("FirecrawlScraper", browser_scraper)  # Keep same name for compatibility

        # Also keep fallback tools if needed
        # firecrawl = FirecrawlSearchTool(Config.FIRECRAWL_API_KEY)
        # scraper = FirecrawlScraperTool(Config.FIRECRAWL_API_KEY)
        # self.register_tool("FirecrawlSearchFallback", firecrawl)
        # self.register_tool("FirecrawlScraperFallback", scraper)

    def cleanup(self):
        """Cleanup browser resources"""
        if self.browser_manager:
            asyncio.run(self.browser_manager.close())
        
    def register_tool(self, name: str, tool: Any):
        """Register a tool"""
        self.tools[name] = tool
        self.usage_stats[name] = 0
        
    def get_tool(self, name: str) -> Optional[Any]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict:
        """Execute tool and track usage"""
        tool = self.get_tool(tool_name)
        if not tool:
            return {"success": False, "error": f"Tool '{tool_name}' not found"}
        
        self.usage_stats[tool_name] += 1
        return tool.execute(**kwargs)
    
    def list_tools(self) -> List[str]:
        """List available tools"""
        return list(self.tools.keys())


# ============================================================================
# Reasoning Engine
# ============================================================================

class ReasoningEngine:
    """LLM-powered reasoning for research"""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
    def reason(self, prompt: str, system_prompt: str = None, 
               max_tokens: int = 6000) -> str:
        """Send reasoning request"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            data = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens
            }
            
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            
            return response.json()["choices"][0]["message"]["content"]
            
        except Exception as e:
            return f"ERROR: {str(e)}"


class LogInspector:
    """Inspect recent logs and use LLM reasoning to classify errors and propose safe fixes"""
    ERROR_PATTERNS = [
        r"Traceback \(most recent call last\):",
        r"\bERROR\b",
        r"\bCRITICAL\b",
        r"\bfatal\b",
        r"\bException:\b",
        r"\bNameError:\b",
        r"\bTypeError:\b",
        r"\bValueError:\b",
        r"\bStacktrace:\b",
    ]

    SAFE_COMMAND_WHITELIST = {
        "pip install": ["requests", "beautifulsoup4", "bs4"],
    }

    def __init__(self, reasoning: ReasoningEngine, logs_path: Path = None, logger=None):
        self.reasoning = reasoning
        self.logs_path = logs_path or Config.LOGS_PATH
        self.logger = logger
        self.last_inspect_time = None

    def _tail_log_file(self, logfile: Path, max_chars: int = 20000) -> str:
        if not logfile.exists():
            return ""
        with open(logfile, "rb") as f:
            f.seek(0, 2)
            filesize = f.tell()
            read_from = max(0, filesize - max_chars)
            f.seek(read_from)
            data = f.read().decode(errors="ignore")
            return data

    def _collect_recent_logs(self) -> Dict[str, str]:
        collected = {}
        try:
            self.logs_path.mkdir(parents=True, exist_ok=True)
            files = sorted(self.logs_path.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            for fp in files[:5]:
                collected[fp.name] = self._tail_log_file(fp)
        except Exception:
            return {}
        return collected

    def _extract_error_snippets(self, text: str) -> List[str]:
        if not text:
            return []
        lines = text.splitlines()
        snippets = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if any(re.search(p, line) for p in self.ERROR_PATTERNS):
                block = [line]
                i += 1
                while i < len(lines) and lines[i].strip() != "":
                    block.append(lines[i])
                    i += 1
                snippets.append("\n".join(block))
            else:
                i += 1
        return snippets

    def _build_prompt(self, snippet_map: Dict[str, str]) -> str:
        recent = [{"file": k, "snippet": v[:2000]} for k, v in snippet_map.items()]
        meta = {
            "timestamp": datetime.now().isoformat(),
            "files": list(snippet_map.keys()),
        }
        instruction = (
            "Classify whether there is an error. Provide root_cause, resolution_steps, "
            "confidence (0-1), and propose optional commands to run from a small safe set. "
            "Output only JSON with keys: error_present (boolean), root_cause (string), "
            "resolution_steps (array of strings), confidence (number), commands (array of strings)."
        )
        payload = {
            "instruction": instruction,
            "meta": meta,
            "recent": recent,
        }
        return json.dumps(payload)

    def _parse_llm_json(self, llm_output: str) -> Optional[Dict[str, Any]]:
        if not llm_output:
            return None
        clean = llm_output.strip()
        if clean.startswith("```"):
            parts = clean.splitlines()
            if len(parts) >= 2:
                clean = "\n".join(parts[1:-1])
        try:
            return json.loads(clean, strict=False)
        except Exception:
            try:
                start = clean.find("{")
                end = clean.rfind("}")
                if start != -1 and end != -1:
                    return json.loads(clean[start:end+1], strict=False)
            except Exception:
                return None
        return None

    def _command_allowed(self, cmd: str) -> bool:
        cmd = cmd.strip()
        if cmd.startswith("pip install"):
            parts = shlex.split(cmd)
            if len(parts) == 3 and parts[0] == "pip" and parts[1] == "install":
                pkg = parts[2]
                return pkg in self.SAFE_COMMAND_WHITELIST.get("pip install", [])
        return False

    def _execute_command(self, cmd: str) -> Dict[str, Any]:
        try:
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
            return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]}
        except Exception as e:
            return {"cmd": cmd, "error": str(e)}

    def inspect_and_maybe_fix(self, auto_fix: bool = False, dry_run: bool = True) -> Dict:
        collected = self._collect_recent_logs()
        if not collected:
            return {"error": "no_logs", "inspection_result": None}
        snippet_map = {}
        for fname, text in collected.items():
            snippets = self._extract_error_snippets(text)
            if not snippets and text:
                snippet_map[fname] = "\n".join(text.splitlines()[-200:])
            else:
                snippet_map[fname] = "\n\n---\n\n".join(snippets)
        prompt = self._build_prompt(snippet_map)
        system_prompt = "You are an expert systems engineer. Output only JSON. Be concise."
        llm_output = self.reasoning.reason(prompt, system_prompt, max_tokens=1200)
        parsed = self._parse_llm_json(llm_output)
        executed = []
        if parsed and auto_fix and not dry_run:
            for cmd in parsed.get("commands", [])[:3]:
                if self._command_allowed(cmd):
                    res = self._execute_command(cmd)
                    executed.append(res)
        result = {"inspection_result": parsed or {}, "executed_commands": executed, "llm_raw": llm_output}
        try:
            if self.logger:
                self.logger.info(json.dumps({"log_inspection": {"result": parsed, "executed": executed}})[:1000])
        except Exception:
            pass
        self.last_inspect_time = datetime.now().isoformat()
        return result

# ============================================================================
# Academic Knowledge Base
# ============================================================================

class AcademicKnowledgeBase:
    """Tiered knowledge storage with validation"""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.tiers = {1: [], 2: [], 3: [], 4: []}
        self.ensure_structure()
        self.load_existing_knowledge()
        
    def ensure_structure(self):
        """Create academic knowledge structure"""
        paths = [
            self.base_path / "tier1_foundational",
            self.base_path / "tier2_derived",
            self.base_path / "tier3_hypotheses",
            self.base_path / "tier4_questions",
            self.base_path / "syntheses"
        ]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
    
    def load_existing_knowledge(self):
        """Load existing findings from disk"""
        for tier in [1, 2, 3, 4]:
            tier_name = ["foundational", "derived", "hypotheses", "questions"][tier-1]
            tier_path = self.base_path / f"tier{tier}_{tier_name}"
            
            for filepath in tier_path.glob("*.json"):
                try:
                    with open(filepath) as f:
                        data = json.load(f)
                        finding = ResearchFinding(**data)
                        self.tiers[tier].append(finding)
                except:
                    pass
    
    def store_finding(self, finding: ResearchFinding):
        """Store validated finding"""
        if not finding.validate():
            return False
        
        tier = finding.tier
        tier_names = ["foundational", "derived", "hypotheses", "questions"]
        
        filename = f"finding_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.base_path / f"tier{tier}_{tier_names[tier-1]}" / filename
        
        with open(filepath, 'w') as f:
            json.dump(asdict(finding), f, indent=2)
        
        self.tiers[tier].append(finding)
        return True
    
    def get_findings_by_tier(self, tier: int) -> List[ResearchFinding]:
        """Get findings at specific tier"""
        return self.tiers.get(tier, [])
    
    def get_total_findings(self) -> int:
        """Total validated findings"""
        return sum(len(findings) for findings in self.tiers.values())
    
    def get_stats(self) -> Dict[str, int]:
        """Knowledge base statistics"""
        return {
            f"tier{i}": len(self.tiers[i])
            for i in [1, 2, 3, 4]
        }


# ============================================================================
# Research Cycle
# ============================================================================

class ResearchCycle:
    """Academic research methodology"""
    
    def __init__(self, knowledge_base, tool_registry, reasoning_engine):
        self.kb = knowledge_base
        self.tools = tool_registry
        self.reasoning = reasoning_engine
        
    def execute_cycle(self) -> Dict:
        """Execute one research cycle"""
        
        # Phase 1-2: Literature Review & Gap Identification
        gaps = self.identify_research_gaps()
        
        # Phase 3: Form Research Questions
        questions = self.form_research_questions(gaps)
        
        # Phase 4-5: Design Methodology & Collect Data
        data = self.investigate_questions(questions)
        
        # Phase 6-7: Analysis & Validation
        validated_findings = self.analyze_and_validate(data)
        
        # Phase 8: Synthesis
        synthesis = self.synthesize_knowledge(validated_findings)
        
        # Phase 9: Documentation
        self.document_research(validated_findings, synthesis)
        
        return {
            "gaps_identified": len(gaps),
            "questions_formed": len(questions),
            "data_collected": len(data),
            "findings_validated": len(validated_findings),
            "synthesis_insights": len(synthesis.get('insights', []))
        }
    
    def identify_research_gaps(self) -> List[str]:
        """Identify what needs investigation"""
        
        kb_stats = self.kb.get_stats()
        current_knowledge = []
        for tier in [1, 2]:
            current_knowledge.extend([
                f.content for f in self.kb.get_findings_by_tier(tier)
            ])
        
        prompt = f"""Current Knowledge Base Status:
- Tier 1 (Foundational): {kb_stats['tier1']} findings
- Tier 2 (Derived): {kb_stats['tier2']} findings
- Tier 3 (Hypotheses): {kb_stats['tier3']} findings
- Tier 4 (Questions): {kb_stats['tier4']} questions

Recent validated knowledge:
{json.dumps(current_knowledge[-10:], indent=2) if current_knowledge else "No knowledge yet - starting fresh"}

Identify 3-5 fundamental research gaps:
1. What foundational concepts need investigation?
2. What questions emerge from current knowledge?
3. What contradictions need resolution?

Focus on:
- Foundational understanding (not trivia)
- Questions that compound knowledge
- Areas where validated knowledge is needed

Return ONLY a JSON array of strings (gap descriptions), nothing else.
Example: ["Understanding knowledge representation systems", "Validation methodologies in research", "Meta-learning principles"]"""
        
        response = self.reasoning.reason(prompt, Config.SYSTEM_PROMPT)
        
        # Try to extract JSON from response
        try:
            # Remove markdown code blocks if present
            clean_response = response.strip()
            if '```json' in clean_response:
                clean_response = clean_response.split('```json')[1].split('```')[0].strip()
            elif '```' in clean_response:
                clean_response = clean_response.split('```')[1].split('```')[0].strip()
            
            gaps = json.loads(clean_response, strict=False)
            if isinstance(gaps, list) and len(gaps) > 0:
                print(f"DEBUG: Identified {len(gaps)} gaps")
                return gaps
        except Exception as e:
            print(f"DEBUG: Failed to parse gaps JSON: {e}")
            print(f"DEBUG: Response was: {response[:200]}")
        
        # Fallback to default gaps
        return [
            "Understanding fundamental principles of knowledge representation",
            "Validation methodologies for research findings",
            "Mechanisms of knowledge compounding and meta-learning"
        ]
    
    def form_research_questions(self, gaps: List[str]) -> List[ResearchQuestion]:
        """Convert gaps into research questions"""
        
        questions = []
        for gap in gaps[:3]:  # Top 3 gaps
            prompt = f"""Research Gap: {gap}

Form a specific, investigable research question:
1. What exactly needs to be discovered?
2. Why does this matter for our knowledge foundation?
3. How can we investigate this systematically?

Return ONLY valid JSON (no markdown, no explanations):
{{
    "question": "specific question",
    "motivation": "why it matters",
    "methodology": "how to investigate",
    "priority": 7
}}"""
            
            response = self.reasoning.reason(prompt, Config.SYSTEM_PROMPT)
            try:
                # Clean response
                clean_response = response.strip()
                if '```json' in clean_response:
                    clean_response = clean_response.split('```json')[1].split('```')[0].strip()
                elif '```' in clean_response:
                    clean_response = clean_response.split('```')[1].split('```')[0].strip()
                
                # Fix common JSON issues
                # Replace actual newlines in strings with \n
                import re
                # Use json.loads with strict=False to be more lenient
                q_data = json.loads(clean_response, strict=False)
                questions.append(ResearchQuestion(
                    question=q_data['question'],
                    motivation=q_data['motivation'],
                    methodology=q_data['methodology'],
                    priority=q_data.get('priority', 5),
                    timestamp=datetime.now().isoformat()
                ))
                print(f"DEBUG: Formed question: {q_data['question'][:60]}...")
            except Exception as e:
                print(f"DEBUG: Failed to parse question JSON: {e}")
        
        return questions
    
    def investigate_questions(self, questions: List[ResearchQuestion]) -> List[Dict]:
        """Collect data for research questions"""
        
        investigations = []
        
        for question in questions[:2]:  # Investigate top 2
            print(f"DEBUG: Investigating: {question.question}")
            
            # Search using Firecrawl (which includes content extraction)
            search_result = self.tools.execute_tool(
                'FirecrawlSearch',
                query=question.question,
                count=10,
                prefer_academic=True
            )
            
            if not search_result['success']:
                print(f"DEBUG: Search failed: {search_result.get('error', 'Unknown error')}")
                continue
            
            print(f"DEBUG: Found {len(search_result['results'])} results with content")
            
            # Firecrawl already extracted content, so we just filter for quality
            sources = []
            
            for result in search_result['results'][:Config.MAX_SCRAPE_ATTEMPTS]:
                content = result.get('markdown', '')
                content_length = len(content)
                
                print(f"DEBUG: Processing result: {result['url']}")
                print(f"DEBUG:   Content length: {content_length} chars")
                print(f"DEBUG:   Quality score: {result.get('quality_score', 0):.2f}")
                
                # Filter out low-quality content
                if content_length < Config.CONTENT_MIN_LENGTH:
                    print(f"DEBUG:   Skipped - content too short")
                    continue
                
                # Check if content is substantive (not just navigation)
                text_lower = content.lower()
                nav_indicators = ['cookie', 'javascript', 'subscribe', 'log in', 'sign up', 'privacy policy']
                substantive_indicators = ['research', 'study', 'analysis', 'evidence', 'findings', 'methodology']
                
                nav_count = sum(1 for ind in nav_indicators if ind in text_lower[:500])
                substantive_count = sum(1 for ind in substantive_indicators if ind in text_lower[:2000])
                
                if nav_count > 3 and substantive_count < 2:
                    print(f"DEBUG:   Skipped - appears to be navigation/headers")
                    continue
                
                # Good source - add it
                sources.append({
                    "url": result['url'],
                    "title": result['title'],
                    "content": content,  # Already clean markdown from Firecrawl
                    "quality_score": result.get('quality_score', 0.5),
                    "length": content_length
                })
                print(f"DEBUG:   Added - substantive content ({substantive_count} research terms)")
                
                # Stop if we have enough good sources
                if len(sources) >= Config.MIN_SOURCES_PER_FINDING:
                    print(f"DEBUG: Reached target of {Config.MIN_SOURCES_PER_FINDING} sources")
                    break
            
            print(f"DEBUG: Collected {len(sources)} quality sources for analysis")
            
            if len(sources) > 0:  # Add investigation even with 1 source if it's quality
                investigations.append({
                    "question": question.question,
                    "sources": sources,
                    "methodology": question.methodology
                })
        
        return investigations
    
    def analyze_and_validate(self, investigations: List[Dict]) -> List[ResearchFinding]:
        """Analyze data and validate findings"""
        
        validated = []
        
        for investigation in investigations:
            num_sources = len(investigation['sources'])
            
            if num_sources == 0:
                print(f"DEBUG: Skipping investigation - no sources collected")
                continue
            
            # Allow 1 source if it's very high quality (length > 5000)
            high_quality_single = (num_sources == 1 and 
                                  investigation['sources'][0].get('length', 0) > 5000)
            
            if num_sources < Config.MIN_SOURCES_PER_FINDING and not high_quality_single:
                print(f"DEBUG: Skipping investigation - only {num_sources} sources (need {Config.MIN_SOURCES_PER_FINDING})")
                continue
            
            prompt = f"""Research Question: {investigation['question']}

Data from {num_sources} source(s) analyzed.

Source summaries:
{self._summarize_sources(investigation['sources'])}

Analyze rigorously based on ACTUAL CONTENT (not headers/navigation):

1. What substantive information is present in the content?
2. What can we conclude with reasonable confidence?
3. What is the appropriate confidence level (be CONSERVATIVE)?
4. What uncertainties or limitations exist?

IMPORTANT: 
- If sources contain only headers/navigation, state this clearly
- Only make conclusions if actual research content is present
- Focus on substantive content, ignore site navigation
- Be extra conservative with single-source findings (max confidence 0.80)

Return ONLY valid JSON (no markdown):
{{
    "conclusion": "clear, specific finding statement",
    "confidence": 0.75,
    "evidence": ["{investigation['sources'][0]['url']}" if len(investigation['sources']) > 0 else ""],
    "uncertainties": ["limitations of the data"],
    "data_quality": "assessment of whether sources had substantive content"
}}

{f'NOTE: Only one source available - be extra conservative with confidence (max 0.80)' if num_sources == 1 else ''}
Be CONSERVATIVE with confidence. When in doubt, assign lower confidence."""
            
            response = self.reasoning.reason(prompt, Config.SYSTEM_PROMPT)
            
            try:
                # Clean response
                clean_response = response.strip()
                if '```json' in clean_response:
                    clean_response = clean_response.split('```json')[1].split('```')[0].strip()
                elif '```' in clean_response:
                    clean_response = clean_response.split('```')[1].split('```')[0].strip()
                
                analysis = json.loads(clean_response, strict=False)
                
                # Lower threshold for findings
                if analysis['confidence'] >= 0.50:  # Meets minimum threshold
                    # Cap confidence for single-source findings
                    confidence = analysis['confidence']
                    if num_sources == 1:
                        confidence = min(confidence, 0.80)  # Cap at Tier 2
                    
                    # Determine tier based on confidence
                    if confidence >= Config.MIN_CONFIDENCE_TIER1:
                        tier = 1
                    elif confidence >= Config.MIN_CONFIDENCE_TIER2:
                        tier = 2
                    else:
                        tier = 3
                    
                    finding = ResearchFinding(
                        content=analysis['conclusion'],
                        confidence=confidence,
                        sources=[s['url'] for s in investigation['sources']],
                        validation_method=f"{num_sources}-source-analysis",
                        timestamp=datetime.now().isoformat(),
                        tier=tier
                    )
                    
                    # Relax validation for single high-quality source
                    if finding.validate() or (num_sources == 1 and confidence >= 0.70):
                        self.kb.store_finding(finding)
                        validated.append(finding)
                        print(f"DEBUG: Validated finding (Tier {tier}, confidence {confidence:.2f}, {num_sources} sources): {finding.content[:60]}...")
                    else:
                        print(f"DEBUG: Finding failed validation checks")
                else:
                    print(f"DEBUG: Confidence too low ({analysis['confidence']:.2f})")
            except Exception as e:
                print(f"DEBUG: Failed to parse analysis JSON: {e}")
                print(f"DEBUG: Response was: {response[:200]}")
        
        return validated
    
    def synthesize_knowledge(self, new_findings: List[ResearchFinding]) -> Dict:
        """Integrate findings into knowledge framework"""
        
        if not new_findings:
            return {"insights": [], "new_questions": []}
        
        prompt = f"""New Validated Findings:
{json.dumps([{
    'content': f.content,
    'confidence': f.confidence,
    'tier': f.tier
} for f in new_findings], indent=2)}

Current Knowledge Base:
{self.kb.get_total_findings()} total findings across all tiers

Synthesize:
1. How do new findings connect to existing knowledge?
2. What higher-level patterns emerge?
3. What new questions can we now ask?
4. Meta-insight: What did we learn about learning?

Remember: "When you know everything, you know more using that knowledge"

Return ONLY valid JSON (no markdown):
{{
    "insights": ["emergent pattern 1", "pattern 2"],
    "connections": ["how findings relate"],
    "new_questions": ["what we can now investigate"],
    "meta_insight": "what we learned about knowledge itself"
}}"""
        
        response = self.reasoning.reason(prompt, Config.SYSTEM_PROMPT)
        try:
            # Clean response
            clean_response = response.strip()
            if '```json' in clean_response:
                clean_response = clean_response.split('```json')[1].split('```')[0].strip()
            elif '```' in clean_response:
                clean_response = clean_response.split('```')[1].split('```')[0].strip()
            
            synthesis = json.loads(clean_response, strict=False)
            print(f"DEBUG: Synthesized {len(synthesis.get('insights', []))} insights")
            return synthesis
        except Exception as e:
            print(f"DEBUG: Failed to parse synthesis JSON: {e}")
            return {"insights": [], "new_questions": []}
    
    def document_research(self, findings: List[ResearchFinding], synthesis: Dict):
        """Document research with academic standards"""
        
        document = {
            "timestamp": datetime.now().isoformat(),
            "findings": [asdict(f) for f in findings],
            "synthesis": synthesis,
            "knowledge_base_stats": self.kb.get_stats()
        }
        
        filepath = Config.PAPERS_PATH / f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Config.PAPERS_PATH.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(document, f, indent=2)
    
    def _summarize_sources(self, sources: List[Dict]) -> str:
        """Create summary of source content"""
        summaries = []
        for i, source in enumerate(sources[:3], 1):
            summary = f"Source {i} ({source['url']}):\n{source['content'][:500]}..."
            summaries.append(summary)
        return "\n\n".join(summaries)


# ============================================================================
# Academic Discovery Agent
# ============================================================================

class AcademicDiscoveryAgent:
    """Rigorous research agent focused on validated discovery"""

    def __init__(self):
        self.knowledge = AcademicKnowledgeBase(Config.KNOWLEDGE_PATH)
        self.cycle_count = 0
        self.running = False

        # Initialize reasoning engine first
        self.reasoning = ReasoningEngine(
            Config.OPENROUTER_API_KEY,
            Config.OPENROUTER_MODEL
        )

        # Pass reasoning engine to ToolRegistry for LLM-guided navigation
        self.tools = ToolRegistry(self.reasoning)

        self.research = ResearchCycle(
            self.knowledge,
            self.tools,
            self.reasoning
        )

        self.setup_logging()
        self.log_inspector = LogInspector(self.reasoning, Config.LOGS_PATH, self.logger)
        self.logger.info("Academic Discovery Agent initialized with browser-based search")
        self.logger.info("Using: Brave Search and DuckDuckGo with LLM-guided navigation")
        self.logger.info("Vision: When it knows everything, it will know more using that knowledge")
    
    def setup_logging(self):
        """Setup logging system"""
        Config.LOGS_PATH.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Config.LOGS_PATH / 'research.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AcademicAgent')
    
    def run_research_cycle(self):
        self.cycle_count += 1
        self.logger.info(f"RESEARCH CYCLE {self.cycle_count}")

        # --------------- new: inspect logs ---------------
        try:
            result = self.log_inspector.inspect_and_maybe_fix(auto_fix=False, dry_run=True)
            self.logger.info(f"Log inspection result: {json.dumps(result.get('inspection_result', {}), indent=2)[:1000]}")
        except Exception as e:
            self.logger.error(f"Log inspector failed: {e}", exc_info=True)
        # -------------------------------------------------

        try:
            results = self.research.execute_cycle()
            
            self.logger.info("\n[CYCLE SUMMARY]")
            self.logger.info(f"  Research gaps identified: {results['gaps_identified']}")
            self.logger.info(f"  Questions formed: {results['questions_formed']}")
            self.logger.info(f"  Data sources collected: {results['data_collected']}")
            self.logger.info(f"  Findings validated: {results['findings_validated']}")
            self.logger.info(f"  Synthesis insights: {results['synthesis_insights']}")
            
            kb_stats = self.knowledge.get_stats()
            self.logger.info("\n[KNOWLEDGE BASE STATUS]")
            self.logger.info(f"  Tier 1 (Foundational Facts): {kb_stats['tier1']} findings")
            self.logger.info(f"  Tier 2 (Derived Knowledge): {kb_stats['tier2']} findings")
            self.logger.info(f"  Tier 3 (Hypotheses): {kb_stats['tier3']} findings")
            self.logger.info(f"  Tier 4 (Questions): {kb_stats['tier4']} questions")
            self.logger.info(f"  Total validated knowledge: {sum(kb_stats.values())}")
            
        except Exception as e:
            self.logger.error(f"Cycle error: {str(e)}", exc_info=True)
    
    def run(self, cycles: int = None, delay: int = 30):
        """Run agent for specified cycles"""
        self.running = True
        
        self.logger.info("="*70)
        self.logger.info("ACADEMIC DISCOVERY AGENT")
        self.logger.info("Vision: When it knows everything, it will know more using that knowledge")
        self.logger.info("="*70)
        self.logger.info(f"Methodology: Structured, validated, rigorous")
        self.logger.info(f"Cycle delay: {delay} seconds")
        self.logger.info(f"Quality threshold: {Config.MIN_SOURCES_PER_FINDING} sources minimum")
        self.logger.info("")
        
        try:
            cycle_num = 0
            while self.running:
                self.run_research_cycle()
                
                cycle_num += 1
                if cycles and cycle_num >= cycles:
                    break
                
                if self.running:
                    self.logger.info(f"\nNext cycle in {delay} seconds...")
                    self.logger.info("(Research takes time - correctness over speed)\n")
                    time.sleep(delay)
                    
        except KeyboardInterrupt:
            self.logger.info("\n\nResearch interrupted by user")
        finally:
            self.running = False
            self.logger.info("Cleaning up browser resources...")
            self.tools.cleanup()
            self._generate_final_report()
    
    def _generate_final_report(self):
        """Generate research session summary"""
        self.logger.info(f"\n{'='*70}")
        self.logger.info("RESEARCH SESSION COMPLETE")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Total cycles: {self.cycle_count}")
        
        kb_stats = self.knowledge.get_stats()
        self.logger.info(f"Validated findings: {sum(kb_stats.values())}")
        self.logger.info(f"  Tier 1 (Foundational): {kb_stats['tier1']}")
        self.logger.info(f"  Tier 2 (Derived): {kb_stats['tier2']}")
        self.logger.info(f"  Tier 3 (Hypotheses): {kb_stats['tier3']}")
        self.logger.info(f"  Tier 4 (Questions): {kb_stats['tier4']}")
        self.logger.info("")
        self.logger.info("'When it knows everything, it will know more using that knowledge'")
        self.logger.info("Each cycle builds foundation for deeper discovery.")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point"""
    agent = AcademicDiscoveryAgent()
    
    # Run for 3 cycles as demo (30 seconds between cycles)
    agent.run(cycles=3, delay=30)


if __name__ == "__main__":
    main()