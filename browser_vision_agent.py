"""
Browser Vision Agent - Chrome automation with computer vision capabilities
Similar to agent-zero's browser_agent but using Playwright and OpenRouter vision models
"""

import json
import base64
import time
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import requests


class VisionReasoningEngine:
    """LLM with vision capabilities for analyzing screenshots"""

    def __init__(self, api_key: str, model: str = "anthropic/claude-3.5-sonnet"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def analyze_screenshot(self, screenshot_b64: str, prompt: str,
                          system_prompt: str = None) -> str:
        """Analyze a screenshot using vision-enabled model"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Build message with image
            user_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot_b64}"
                        }
                    }
                ]
            }
            messages.append(user_message)

            data = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 4000
            }

            response = requests.post(self.base_url, headers=headers, json=data, timeout=120)
            response.raise_for_status()

            return response.json()["choices"][0]["message"]["content"]

        except Exception as e:
            return f"ERROR: Vision analysis failed: {str(e)}"

    def reason_with_text(self, prompt: str, system_prompt: str = None) -> str:
        """Standard text reasoning without vision"""
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
                "max_tokens": 2000
            }

            response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()

            return response.json()["choices"][0]["message"]["content"]

        except Exception as e:
            return f"ERROR: {str(e)}"


class BrowserVisionAgent:
    """
    Browser automation agent using computer vision to understand and interact with web pages
    Similar to agent-zero's implementation but standalone
    """

    def __init__(self, api_key: str, vision_model: str = "anthropic/claude-3.5-sonnet",
                 headless: bool = True, screenshots_dir: str = "./browser_screenshots"):
        self.api_key = api_key
        self.vision_model = vision_model
        self.headless = headless
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        self.vision_engine = VisionReasoningEngine(api_key, vision_model)
        self.browser = None
        self.context = None
        self.page = None

        # System prompt for vision analysis
        self.vision_system_prompt = """You are a browser automation assistant with computer vision capabilities.
You analyze screenshots of web pages to help extract information, identify elements, and understand content.

When analyzing screenshots:
1. Identify key visual elements (buttons, forms, navigation, content areas)
2. Extract text and structured information visible on the page
3. Understand the page layout and purpose
4. Provide actionable insights for data extraction

Be precise and focus on the actual visual content you see."""

    async def start_browser(self):
        """Initialize Playwright browser"""
        try:
            from playwright.async_api import async_playwright

            self.playwright = await async_playwright().start()

            # Launch Chromium (Chrome)
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=['--disable-blink-features=AutomationControlled']
            )

            # Create context with realistic viewport
            self.context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )

            self.page = await self.context.new_page()

            print(f"✓ Browser started (headless={self.headless})")
            return True

        except ImportError:
            print("ERROR: Playwright not installed. Run: pip install playwright && playwright install chromium")
            return False
        except Exception as e:
            print(f"ERROR: Failed to start browser: {e}")
            return False

    async def close_browser(self):
        """Close browser and cleanup"""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if hasattr(self, 'playwright'):
                await self.playwright.stop()
            print("✓ Browser closed")
        except Exception as e:
            print(f"WARNING: Error closing browser: {e}")

    async def navigate(self, url: str, wait_time: int = 2) -> Dict[str, Any]:
        """Navigate to URL and wait for page load"""
        try:
            if not self.page:
                success = await self.start_browser()
                if not success:
                    return {"success": False, "error": "Failed to start browser"}

            print(f"Navigating to: {url}")
            await self.page.goto(url, wait_until='networkidle', timeout=30000)

            # Wait for page to stabilize
            await asyncio.sleep(wait_time)

            title = await self.page.title()
            current_url = self.page.url

            return {
                "success": True,
                "url": current_url,
                "title": title
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def take_screenshot(self, save_name: str = None) -> Dict[str, Any]:
        """Take screenshot and return base64 encoded image"""
        try:
            if not self.page:
                return {"success": False, "error": "No page loaded"}

            # Generate filename
            if not save_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_name = f"screenshot_{timestamp}.png"

            filepath = self.screenshots_dir / save_name

            # Take screenshot
            screenshot_bytes = await self.page.screenshot(full_page=True)

            # Save to file
            with open(filepath, 'wb') as f:
                f.write(screenshot_bytes)

            # Convert to base64
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')

            print(f"✓ Screenshot saved: {filepath}")

            return {
                "success": True,
                "filepath": str(filepath),
                "base64": screenshot_b64,
                "size_bytes": len(screenshot_bytes)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def extract_with_vision(self, url: str, extraction_goal: str) -> Dict[str, Any]:
        """
        Navigate to URL, take screenshot, and use vision to extract information

        Args:
            url: URL to scrape
            extraction_goal: What information to extract (e.g., "Extract all product names and prices")
        """
        try:
            # Navigate
            nav_result = await self.navigate(url)
            if not nav_result['success']:
                return {
                    "success": False,
                    "error": f"Navigation failed: {nav_result.get('error')}"
                }

            # Take screenshot
            screenshot_result = await self.take_screenshot()
            if not screenshot_result['success']:
                return {
                    "success": False,
                    "error": f"Screenshot failed: {screenshot_result.get('error')}"
                }

            # Analyze with vision
            print(f"Analyzing screenshot with vision model...")

            vision_prompt = f"""Analyze this screenshot of a web page.

URL: {url}
Page Title: {nav_result['title']}

EXTRACTION GOAL:
{extraction_goal}

Please analyze the visual content and extract the requested information.
Return the information in a structured format (JSON if applicable).
Focus on what you actually see in the screenshot."""

            analysis = self.vision_engine.analyze_screenshot(
                screenshot_result['base64'],
                vision_prompt,
                self.vision_system_prompt
            )

            # Also get page HTML for reference
            html_content = await self.page.content()

            return {
                "success": True,
                "url": url,
                "title": nav_result['title'],
                "screenshot_path": screenshot_result['filepath'],
                "vision_analysis": analysis,
                "html_length": len(html_content),
                "html_content": html_content[:10000]  # First 10k chars
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def scrape_with_interaction(self, url: str, task_description: str) -> Dict[str, Any]:
        """
        Advanced scraping with interaction - use vision to guide actions

        Args:
            url: Starting URL
            task_description: Description of scraping task (e.g., "Find all research papers on AI")
        """
        try:
            # Navigate
            nav_result = await self.navigate(url)
            if not nav_result['success']:
                return {"success": False, "error": f"Navigation failed: {nav_result.get('error')}"}

            # Take initial screenshot
            screenshot_result = await self.take_screenshot(f"initial_{datetime.now().strftime('%H%M%S')}.png")
            if not screenshot_result['success']:
                return {"success": False, "error": f"Screenshot failed: {screenshot_result.get('error')}"}

            # Plan actions using vision
            planning_prompt = f"""Analyze this screenshot of a web page.

URL: {url}
Task: {task_description}

Based on what you see in the screenshot:
1. What interactive elements are visible? (buttons, forms, links)
2. What actions should be taken to accomplish the task?
3. What information can be extracted from the current view?

Return a JSON response with:
{{
    "visible_elements": ["element descriptions"],
    "recommended_actions": ["action 1", "action 2"],
    "extractable_data": {{"key": "description"}},
    "needs_interaction": true/false
}}"""

            plan = self.vision_engine.analyze_screenshot(
                screenshot_result['base64'],
                planning_prompt,
                self.vision_system_prompt
            )

            # Get current page content
            html_content = await self.page.content()
            page_text = await self.page.inner_text('body')

            return {
                "success": True,
                "url": url,
                "title": nav_result['title'],
                "screenshot_path": screenshot_result['filepath'],
                "action_plan": plan,
                "page_text": page_text[:5000],  # First 5k chars
                "html_length": len(html_content)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def multi_page_scrape(self, urls: List[str], extraction_goal: str) -> List[Dict[str, Any]]:
        """Scrape multiple URLs with vision analysis"""
        results = []

        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}] Processing: {url}")

            result = await self.extract_with_vision(url, extraction_goal)
            results.append(result)

            # Brief delay between pages
            if i < len(urls):
                await asyncio.sleep(2)

        return results


class BrowserVisionTool:
    """Tool wrapper for browser vision agent - compatible with existing tool registry"""

    def __init__(self, api_key: str, vision_model: str = "anthropic/claude-3.5-sonnet"):
        self.api_key = api_key
        self.vision_model = vision_model
        self.name = "BrowserVision"
        self.agent = None

    def execute(self, url: str, extraction_goal: str, **kwargs) -> Dict:
        """Execute browser vision scraping - sync wrapper"""
        try:
            # Run async function in sync context
            return asyncio.run(self._async_execute(url, extraction_goal, **kwargs))
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _async_execute(self, url: str, extraction_goal: str, **kwargs) -> Dict:
        """Async execution"""
        try:
            # Create new agent for this execution
            self.agent = BrowserVisionAgent(
                self.api_key,
                self.vision_model,
                headless=kwargs.get('headless', True)
            )

            # Execute extraction
            result = await self.agent.extract_with_vision(url, extraction_goal)

            # Cleanup
            await self.agent.close_browser()

            return result

        except Exception as e:
            if self.agent:
                await self.agent.close_browser()
            raise e


# Standalone usage example
async def demo_vision_scraping():
    """Demonstration of browser vision capabilities"""
    from academic_agent_starter import Config

    print("=" * 70)
    print("Browser Vision Agent - Demo")
    print("=" * 70)

    # Create agent
    agent = BrowserVisionAgent(
        api_key=Config.OPENROUTER_API_KEY,
        vision_model="anthropic/claude-3.5-sonnet",
        headless=False  # Set to False to see browser
    )

    try:
        # Start browser
        await agent.start_browser()

        # Example 1: Extract information from a research page
        print("\n--- Example 1: Research Paper Extraction ---")
        result1 = await agent.extract_with_vision(
            url="https://arxiv.org/list/cs.AI/recent",
            extraction_goal="Extract titles and authors of the most recent AI research papers visible on this page"
        )

        if result1['success']:
            print(f"\n✓ Successfully analyzed: {result1['title']}")
            print(f"Vision Analysis:\n{result1['vision_analysis'][:500]}...")
            print(f"Screenshot saved: {result1['screenshot_path']}")
        else:
            print(f"✗ Failed: {result1['error']}")

        # Wait a bit
        await asyncio.sleep(3)

        # Example 2: Interactive task planning
        print("\n--- Example 2: Interactive Task Planning ---")
        result2 = await agent.scrape_with_interaction(
            url="https://scholar.google.com",
            task_description="Understand how to search for machine learning papers"
        )

        if result2['success']:
            print(f"\n✓ Task planned for: {result2['title']}")
            print(f"Action Plan:\n{result2['action_plan'][:500]}...")
        else:
            print(f"✗ Failed: {result2['error']}")

    finally:
        # Always close browser
        await agent.close_browser()

    print("\n" + "=" * 70)
    print("Demo Complete")
    print("=" * 70)


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_vision_scraping())
