"""
Simple Example: Using Browser Vision to scrape a webpage

This demonstrates basic usage of the browser vision agent to extract
information from a webpage using computer vision capabilities.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from browser_vision_agent import BrowserVisionAgent
from academic_agent_starter import Config


async def main():
    """Simple demonstration of vision-based web scraping"""

    print("=" * 70)
    print("Simple Browser Vision Scraping Example")
    print("=" * 70)
    print()

    # Create the browser vision agent
    agent = BrowserVisionAgent(
        api_key=Config.OPENROUTER_API_KEY,
        vision_model="anthropic/claude-3.5-sonnet",
        headless=True  # Run without showing browser window
    )

    try:
        # Start the browser
        print("Starting browser...")
        await agent.start_browser()
        print("✓ Browser started\n")

        # Example: Extract information from a research page
        url = "https://scholar.google.com"
        extraction_goal = """
        Analyze this page and extract:
        1. What is the main purpose of this website?
        2. What search functionality is available?
        3. What categories or topics are visible?
        4. Any featured or highlighted content

        Provide a structured summary of what you observe.
        """

        print(f"Scraping: {url}")
        print(f"Goal: {extraction_goal.strip()}\n")

        result = await agent.extract_with_vision(url, extraction_goal)

        if result['success']:
            print("=" * 70)
            print("RESULTS")
            print("=" * 70)
            print(f"URL: {result['url']}")
            print(f"Title: {result['title']}")
            print(f"Screenshot: {result['screenshot_path']}")
            print()
            print("Vision Analysis:")
            print("-" * 70)
            print(result['vision_analysis'])
            print("-" * 70)
            print()
            print(f"HTML Content Length: {result['html_length']} characters")
            print()
            print("✓ Scraping successful!")
        else:
            print(f"✗ Scraping failed: {result.get('error')}")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Always close the browser
        print("\nClosing browser...")
        await agent.close_browser()
        print("✓ Browser closed")

    print()
    print("=" * 70)
    print("Example Complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
