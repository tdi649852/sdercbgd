"""
Integrated Example: Academic Agent with Browser Vision

This shows how to use the browser vision tool integrated
with the existing academic agent's tool registry.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from academic_agent_starter import Config, ToolRegistry


def test_browser_vision_tool():
    """Test the browser vision tool through the tool registry"""

    print("=" * 70)
    print("Testing Browser Vision Tool via Tool Registry")
    print("=" * 70)
    print()

    # Create tool registry (will auto-load browser vision tool)
    print("Initializing tool registry...")
    tools = ToolRegistry()

    print(f"Available tools: {', '.join(tools.list_tools())}")
    print()

    # Check if browser vision is available
    if 'BrowserVision' not in tools.list_tools():
        print("✗ Browser Vision tool not loaded!")
        print("Make sure browser_vision_agent.py is in the same directory")
        return

    print("✓ Browser Vision tool is loaded\n")

    # Use the tool through the registry
    print("Executing browser vision scraping...")
    print("-" * 70)

    url = "https://arxiv.org/list/cs.AI/recent"
    extraction_goal = """
    Extract information about recent AI research papers:
    - Identify paper titles
    - Find author names
    - Note submission dates
    - Identify any subject categories

    Provide a structured summary of what you find.
    """

    print(f"URL: {url}")
    print(f"Goal: {extraction_goal.strip()}\n")

    # Execute through tool registry
    result = tools.execute_tool(
        'BrowserVision',
        url=url,
        extraction_goal=extraction_goal,
        headless=True
    )

    print("-" * 70)
    print("RESULTS:")
    print("-" * 70)

    if result.get('success'):
        print(f"✓ Scraping successful!")
        print()
        print(f"URL: {result['url']}")
        print(f"Title: {result['title']}")
        print(f"Screenshot: {result['screenshot_path']}")
        print()
        print("Vision Analysis:")
        print("-" * 70)
        print(result['vision_analysis'])
        print("-" * 70)
    else:
        print(f"✗ Scraping failed: {result.get('error')}")

    print()
    print("Tool usage statistics:")
    print(tools.usage_stats)

    print()
    print("=" * 70)


def compare_scraping_methods():
    """Compare traditional scraping vs vision scraping"""

    print("=" * 70)
    print("Comparing Scraping Methods")
    print("=" * 70)
    print()

    tools = ToolRegistry()

    test_url = "https://scholar.google.com"

    # Method 1: Traditional Firecrawl scraping
    print("1. Traditional Scraping (Firecrawl)")
    print("-" * 70)

    firecrawl_result = tools.execute_tool(
        'FirecrawlScraper',
        url=test_url
    )

    if firecrawl_result.get('success'):
        print(f"✓ Success")
        print(f"Content length: {firecrawl_result.get('length', 0)} characters")
        print(f"Preview: {firecrawl_result.get('text', '')[:200]}...")
    else:
        print(f"✗ Failed: {firecrawl_result.get('error')}")

    print()

    # Method 2: Vision-based scraping
    print("2. Vision-Based Scraping (Browser Vision)")
    print("-" * 70)

    if 'BrowserVision' in tools.list_tools():
        vision_result = tools.execute_tool(
            'BrowserVision',
            url=test_url,
            extraction_goal="Analyze this page and describe what you see. What is the main functionality? What elements are visible?",
            headless=True
        )

        if vision_result.get('success'):
            print(f"✓ Success")
            print(f"Vision Analysis Preview:")
            print(vision_result.get('vision_analysis', '')[:300])
            print()
            print(f"Screenshot: {vision_result.get('screenshot_path')}")
        else:
            print(f"✗ Failed: {vision_result.get('error')}")
    else:
        print("✗ Browser Vision not available")

    print()
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print()
    print("Traditional Scraping (Firecrawl):")
    print("  + Fast and efficient")
    print("  + Good for text extraction")
    print("  + Works well with clean HTML")
    print("  - May miss dynamic content")
    print("  - Limited understanding of visual layout")
    print()
    print("Vision-Based Scraping:")
    print("  + Understands visual layout and design")
    print("  + Can handle dynamic/JavaScript-heavy pages")
    print("  + Better for complex visual information")
    print("  + Mimics human understanding")
    print("  - Slower (screenshots + vision analysis)")
    print("  - Higher API costs (vision models)")
    print()
    print("Recommendation: Use vision scraping when:")
    print("  - Page layout is important")
    print("  - Content is visually structured (tables, cards, etc.)")
    print("  - Traditional scraping misses important context")
    print("  - You need to understand visual relationships")
    print()


def main():
    """Run all integrated examples"""

    # Test 1: Basic tool usage
    test_browser_vision_tool()

    print("\n\n")

    # Test 2: Compare methods
    compare_scraping_methods()

    print()
    print("=" * 70)
    print("Integration Examples Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
