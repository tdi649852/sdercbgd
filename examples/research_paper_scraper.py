"""
Advanced Example: Research Paper Scraping with Vision

This demonstrates using browser vision to scrape research papers
from multiple sources, identifying key information visually.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from browser_vision_agent import BrowserVisionAgent
from academic_agent_starter import Config


async def scrape_research_papers():
    """Scrape research papers using computer vision"""

    print("=" * 70)
    print("Research Paper Vision Scraper")
    print("=" * 70)
    print()

    # Create agent
    agent = BrowserVisionAgent(
        api_key=Config.OPENROUTER_API_KEY,
        vision_model="anthropic/claude-3.5-sonnet",
        headless=True
    )

    results = []

    try:
        await agent.start_browser()
        print("✓ Browser started\n")

        # List of research sources to scrape
        sources = [
            {
                "url": "https://arxiv.org/list/cs.AI/recent",
                "goal": "Extract the titles, authors, and abstracts of recent AI papers visible on this page. Format as a structured list."
            },
            {
                "url": "https://paperswithcode.com/latest",
                "goal": "Identify recent machine learning papers. Extract paper titles, any visible metrics (stars, citations), and topics/categories."
            },
            {
                "url": "https://openreview.net/",
                "goal": "Analyze the homepage. What conferences or venues are featured? What papers or topics are highlighted?"
            }
        ]

        # Process each source
        for i, source in enumerate(sources, 1):
            print(f"\n[{i}/{len(sources)}] Processing: {source['url']}")
            print(f"Goal: {source['goal'][:80]}...")

            try:
                result = await agent.extract_with_vision(
                    url=source['url'],
                    extraction_goal=source['goal']
                )

                if result['success']:
                    print(f"  ✓ Success - Screenshot: {result['screenshot_path']}")

                    results.append({
                        "source": source['url'],
                        "title": result['title'],
                        "extraction": result['vision_analysis'],
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    print(f"  ✗ Failed: {result.get('error')}")
                    results.append({
                        "source": source['url'],
                        "error": result.get('error'),
                        "timestamp": datetime.now().isoformat()
                    })

                # Brief delay between requests
                if i < len(sources):
                    print("  Waiting 3 seconds before next source...")
                    await asyncio.sleep(3)

            except Exception as e:
                print(f"  ✗ Error: {e}")
                results.append({
                    "source": source['url'],
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

    finally:
        await agent.close_browser()
        print("\n✓ Browser closed")

    # Save results
    output_dir = Path("./research_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"vision_scrape_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 70)
    print("SCRAPING COMPLETE")
    print("=" * 70)
    print(f"Processed: {len(sources)} sources")
    print(f"Successful: {sum(1 for r in results if 'extraction' in r)}")
    print(f"Failed: {sum(1 for r in results if 'error' in r)}")
    print(f"Results saved to: {output_file}")
    print()

    # Display summary of results
    print("SUMMARY OF EXTRACTED DATA:")
    print("-" * 70)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.get('title', 'Unknown Title')}")
        print(f"   Source: {result['source']}")
        if 'extraction' in result:
            # Show first 200 characters of extraction
            preview = result['extraction'][:200].replace('\n', ' ')
            print(f"   Preview: {preview}...")
        elif 'error' in result:
            print(f"   Error: {result['error']}")

    print()
    print("=" * 70)


async def scrape_single_paper_details():
    """Example: Deep dive into a single paper page"""

    print("\n" + "=" * 70)
    print("Single Paper Deep Analysis")
    print("=" * 70)
    print()

    agent = BrowserVisionAgent(
        api_key=Config.OPENROUTER_API_KEY,
        vision_model="anthropic/claude-3.5-sonnet",
        headless=True
    )

    try:
        await agent.start_browser()

        # Example arxiv paper
        paper_url = "https://arxiv.org/abs/2303.08774"  # GPT-4 paper

        extraction_goal = """
        Analyze this research paper page in detail:
        1. Paper Title
        2. Authors and their affiliations
        3. Abstract (full text)
        4. Submission date and any version information
        5. Categories/subjects
        6. Any comments or notes
        7. Available download formats

        Extract all visible metadata and content.
        """

        print(f"Analyzing paper: {paper_url}\n")

        result = await agent.extract_with_vision(paper_url, extraction_goal)

        if result['success']:
            print("=" * 70)
            print("PAPER ANALYSIS")
            print("=" * 70)
            print(result['vision_analysis'])
            print()
            print(f"Screenshot saved: {result['screenshot_path']}")
        else:
            print(f"Failed: {result.get('error')}")

    finally:
        await agent.close_browser()


async def main():
    """Run all examples"""

    print("=" * 70)
    print("Browser Vision Research Paper Scraper")
    print("Demonstrates computer vision for academic knowledge extraction")
    print("=" * 70)
    print()

    # Example 1: Multi-source scraping
    await scrape_research_papers()

    # Wait a bit
    await asyncio.sleep(2)

    # Example 2: Single paper analysis
    await scrape_single_paper_details()

    print("\n" + "=" * 70)
    print("All Examples Complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
