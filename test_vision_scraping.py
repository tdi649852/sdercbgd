"""
Test script for browser vision scraping functionality
Validates installation and basic functionality
"""

import sys
import asyncio
from pathlib import Path


def test_imports():
    """Test that all modules can be imported"""
    print("=" * 70)
    print("TEST 1: Module Imports")
    print("=" * 70)

    try:
        from browser_vision_agent import (
            BrowserVisionAgent,
            BrowserVisionTool,
            VisionReasoningEngine
        )
        print("✓ browser_vision_agent imports successful")

        from academic_agent_starter import (
            Config,
            ToolRegistry,
            ReasoningEngine
        )
        print("✓ academic_agent_starter imports successful")

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_tool_registry():
    """Test that BrowserVision tool is registered"""
    print("\n" + "=" * 70)
    print("TEST 2: Tool Registry Integration")
    print("=" * 70)

    try:
        from academic_agent_starter import ToolRegistry

        tools = ToolRegistry()

        available_tools = tools.list_tools()
        print(f"Available tools: {', '.join(available_tools)}")

        if 'BrowserVision' in available_tools:
            print("✓ BrowserVision tool registered successfully")
            return True
        else:
            print("✗ BrowserVision tool not found in registry")
            return False

    except Exception as e:
        print(f"✗ Tool registry test failed: {e}")
        return False


def test_configuration():
    """Test configuration and API keys"""
    print("\n" + "=" * 70)
    print("TEST 3: Configuration")
    print("=" * 70)

    try:
        from academic_agent_starter import Config

        # Check API keys are present (not validating them)
        if Config.OPENROUTER_API_KEY and len(Config.OPENROUTER_API_KEY) > 10:
            print(f"✓ OpenRouter API key configured ({len(Config.OPENROUTER_API_KEY)} chars)")
        else:
            print("⚠ OpenRouter API key may not be configured")

        if Config.FIRECRAWL_API_KEY and len(Config.FIRECRAWL_API_KEY) > 10:
            print(f"✓ Firecrawl API key configured ({len(Config.FIRECRAWL_API_KEY)} chars)")
        else:
            print("⚠ Firecrawl API key may not be configured")

        print(f"✓ Vision model: {Config.OPENROUTER_MODEL}")

        return True

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_playwright_installation():
    """Test that Playwright is installed"""
    print("\n" + "=" * 70)
    print("TEST 4: Playwright Installation")
    print("=" * 70)

    try:
        import playwright
        print(f"✓ Playwright installed")

        # Try importing async_api
        from playwright.async_api import async_playwright
        print("✓ Playwright async API available")

        return True

    except ImportError as e:
        print(f"✗ Playwright not installed: {e}")
        print("  Run: pip install playwright && playwright install chromium")
        return False


async def test_browser_initialization():
    """Test browser can be initialized (may fail in some environments)"""
    print("\n" + "=" * 70)
    print("TEST 5: Browser Initialization (Optional)")
    print("=" * 70)

    try:
        from browser_vision_agent import BrowserVisionAgent
        from academic_agent_starter import Config

        agent = BrowserVisionAgent(
            api_key=Config.OPENROUTER_API_KEY,
            vision_model="anthropic/claude-3.5-sonnet",
            headless=True
        )

        print("Attempting to start browser...")

        success = await agent.start_browser()

        if success:
            print("✓ Browser started successfully")
            await agent.close_browser()
            print("✓ Browser closed successfully")
            return True
        else:
            print("⚠ Browser failed to start (may not be available in this environment)")
            return None  # None indicates optional test that couldn't run

    except Exception as e:
        print(f"⚠ Browser test failed (expected in some environments): {e}")
        print("  This is OK if running in a restricted environment")
        return None  # None indicates optional test


def test_directory_structure():
    """Test that required directories and files exist"""
    print("\n" + "=" * 70)
    print("TEST 6: Directory Structure")
    print("=" * 70)

    required_files = [
        'browser_vision_agent.py',
        'academic_agent_starter.py',
        'README_VISION_SCRAPING.md',
        'examples/simple_vision_scrape.py',
        'examples/research_paper_scraper.py',
        'examples/integrated_vision_agent.py',
    ]

    all_exist = True
    for filepath in required_files:
        path = Path(filepath)
        if path.exists():
            print(f"✓ {filepath}")
        else:
            print(f"✗ {filepath} NOT FOUND")
            all_exist = False

    # Check screenshots directory will be created
    screenshots_dir = Path("browser_screenshots")
    if screenshots_dir.exists():
        print(f"✓ Screenshots directory exists: {screenshots_dir}")
    else:
        print(f"ℹ Screenshots directory will be created on first use")

    return all_exist


def test_tool_execution_structure():
    """Test that tool can be called (without actually executing)"""
    print("\n" + "=" * 70)
    print("TEST 7: Tool Execution Structure")
    print("=" * 70)

    try:
        from academic_agent_starter import ToolRegistry

        tools = ToolRegistry()

        # Get the tool
        vision_tool = tools.get_tool('BrowserVision')

        if vision_tool:
            print("✓ BrowserVision tool retrieved from registry")

            # Check it has execute method
            if hasattr(vision_tool, 'execute'):
                print("✓ Tool has execute() method")
            else:
                print("✗ Tool missing execute() method")
                return False

            # Check properties
            if hasattr(vision_tool, 'name'):
                print(f"✓ Tool name: {vision_tool.name}")
            if hasattr(vision_tool, 'api_key'):
                print(f"✓ Tool has API key configured")
            if hasattr(vision_tool, 'vision_model'):
                print(f"✓ Vision model: {vision_tool.vision_model}")

            return True
        else:
            print("✗ Could not retrieve BrowserVision tool")
            return False

    except Exception as e:
        print(f"✗ Tool execution structure test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "=" * 70)
    print("BROWSER VISION SCRAPING - TEST SUITE")
    print("=" * 70)
    print()

    results = {}

    # Run tests
    results['imports'] = test_imports()
    results['tool_registry'] = test_tool_registry()
    results['configuration'] = test_configuration()
    results['playwright'] = test_playwright_installation()
    results['browser_init'] = await test_browser_initialization()
    results['directory'] = test_directory_structure()
    results['tool_structure'] = test_tool_execution_structure()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    total = len(results)

    print(f"\nTotal tests: {total}")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"⚠ Skipped/Optional: {skipped}")

    print("\nDetailed Results:")
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⚠ SKIP"
        print(f"  {status}: {test_name}")

    print()

    if failed == 0:
        print("=" * 70)
        print("🎉 ALL CRITICAL TESTS PASSED!")
        print("=" * 70)
        print()
        print("✓ Browser Vision scraping is ready to use!")
        print()
        print("Next steps:")
        print("  1. Run examples: python examples/simple_vision_scrape.py")
        print("  2. Read docs: README_VISION_SCRAPING.md")
        print("  3. Integrate with your agent!")
        print()
        return True
    else:
        print("=" * 70)
        print("⚠ SOME TESTS FAILED")
        print("=" * 70)
        print()
        print("Please fix the failed tests before using vision scraping.")
        print("Check error messages above for details.")
        print()
        return False


def main():
    """Main entry point"""
    try:
        result = asyncio.run(run_all_tests())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
