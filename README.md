# Academic Discovery Agent

An AI-powered academic research agent that systematically researches topics, validates findings, and builds a knowledge base using structured research methodology.

## What's New: Browser-Based Search with LLM Navigation

This version replaces the Firecrawl API with **local headless Chromium browser automation** featuring:

- **LLM-Guided Navigation**: Uses Claude to intelligently navigate web pages and extract content
- **Session Persistence**: Saves cookies and session state for improved performance
- **Multiple Search Engines**:
  - Brave Search (https://search.brave.com/?lang=en-in)
  - DuckDuckGo (https://duckduckgo.com/)
- **Intelligent Content Extraction**: LLM analyzes page structure to find and extract relevant content
- **No API Dependencies**: Fully local browser automation using Playwright

## Architecture

### Browser Infrastructure

1. **ChromiumBrowserManager**: Manages browser lifecycle and session persistence
   - Saves cookies and localStorage between sessions
   - Configurable headless/headful mode
   - Session state stored in `./research_workspace/browser_data/`

2. **LLMBrowserNavigator**: LLM-powered page navigation
   - Identifies search inputs and buttons
   - Extracts search results intelligently
   - Scrapes main content from pages
   - Adapts to different page structures

3. **Search Tools**:
   - `BraveSearchTool`: Brave Search integration
   - `DuckDuckGoSearchTool`: DuckDuckGo integration
   - `BrowserScraperTool`: Individual URL scraping

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Playwright Chromium Browser

```bash
playwright install chromium
```

This downloads the Chromium browser binary that Playwright will use.

### 3. Configure API Keys

The agent uses OpenRouter API for LLM reasoning. Update the API key in `academic_agent_starter.py`:

```python
class Config:
    OPENROUTER_API_KEY = "your-openrouter-api-key-here"
    OPENROUTER_MODEL = "anthropic/claude-3.5-sonnet"
```

## Usage

### Basic Usage

Run the agent for 3 research cycles:

```bash
python academic_agent_starter.py
```

### Configuration

Edit `Config` class in `academic_agent_starter.py`:

```python
class Config:
    # Browser settings
    BROWSER_HEADLESS = True  # Set to False to see browser UI
    BROWSER_TIMEOUT = 60000  # Page load timeout (ms)

    # Search engines
    BRAVE_SEARCH_URL = "https://search.brave.com/?lang=en-in"
    DUCKDUCKGO_SEARCH_URL = "https://duckduckgo.com/"

    # Research parameters
    CYCLE_DELAY = 30  # Seconds between research cycles
    MIN_SOURCES_PER_FINDING = 2  # Minimum sources for validation
    CONTENT_MIN_LENGTH = 1000  # Minimum content length (chars)
```

### Switching Search Engines

By default, the agent uses Brave Search. You can switch to DuckDuckGo by modifying `investigate_questions()` in the `ResearchCycle` class:

```python
# Change from:
search_result = self.tools.execute_tool('FirecrawlSearch', ...)

# To:
search_result = self.tools.execute_tool('DuckDuckGoSearch', ...)
```

Or use both engines for comprehensive research:

```python
# Try Brave first
brave_result = self.tools.execute_tool('BraveSearch', ...)

# If needed, also try DuckDuckGo
if len(brave_result['results']) < 5:
    ddg_result = self.tools.execute_tool('DuckDuckGoSearch', ...)
```

## How It Works

### Research Cycle

1. **Gap Identification**: Analyzes current knowledge to find research gaps
2. **Question Formation**: Creates specific, investigable research questions
3. **Data Collection**:
   - Opens headless Chromium browser
   - Navigates to search engine
   - LLM identifies search input and enters query
   - Extracts search results
   - Scrapes content from top results
   - Saves session cookies
4. **Analysis & Validation**: LLM analyzes sources for quality and validity
5. **Knowledge Storage**: Stores validated findings in tiered knowledge base
6. **Synthesis**: Identifies patterns and new questions

### Session Persistence

The browser saves cookies and session state to:
```
./research_workspace/browser_data/session_state.json
```

This allows:
- Faster subsequent searches
- Preserved preferences
- Reduced bot detection

### LLM-Guided Navigation

The `LLMBrowserNavigator` uses pattern matching and fallback strategies to:

1. **Find Search Input**: Tries common selectors for each search engine, falls back to first visible input
2. **Submit Search**: Presses Enter or finds submit button
3. **Extract Results**: Uses engine-specific selectors to extract titles, URLs, and snippets
4. **Scrape Content**: Navigates to each result and extracts main content

## Directory Structure

```
research_workspace/
├── browser_data/          # Browser sessions and cookies
│   └── session_state.json
├── knowledge/             # Validated knowledge base
│   ├── tier1_foundational/
│   ├── tier2_derived/
│   ├── tier3_hypotheses/
│   └── tier4_questions/
├── logs/                  # Research logs
│   └── research.log
└── research_papers/       # Research output documents
```

## Debugging

### Enable Visual Browser Mode

Set `BROWSER_HEADLESS = False` to see the browser in action:

```python
class Config:
    BROWSER_HEADLESS = False
```

### Check Debug Output

The agent prints detailed debug information:

```
DEBUG: Browser initialized with session persistence at ./research_workspace/browser_data
DEBUG: Found search input with selector: input[name="q"]
DEBUG: Found 10 results with selector: div.snippet
DEBUG: Extracted 8 search results
DEBUG: Brave found 5 results with content
```

### Session State

Inspect saved session:

```bash
cat research_workspace/browser_data/session_state.json
```

## Troubleshooting

### Browser won't start

```bash
# Reinstall Chromium
playwright install chromium

# Check Playwright installation
playwright --version
```

### Search input not found

- Try setting `BROWSER_HEADLESS = False` to see the page
- Check if search engine changed their HTML structure
- Update selectors in `_find_search_input()` method

### Content extraction fails

- Verify URLs are accessible
- Check if sites block automated browsers
- Adjust timeout settings in Config

## Performance

- **First search**: ~10-15 seconds (including browser initialization)
- **Subsequent searches**: ~5-8 seconds (using cached session)
- **Content scraping**: ~2-3 seconds per URL
- **Full research cycle**: ~3-5 minutes (depending on number of sources)

## Advanced Features

### Custom Search Engine

Add your own search engine:

```python
class CustomSearchTool:
    def __init__(self, browser_manager, navigator):
        self.browser_manager = browser_manager
        self.navigator = navigator
        self.name = "CustomSearch"

    def execute(self, query: str, count: int = 10) -> Dict:
        return asyncio.run(self._async_execute(query, count))

    async def _async_execute(self, query: str, count: int) -> Dict:
        # Your implementation here
        pass

# Register in ToolRegistry
self.register_tool("CustomSearch", custom_search)
```

### Research Papers Access

For accessing research papers behind paywalls or requiring login:

1. Set `BROWSER_HEADLESS = False`
2. Manually log in to the required site
3. Session cookies will be saved automatically
4. Future searches will use authenticated session

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please ensure:
- Browser automation remains local (no cloud dependencies)
- Session persistence is maintained
- LLM navigation logic is well-documented
- Debug output is comprehensive

## Support

For issues or questions:
1. Check debug output in logs
2. Try with `BROWSER_HEADLESS = False`
3. Verify Playwright installation
4. Check search engine accessibility

---

**Vision**: "When it knows everything, it will know more using that knowledge"

Built with structured academic methodology, prioritizing correctness over speed.
