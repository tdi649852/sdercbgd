# Academic Discovery Agent

An AI-powered academic research agent that systematically researches topics, validates findings, and builds a knowledge base using structured research methodology.

## What's New: Agent0-Style Computer Vision Browser Automation

This version features **Agent0-style computer vision navigation** for intelligent browser control:

### 🎯 **Computer Vision Capabilities**
- **Screenshot Analysis**: Vision-capable LLM (GPT-4o) analyzes screenshots to understand page layout
- **Intelligent Navigation**: AI determines where to click, what to type, and how to extract data
- **Adaptive Control**: Handles dynamic websites and complex interactions automatically
- **No Selectors Needed**: No brittle CSS selectors or XPath expressions

### 🔧 **Dual Navigation Modes**
1. **Vision Mode** (Agent0-style): Uses `browser-use` library with GPT-4o for visual browser control
2. **Pattern Mode** (Fallback): Uses selector patterns when vision is disabled

### ⚡ **Key Features**
- **Local Browser Automation**: Headless Chromium using Playwright
- **Session Persistence**: Saves cookies and state for improved performance
- **Multiple Search Engines**: Brave Search and DuckDuckGo
- **Academic Focus**: Prioritizes scholarly sources automatically

## Architecture

### Browser Infrastructure

1. **ChromiumBrowserManager**: Manages browser lifecycle and session persistence
   - Saves cookies and localStorage between sessions
   - Configurable headless/headful mode
   - Session state stored in `./research_workspace/browser_data/`

2. **Navigation Systems** (Dual Mode):

   **A. VisionBrowserNavigator** (Agent0-style - Default)
   - Uses `browser-use` library for computer vision navigation
   - Takes screenshots and sends to vision LLM (GPT-4o)
   - LLM returns actions: click coordinates, text input, scroll
   - Handles complex page interactions automatically
   - Adapts to any website layout without configuration

   **B. LLMBrowserNavigator** (Pattern-based - Fallback)
   - Uses CSS selector patterns
   - Identifies search inputs and buttons
   - Extracts search results with predefined patterns
   - Faster but less adaptive than vision mode

3. **ReasoningEngine**: Dual-model LLM interface
   - Text reasoning: Claude 3.5 Sonnet (default)
   - Vision reasoning: GPT-4o (for screenshot analysis)
   - Supports OpenRouter API for both models

4. **Search Tools** (Vision-aware):
   - `BraveSearchTool`: Brave Search with optional vision
   - `DuckDuckGoSearchTool`: DuckDuckGo with optional vision
   - `BrowserScraperTool`: URL scraping with optional vision
   - All tools automatically switch between vision/pattern mode

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `playwright` - Browser automation
- `browser-use` - Agent0-style vision navigation
- `pillow` - Image processing
- `requests`, `beautifulsoup4` - HTTP and HTML parsing

### 2. Install Playwright Chromium Browser

```bash
playwright install chromium
```

This downloads the Chromium browser binary for both Playwright and browser-use.

### 3. Configure API Keys

The agent uses OpenRouter API for both text and vision models. Update in `academic_agent_starter.py`:

```python
class Config:
    # Text reasoning model
    OPENROUTER_API_KEY = "your-openrouter-api-key-here"
    OPENROUTER_MODEL = "anthropic/claude-3.5-sonnet"

    # Vision model for browser automation
    VISION_MODEL = "openai/gpt-4o"
    VISION_MODEL_API_KEY = OPENROUTER_API_KEY  # Can use same or different key
    USE_VISION = True  # Enable computer vision
```

**Note**: OpenRouter provides access to both Claude and GPT-4o through a single API. Get your key at [openrouter.ai](https://openrouter.ai/)

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
    # Vision settings (Agent0-style)
    USE_VISION = True  # Enable computer vision navigation
    VISION_MODEL = "openai/gpt-4o"  # Vision-capable model
    VISION_MODEL_API_KEY = OPENROUTER_API_KEY  # API key for vision

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

### Vision vs Pattern Mode

**When to use Vision Mode** (Default):
- ✅ Complex or dynamic websites
- ✅ Sites that frequently change their HTML structure
- ✅ JavaScript-heavy single-page applications
- ✅ Maximum adaptability and reliability
- ❌ Slower (requires screenshot analysis)
- ❌ Higher token costs (vision LLM)

**When to use Pattern Mode**:
```python
Config.USE_VISION = False
```
- ✅ Simple, static websites
- ✅ Faster execution
- ✅ Lower API costs
- ❌ Less reliable on complex sites
- ❌ Requires selector updates if sites change

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

### Vision-Guided Navigation (Agent0-style)

When `USE_VISION = True` (default), the system uses computer vision for browser control:

1. **Screenshot Capture**: Browser takes screenshot of current page
2. **Vision Analysis**: GPT-4o analyzes the image:
   - "Where is the search box?"
   - "What should I click to submit?"
   - "Extract all search results from this page"
3. **Action Execution**: Vision LLM returns coordinates/actions:
   ```json
   {
     "action": "click",
     "x": 450,
     "y": 120,
     "reason": "Search input field"
   }
   ```
4. **Iterative Process**: Repeats until task completed
5. **No Selectors**: Works on any website layout automatically

### Research Cycle

1. **Gap Identification**: Analyzes current knowledge to find research gaps
2. **Question Formation**: Creates specific, investigable research questions
3. **Data Collection** (Vision-powered):
   - Opens headless Chromium browser
   - Vision LLM navigates to search engine (analyzing screenshots)
   - AI determines where search box is and types query
   - AI identifies and clicks search button
   - AI extracts search results by analyzing result page
   - Scrapes content from top results (vision-guided)
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

### Vision Mode Issues

**"browser-use not installed" warning:**
```bash
pip install browser-use
```

**Vision navigation is slow:**
- Expected: Vision mode takes 2-3x longer than pattern mode
- Each screenshot analysis requires vision LLM call
- Consider using pattern mode for simple sites: `USE_VISION = False`

**Vision model errors:**
- Check OpenRouter API key is valid
- Verify GPT-4o is available on your plan
- Try alternative vision model: `VISION_MODEL = "anthropic/claude-3-5-sonnet"`

**Vision not working, falling back to pattern mode:**
- Check debug output: "WARNING: Vision requested but browser-use not available"
- Install browser-use: `pip install browser-use`
- Restart the agent after installation

### General Browser Issues

**Browser won't start:**
```bash
# Reinstall Chromium
playwright install chromium

# Check Playwright installation
playwright --version
```

**Search input not found (Pattern Mode):**
- Enable vision mode: `USE_VISION = True`
- Or try headful browser: `BROWSER_HEADLESS = False`
- Check if search engine changed their HTML structure
- Update selectors in `_find_search_input()` method

**Content extraction fails:**
- Enable vision mode for better adaptability
- Verify URLs are accessible
- Check if sites block automated browsers
- Adjust timeout settings in Config

### Debug Output

**Vision Mode:**
```
DEBUG: Using computer vision for Brave search
DEBUG: Running vision agent for Brave search...
DEBUG: Vision-based Brave search found 5 results
```

**Pattern Mode:**
```
DEBUG: Found search input with selector: input[name="q"]
DEBUG: Found 10 results with selector: div.snippet
DEBUG: Brave found 5 results with content
```

## Performance

### Vision Mode (Default)
- **First search**: ~15-25 seconds (browser init + vision analysis)
- **Subsequent searches**: ~10-15 seconds (cached session + vision)
- **Content scraping**: ~5-8 seconds per URL (vision-guided)
- **Full research cycle**: ~5-8 minutes (vision overhead + quality)
- **Token usage**: Higher (vision models use more tokens)

### Pattern Mode (`USE_VISION = False`)
- **First search**: ~8-12 seconds (browser init + pattern matching)
- **Subsequent searches**: ~4-6 seconds (cached session)
- **Content scraping**: ~2-3 seconds per URL
- **Full research cycle**: ~3-5 minutes
- **Token usage**: Lower (text-only LLM)

**Trade-off**: Vision mode is slower but significantly more reliable on complex/dynamic sites.

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
