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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import requests

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
# Foundation Tools
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
    
    def __init__(self):
        self.tools = {}
        self.usage_stats = {}
        self.load_foundation_tools()
        
    def load_foundation_tools(self):
        """Load core research tools"""
        firecrawl = FirecrawlSearchTool(Config.FIRECRAWL_API_KEY)
        scraper = FirecrawlScraperTool(Config.FIRECRAWL_API_KEY)
        
        self.register_tool(firecrawl.name, firecrawl)
        self.register_tool(scraper.name, scraper)
        
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
        self.tools = ToolRegistry()
        self.cycle_count = 0
        self.running = False
        self.reasoning = ReasoningEngine(
            Config.OPENROUTER_API_KEY,
            Config.OPENROUTER_MODEL
        )
        self.research = ResearchCycle(
            self.knowledge,
            self.tools,
            self.reasoning
        )
        
        self.setup_logging()
        self.log_inspector = LogInspector(self.reasoning, Config.LOGS_PATH, self.logger)
        self.logger.info("Academic Discovery Agent initialized")
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