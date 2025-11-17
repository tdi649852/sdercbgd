# log_inspector.py
import re
import time
import json
import shlex
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

from datetime import datetime

# import agent Config and ReasoningEngine from your existing module
# adjust imports if you put this inside academic_agent_starter.py
from academic_agent_starter import Config, ReasoningEngine

LOG_READ_BYTES = 20000  # number of chars to read from end of log

class LogInspector:
    """
    Inspect terminal / application logs and use LLM reasoning to detect errors
    and propose fixes. Optionally executes safe fixes from a whitelist.
    """

    ERROR_PATTERNS = [
        r"Traceback \(most recent call last\):",
        r"Exception:",
        r"ERROR:",
        r"CRITICAL:",
        r"panic:",
        r"segmentation fault",
        r"Stacktrace:",
        r"fatal:",
    ]

    # Commands that are allowed for auto-fix (example set). Customize carefully.
    # Only exact commands in this list will be executed when auto_fix=True.
    SAFE_COMMAND_WHITELIST = {
        # pip installs (package name only, executed as: pip install <pkg>)
        "pip install": ["requests", "beautifulsoup4", "bs4"],
        # systemctl restart allowed services (service names)
        "systemctl restart": ["nginx", "redis", "postgresql"],
        # apt-get update/upgrade (non-destructive, apt-get upgrade still a change - use with caution)
        # for demonstration; you may remove these in production
        "apt-get update": [],
    }

    def __init__(self, reasoning: ReasoningEngine, logs_path: Path = None, logger=None):
        self.reasoning = reasoning
        self.logs_path = logs_path or Config.LOGS_PATH
        self.logger = logger
        self.last_inspect_time = None

    def _tail_log_file(self, logfile: Path, max_chars: int = LOG_READ_BYTES) -> str:
        """Read tail of a logfile (works for files that may be large)."""
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
        """Collect latest data from all .log files in logs_path and return mapping."""
        logs = {}
        if not self.logs_path.exists():
            return logs
        for p in sorted(self.logs_path.glob("*.log")):
            logs[p.name] = self._tail_log_file(p)
        return logs

    def _extract_error_snippets(self, text: str, max_snippets: int = 5) -> List[str]:
        """Extract likely error blocks (stack trace or nearby lines)."""
        if not text:
            return []
        snippets = []
        # Search for error anchors
        for pattern in self.ERROR_PATTERNS:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                # capture 800 chars before and 1200 chars after for context (tunable)
                start = max(0, m.start() - 800)
                end = min(len(text), m.end() + 1200)
                snippet = text[start:end].strip()
                snippets.append(snippet)
                if len(snippets) >= max_snippets:
                    return snippets

        # fallback: look for lines containing "ERROR" or "Exception"
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if "error" in line.lower() or "exception" in line.lower() or "traceback" in line.lower():
                start = max(0, i - 8)
                end = min(len(lines), i + 12)
                snippet = "\n".join(lines[start:end])
                snippets.append(snippet)
                if len(snippets) >= max_snippets:
                    break

        return snippets

    def _build_prompt(self, snippet_map: Dict[str, str]) -> str:
        """
        Build a succinct prompt for the LLM. Keep the prompt structured and ask for JSON output.
        """
        # Prepare compact representation of logs
        summary_parts = []
        for fname, content in snippet_map.items():
            truncated = content.strip()
            if len(truncated) > 3000:
                truncated = truncated[-3000:]  # keep tail where stack traces often appear
            summary_parts.append(f"=== FILE: {fname} ===\n{truncated}\n")

        prompt = f"""
You are a devops/engineer assistant. I will feed you recent log snippets from an application.
Task:
1) Decide if there is an active error or failure that needs attention.
2) If yes, classify the problem (short label), give a 1-2 sentence root-cause hypothesis, list concrete, ordered resolution steps (commands to run or code changes), and an estimated confidence (0.0-1.0).
3) If no issue detected, respond with "no_issue".
Output: Return ONLY JSON, no prose, exactly matching this schema:

{{
  "issue_detected": true|false,
  "short_label": "single short phrase or EMPTY if none",
  "root_cause": "one sentence hypothesis or EMPTY",
  "resolution_steps": ["step 1 (human-readable)","step 2", ...],
  "shell_commands": ["exact shell command 1", "exact shell command 2", ...],
  "confidence": 0.0,
  "notes": "any short note"
}}

Context:
- Use the logs provided below to make your determination.
- Avoid speculative unrelated fixes; target the specific signals in the logs.
- Only include shell commands that are reasonable given the logs (do not include destructive commands like rm -rf).
- If you cannot identify a problem, set issue_detected=false.

Logs:
----------------------------------------
{chr(10).join(summary_parts)}
----------------------------------------
"""
        return prompt

    def inspect_and_maybe_fix(self, auto_fix: bool = False, dry_run: bool = True) -> Dict:
        """
        Inspect logs, ask the LLM to reason, and (optionally) run whitelisted fixes.

        Returns a dict with: inspection_result, executed_commands, llm_raw
        """
        collected = self._collect_recent_logs()
        if not collected:
            return {"error": "no_logs", "inspection_result": None}

        # Extract error snippets per file
        snippet_map = {}
        for fname, text in collected.items():
            snippets = self._extract_error_snippets(text)
            # if none extracted, include last N lines as context
            if not snippets and text:
                snippet_map[fname] = "\n".join(text.splitlines()[-200:])
            else:
                snippet_map[fname] = "\n\n---\n\n".join(snippets)

        prompt = self._build_prompt(snippet_map)
        # Ask the LLM (use system prompt to enforce short, JSON-only)
        system_prompt = (
            "You are an expert systems engineer. Be concise. Output only JSON as requested."
        )
        llm_output = self.reasoning.reason(prompt, system_prompt, max_tokens=1200)

        # Try to parse JSON safely
        parsed = None
        try:
            clean = llm_output.strip()
            # remove markdown code fences if any
            if clean.startswith("```"):
                clean = "\n".join(clean.splitlines()[1:-1])
            parsed = json.loads(clean)
        except Exception as e:
            # fallback: try to extract JSON substring
            import re
            m = re.search(r"\{.*\}", llm_output, flags=re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except:
                    parsed = {"error_parsing_llm": llm_output}
            else:
                parsed = {"error_parsing_llm": llm_output}

        result = {"inspection_result": parsed, "llm_raw": llm_output, "executed_commands": []}

        # If issue detected and auto_fix True, attempt to run whitelisted commands
        if isinstance(parsed, dict) and parsed.get("issue_detected") and parsed.get("shell_commands") and auto_fix:
            for cmd in parsed.get("shell_commands", []):
                cmd_trim = cmd.strip()
                if not self._is_command_whitelisted(cmd_trim):
                    # log skip
                    if self.logger:
                        self.logger.warning(f"Skipping non-whitelisted command: {cmd_trim}")
                    continue

                # If dry_run, only log
                if dry_run:
                    if self.logger:
                        self.logger.info(f"[DRY-RUN] Would execute: {cmd_trim}")
                    result["executed_commands"].append({"command": cmd_trim, "status": "dry_run"})
                else:
                    try:
                        # run with shell=False using shlex split for safety
                        proc = subprocess.run(shlex.split(cmd_trim), capture_output=True, text=True, timeout=120)
                        status = "ok" if proc.returncode == 0 else f"failed({proc.returncode})"
                        out = proc.stdout
                        err = proc.stderr
                        result["executed_commands"].append({
                            "command": cmd_trim,
                            "status": status,
                            "returncode": proc.returncode,
                            "stdout": out,
                            "stderr": err
                        })
                        if self.logger:
                            self.logger.info(f"Executed: {cmd_trim} -> {status}")
                    except Exception as ex:
                        result["executed_commands"].append({"command": cmd_trim, "status": f"error: {ex}"})
                        if self.logger:
                            self.logger.error(f"Error executing command {cmd_trim}: {ex}", exc_info=True)

        # persist an inspection file
        try:
            outdir = Config.LOGS_PATH / "inspections"
            outdir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(outdir / f"inspection_{timestamp}.json", "w") as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass

        return result

    def _is_command_whitelisted(self, cmd: str) -> bool:
        """Check if command is allowed. Very conservative."""
        # Exact match for simple commands
        for allowed_prefix, allowed_values in self.SAFE_COMMAND_WHITELIST.items():
            if cmd.startswith(allowed_prefix):
                # if whitelist has empty list -> allow direct prefix (e.g., apt-get update)
                if not allowed_values:
                    return True
                # Otherwise, parse the argument and check
                parts = cmd[len(allowed_prefix):].strip().split()
                if parts and parts[0] in allowed_values:
                    return True
        return False
