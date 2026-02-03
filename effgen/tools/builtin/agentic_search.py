"""
Agentic Search tool using bash commands for exact query matching.

This module provides an advanced retrieval tool that uses bash commands
(grep, find, etc.) to search for exact matches in a knowledge base,
then provides context around the matches.

This is an alternative to embedding-based RAG that works particularly
well for:
- Exact phrase matching
- Technical queries (code, formulas, specific terms)
- When semantic similarity might miss exact answers
- Large knowledge bases where indexing is impractical
"""

import os
import re
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import shutil

from ..base_tool import (
    BaseTool,
    ToolMetadata,
    ToolCategory,
    ParameterSpec,
    ParameterType,
)


logger = logging.getLogger(__name__)


@dataclass
class SearchMatch:
    """A search match with context."""
    file_path: str
    line_number: int
    match_line: str
    context_before: List[str]
    context_after: List[str]
    relevance_score: float


class AgenticSearch(BaseTool):
    """
    Agentic Search tool using bash commands for precise retrieval.

    Unlike embedding-based RAG, this tool uses exact string matching
    with grep, find, and other bash commands to locate information.
    It then provides context around matches.

    Features:
    - Exact string matching (grep-based)
    - Fuzzy matching with agrep (if available)
    - Context extraction (configurable lines before/after)
    - Multiple file format support (txt, json, jsonl, md, csv)
    - Case-sensitive/insensitive search
    - Regular expression support
    - Keyword extraction and multi-term search
    - Relevance scoring based on match quality

    This approach is particularly effective for:
    - Technical/scientific queries
    - Code search
    - Finding specific facts, numbers, formulas
    - Cases where semantic similarity might miss exact answers
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        context_lines: int = 5,
        max_results: int = 10,
        supported_extensions: Optional[List[str]] = None,
    ):
        """
        Initialize the agentic search tool.

        Args:
            data_path: Path to the knowledge base directory/file
            context_lines: Number of lines of context to include before/after match
            max_results: Maximum number of results to return
            supported_extensions: File extensions to search (default: txt, json, jsonl, md, csv)
        """
        super().__init__(
            metadata=ToolMetadata(
                name="agentic_search",
                description=(
                    "Search a knowledge base using exact string matching with grep/find commands. "
                    "Returns matching passages with surrounding context. "
                    "Use this tool when you need to find EXACT information, specific terms, "
                    "numbers, formulas, or technical content. Works better than semantic search "
                    "for precise queries."
                ),
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    ParameterSpec(
                        name="query",
                        type=ParameterType.STRING,
                        description=(
                            "The search query. Can be exact text, keywords, or a regex pattern. "
                            "For best results, use specific terms from the expected answer."
                        ),
                        required=True,
                        min_length=1,
                        max_length=500,
                    ),
                    ParameterSpec(
                        name="context_lines",
                        type=ParameterType.INTEGER,
                        description="Number of lines of context to show before and after each match",
                        required=False,
                        default=5,
                        min_value=0,
                        max_value=50,
                    ),
                    ParameterSpec(
                        name="case_sensitive",
                        type=ParameterType.BOOLEAN,
                        description="Whether to perform case-sensitive search",
                        required=False,
                        default=False,
                    ),
                    ParameterSpec(
                        name="use_regex",
                        type=ParameterType.BOOLEAN,
                        description="Whether to treat query as a regular expression",
                        required=False,
                        default=False,
                    ),
                    ParameterSpec(
                        name="max_results",
                        type=ParameterType.INTEGER,
                        description="Maximum number of results to return",
                        required=False,
                        default=10,
                        min_value=1,
                        max_value=50,
                    ),
                    ParameterSpec(
                        name="search_mode",
                        type=ParameterType.STRING,
                        description=(
                            "Search mode: 'exact' for exact phrase, "
                            "'keywords' to search for all keywords, "
                            "'any' to match any keyword"
                        ),
                        required=False,
                        default="keywords",
                        enum=["exact", "keywords", "any"],
                    ),
                ],
                returns={
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string"},
                                    "file": {"type": "string"},
                                    "line_number": {"type": "integer"},
                                    "score": {"type": "number"},
                                },
                            },
                        },
                        "total_matches": {"type": "integer"},
                        "query_terms": {"type": "array"},
                    },
                },
                timeout_seconds=60,
                tags=["search", "retrieval", "grep", "exact-match", "knowledge-base"],
                examples=[
                    {
                        "query": "photosynthesis converts",
                        "context_lines": 5,
                        "output": {
                            "results": [
                                {
                                    "content": "...context before...\nPhotosynthesis converts light energy into chemical energy.\n...context after...",
                                    "file": "biology.txt",
                                    "line_number": 42,
                                    "score": 0.95,
                                }
                            ],
                            "total_matches": 1,
                        },
                    }
                ],
            )
        )

        self.data_path = data_path
        self.default_context_lines = context_lines
        self.default_max_results = max_results
        self.supported_extensions = supported_extensions or [
            ".txt", ".json", ".jsonl", ".md", ".csv", ".tsv",
        ]

        # Cache for file contents (optional optimization)
        self._file_cache: Dict[str, List[str]] = {}
        self._cache_enabled = False

    async def initialize(self) -> None:
        """Initialize the agentic search tool."""
        await super().initialize()

        # Verify data path exists
        if self.data_path:
            if not os.path.exists(self.data_path):
                logger.warning(f"Data path does not exist: {self.data_path}")
            else:
                logger.info(f"Agentic search initialized with data path: {self.data_path}")

        # Check for required commands
        self._check_commands()

    def _check_commands(self):
        """Check if required bash commands are available."""
        required = ["grep", "find"]
        optional = ["agrep", "rg"]  # ripgrep as alternative

        for cmd in required:
            if not shutil.which(cmd):
                logger.warning(f"Required command not found: {cmd}")

        for cmd in optional:
            if shutil.which(cmd):
                logger.debug(f"Optional command available: {cmd}")

    def set_data_path(self, path: str):
        """
        Set the data path for searching.

        Args:
            path: Path to the knowledge base directory or file
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data path not found: {path}")
        self.data_path = path
        self._file_cache.clear()
        logger.info(f"Data path set to: {path}")

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract meaningful keywords from query.

        Args:
            query: The search query

        Returns:
            List of keywords
        """
        # Remove common stop words
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "under",
            "again", "further", "then", "once", "here", "there", "when",
            "where", "why", "how", "all", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "but", "and",
            "or", "if", "because", "until", "while", "what", "which", "who",
            "whom", "this", "that", "these", "those", "am", "it", "its"
        }

        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        # Keep original case for proper nouns / technical terms
        original_words = re.findall(r'\b\w+\b', query)
        keyword_set = set(keywords)

        # Prefer original case if keyword matches
        result = []
        for orig in original_words:
            if orig.lower() in keyword_set and orig.lower() not in [r.lower() for r in result]:
                result.append(orig)

        return result if result else words[:5]  # Fallback to first 5 words

    def _run_grep(
        self,
        pattern: str,
        path: str,
        case_sensitive: bool = False,
        context_lines: int = 5,
        use_regex: bool = False,
        max_count: int = 100,
    ) -> List[Tuple[str, int, str, List[str], List[str]]]:
        """
        Run grep command and parse results.

        Args:
            pattern: Search pattern
            path: Path to search
            case_sensitive: Case sensitivity
            context_lines: Lines of context
            use_regex: Use regex mode
            max_count: Maximum matches per file

        Returns:
            List of (file, line_num, match_line, context_before, context_after)
        """
        # Build grep command
        cmd = ["grep"]

        if not case_sensitive:
            cmd.append("-i")

        if not use_regex:
            cmd.append("-F")  # Fixed string mode

        # Context
        cmd.extend(["-B", str(context_lines), "-A", str(context_lines)])

        # Line numbers
        cmd.append("-n")

        # Recursive if directory
        if os.path.isdir(path):
            cmd.append("-r")
            # Include only supported file types
            for ext in self.supported_extensions:
                cmd.extend(["--include", f"*{ext}"])

        # Max count per file
        cmd.extend(["-m", str(max_count)])

        # Pattern and path
        cmd.append(pattern)
        cmd.append(path)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode > 1:  # 0 = found, 1 = not found, >1 = error
                logger.warning(f"grep error: {result.stderr}")
                return []

            return self._parse_grep_output(result.stdout, context_lines, path)

        except subprocess.TimeoutExpired:
            logger.warning("grep command timed out")
            return []
        except Exception as e:
            logger.error(f"Error running grep: {e}")
            return []

    def _parse_grep_output(
        self,
        output: str,
        context_lines: int,
        source_file: str = "unknown",
    ) -> List[Tuple[str, int, str, List[str], List[str]]]:
        """
        Parse grep output with context.

        Args:
            output: grep output string
            context_lines: Expected context lines
            source_file: Source file path (used when grep doesn't include filename)

        Returns:
            List of (file, line_num, match_line, context_before, context_after)
        """
        results = []
        current_match_line = None
        current_line_num = None
        context_before = []
        context_after = []
        in_after_context = False
        after_count = 0

        for line in output.split("\n"):
            if not line:
                continue

            # Group separator
            if line == "--":
                if current_match_line is not None:
                    results.append((
                        source_file,
                        current_line_num or 0,
                        current_match_line,
                        context_before.copy(),
                        context_after.copy(),
                    ))
                current_match_line = None
                current_line_num = None
                context_before = []
                context_after = []
                in_after_context = False
                after_count = 0
                continue

            # Try different grep output formats
            # Format 1: FILE:LINE_NUM:CONTENT (recursive search)
            # Format 2: LINE_NUM:CONTENT (single file, match line)
            # Format 3: LINE_NUM-CONTENT (single file, context line)

            # Try single-file format first (simpler)
            single_match = re.match(r'^(\d+)([:\-])(.*)$', line)
            if single_match:
                line_num = int(single_match.group(1))
                separator = single_match.group(2)
                content = single_match.group(3)
                file_path = source_file
            else:
                # Try format with filename (recursive search)
                multi_match = re.match(r'^(.+?):(\d+)([:\-])(.*)$', line)
                if multi_match:
                    file_path = multi_match.group(1)
                    line_num = int(multi_match.group(2))
                    separator = multi_match.group(3)
                    content = multi_match.group(4)
                else:
                    continue
            # Determine if this is the match line or context based on separator
            is_match = (separator == ":")

            if is_match and not in_after_context:
                # This is a match line
                if current_match_line is not None:
                    # Save previous match
                    results.append((
                        file_path,
                        current_line_num or 0,
                        current_match_line,
                        context_before.copy(),
                        context_after.copy(),
                    ))
                    context_before = []
                    context_after = []

                current_line_num = line_num
                current_match_line = content
                in_after_context = True
                after_count = 0

            elif in_after_context:
                # After context
                context_after.append(content)
                after_count += 1
                if after_count >= context_lines:
                    in_after_context = False

            else:
                # Before context
                context_before.append(content)
                if len(context_before) > context_lines:
                    context_before.pop(0)

        # Don't forget the last match
        if current_match_line is not None:
            results.append((
                source_file,
                current_line_num or 0,
                current_match_line,
                context_before,
                context_after,
            ))

        return results

    def _calculate_relevance(
        self,
        match_line: str,
        context: str,
        query_terms: List[str],
    ) -> float:
        """
        Calculate relevance score for a match.

        Args:
            match_line: The matched line
            context: Full context string
            query_terms: Query keywords

        Returns:
            Relevance score (0-1)
        """
        if not query_terms:
            return 0.5

        full_text = (match_line + " " + context).lower()
        match_line_lower = match_line.lower()

        # Count term matches
        terms_in_match = sum(1 for t in query_terms if t.lower() in match_line_lower)
        terms_in_context = sum(1 for t in query_terms if t.lower() in full_text)

        # Weights
        match_weight = 0.7
        context_weight = 0.3

        # Calculate scores
        match_score = terms_in_match / len(query_terms) if query_terms else 0
        context_score = terms_in_context / len(query_terms) if query_terms else 0

        # Combined score
        score = match_weight * match_score + context_weight * context_score

        # Boost for exact phrase match
        query_phrase = " ".join(query_terms).lower()
        if query_phrase in match_line_lower:
            score = min(1.0, score + 0.2)

        return round(score, 4)

    async def _execute(
        self,
        query: str,
        context_lines: Optional[int] = None,
        case_sensitive: bool = False,
        use_regex: bool = False,
        max_results: Optional[int] = None,
        search_mode: str = "keywords",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute agentic search.

        Args:
            query: Search query
            context_lines: Lines of context around matches
            case_sensitive: Case sensitivity
            use_regex: Use regex mode
            max_results: Maximum results
            search_mode: Search mode (exact, keywords, any)

        Returns:
            Dict with results and metadata
        """
        if not self.data_path:
            return {
                "results": [],
                "total_matches": 0,
                "error": "No data path set. Use set_data_path() to specify the knowledge base location.",
            }

        if not os.path.exists(self.data_path):
            return {
                "results": [],
                "total_matches": 0,
                "error": f"Data path not found: {self.data_path}",
            }

        context_lines = context_lines or self.default_context_lines
        max_results = max_results or self.default_max_results

        # Extract keywords
        query_terms = self._extract_keywords(query)

        all_matches = []

        if search_mode == "exact":
            # Search for exact phrase
            matches = self._run_grep(
                query,
                self.data_path,
                case_sensitive=case_sensitive,
                context_lines=context_lines,
                use_regex=use_regex,
            )
            all_matches.extend(matches)

        elif search_mode == "keywords":
            # Search for all keywords (AND logic - filter results that have all terms)
            if query_terms:
                # First, find matches for any term
                first_term = query_terms[0]
                matches = self._run_grep(
                    first_term,
                    self.data_path,
                    case_sensitive=case_sensitive,
                    context_lines=context_lines,
                    use_regex=False,
                )

                # Filter to keep only matches containing all terms
                for match in matches:
                    file_path, line_num, match_line, before, after = match
                    full_context = " ".join(before) + " " + match_line + " " + " ".join(after)
                    full_context_lower = full_context.lower()

                    if all(term.lower() in full_context_lower for term in query_terms):
                        all_matches.append(match)

        elif search_mode == "any":
            # Search for any keyword (OR logic)
            seen = set()  # Avoid duplicates
            for term in query_terms:
                matches = self._run_grep(
                    term,
                    self.data_path,
                    case_sensitive=case_sensitive,
                    context_lines=context_lines,
                    use_regex=False,
                )
                for match in matches:
                    key = (match[0], match[1])  # (file, line_num)
                    if key not in seen:
                        seen.add(key)
                        all_matches.append(match)

        # Build results with relevance scoring
        results = []
        for file_path, line_num, match_line, before, after in all_matches:
            context_str = "\n".join(before) + "\n>>> " + match_line + " <<<\n" + "\n".join(after)

            score = self._calculate_relevance(match_line, context_str, query_terms)

            results.append({
                "content": context_str.strip(),
                "match_line": match_line,
                "file": os.path.basename(file_path),
                "file_path": file_path,
                "line_number": line_num,
                "score": score,
                "context_before": before,
                "context_after": after,
            })

        # Sort by relevance and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:max_results]

        # Add rank
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return {
            "results": results,
            "total_matches": len(all_matches),
            "query_terms": query_terms,
            "search_mode": search_mode,
            "data_path": self.data_path,
        }

    def search_in_json(
        self,
        query: str,
        json_path: str,
        content_field: str = "text",
        context_lines: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search within a JSON/JSONL file's specific field.

        Args:
            query: Search query
            json_path: Path to JSON/JSONL file
            content_field: Field containing searchable content
            context_lines: Context lines (for chunked content)

        Returns:
            List of matching entries
        """
        results = []

        path = Path(json_path)
        if not path.exists():
            return results

        query_lower = query.lower()
        keywords = self._extract_keywords(query)

        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    try:
                        entry = json.loads(line.strip())
                        content = entry.get(content_field, "")

                        # Check if query or keywords match
                        content_lower = str(content).lower()
                        if query_lower in content_lower or all(k.lower() in content_lower for k in keywords):
                            score = self._calculate_relevance(str(content), "", keywords)
                            results.append({
                                "content": content,
                                "entry": entry,
                                "line_index": i,
                                "score": score,
                            })
                    except json.JSONDecodeError:
                        continue

        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for i, entry in enumerate(data):
                    if isinstance(entry, dict):
                        content = entry.get(content_field, "")
                    else:
                        content = str(entry)

                    content_lower = content.lower()
                    if query_lower in content_lower or all(k.lower() in content_lower for k in keywords):
                        score = self._calculate_relevance(content, "", keywords)
                        results.append({
                            "content": content,
                            "entry": entry if isinstance(entry, dict) else {"text": entry},
                            "index": i,
                            "score": score,
                        })

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)

        return results

    async def cleanup(self) -> None:
        """Clean up resources."""
        self._file_cache.clear()
        await super().cleanup()
