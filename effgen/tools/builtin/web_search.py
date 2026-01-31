"""
Web search tool with multiple backend support.

This module provides web search capabilities with support for multiple
search backends including DuckDuckGo, SerpAPI, and Google Custom Search.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import hashlib
import json
from enum import Enum

from ..base_tool import (
    BaseTool,
    ToolMetadata,
    ToolCategory,
    ParameterSpec,
    ParameterType,
)


logger = logging.getLogger(__name__)


class SearchBackend(Enum):
    """Supported search backends."""
    DUCKDUCKGO = "duckduckgo"
    SERPAPI = "serpapi"
    GOOGLE = "google"


class WebSearch(BaseTool):
    """
    Web search tool with multiple backend support.

    Features:
    - Multiple search backends (DuckDuckGo, SerpAPI, Google)
    - Result parsing and formatting
    - Metadata extraction (title, snippet, URL)
    - Rate limiting
    - Query caching
    - Result deduplication
    - Configurable result count

    Backends:
    - DuckDuckGo: Free, no API key required (default)
    - SerpAPI: Paid, requires API key, comprehensive results
    - Google: Requires API key and CSE ID, official Google results
    """

    def __init__(self):
        """Initialize the web search tool."""
        super().__init__(
            metadata=ToolMetadata(
                name="web_search",
                description="Search the web for information using various search engines",
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    ParameterSpec(
                        name="query",
                        type=ParameterType.STRING,
                        description="Search query",
                        required=True,
                        min_length=1,
                        max_length=500,
                    ),
                    ParameterSpec(
                        name="num_results",
                        type=ParameterType.INTEGER,
                        description="Number of results to return",
                        required=False,
                        default=5,
                        min_value=1,
                        max_value=20,
                    ),
                    ParameterSpec(
                        name="backend",
                        type=ParameterType.STRING,
                        description="Search backend to use",
                        required=False,
                        default="duckduckgo",
                        enum=["duckduckgo", "serpapi", "google"],
                    ),
                    ParameterSpec(
                        name="time_range",
                        type=ParameterType.STRING,
                        description="Time range for results (day, week, month, year, all)",
                        required=False,
                        enum=["day", "week", "month", "year", "all"],
                    ),
                    ParameterSpec(
                        name="language",
                        type=ParameterType.STRING,
                        description="Language code for results (e.g., 'en', 'es')",
                        required=False,
                        default="en",
                    ),
                    ParameterSpec(
                        name="region",
                        type=ParameterType.STRING,
                        description="Region code for results (e.g., 'us', 'uk')",
                        required=False,
                    ),
                ],
                returns={
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "snippet": {"type": "string"},
                            "position": {"type": "integer"},
                        },
                    },
                },
                timeout_seconds=30,
                requires_api_key=False,  # DuckDuckGo doesn't require one
                tags=["search", "web", "information", "internet"],
                examples=[
                    {
                        "query": "Python asyncio tutorial",
                        "num_results": 5,
                        "output": [
                            {
                                "title": "Asyncio Tutorial",
                                "url": "https://example.com/asyncio",
                                "snippet": "Learn Python asyncio...",
                                "position": 1,
                            }
                        ],
                    }
                ],
            )
        )
        self._cache: Dict[str, tuple[List[Dict[str, Any]], datetime]] = {}
        self._cache_ttl = timedelta(hours=1)
        self._api_keys: Dict[str, str] = {}

    async def initialize(self) -> None:
        """Initialize the web search tool."""
        await super().initialize()

        # Try to import search backends
        self._available_backends = set()

        # Check DuckDuckGo
        try:
            try:
                import ddgs
            except ImportError:
                import duckduckgo_search
            self._available_backends.add(SearchBackend.DUCKDUCKGO)
        except ImportError:
            logger.warning(
                "ddgs not installed. Install with: pip install ddgs"
            )

        # Check SerpAPI
        if self._api_keys.get("serpapi"):
            try:
                import serpapi
                self._available_backends.add(SearchBackend.SERPAPI)
            except ImportError:
                logger.warning(
                    "serpapi not installed. Install with: pip install google-search-results"
                )

        # Check Google
        if self._api_keys.get("google_api_key") and self._api_keys.get("google_cse_id"):
            try:
                import googleapiclient.discovery
                self._available_backends.add(SearchBackend.GOOGLE)
            except ImportError:
                logger.warning(
                    "google-api-python-client not installed. Install with: pip install google-api-python-client"
                )

        if not self._available_backends:
            logger.warning("No search backends available. Please install dependencies.")

    def configure_api_keys(self, **api_keys: str) -> None:
        """
        Configure API keys for various search backends.

        Args:
            serpapi: SerpAPI key
            google_api_key: Google API key
            google_cse_id: Google Custom Search Engine ID
        """
        self._api_keys.update(api_keys)

    async def _execute(
        self,
        query: str,
        num_results: int = 5,
        backend: str = "duckduckgo",
        time_range: Optional[str] = None,
        language: str = "en",
        region: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Execute a web search.

        Args:
            query: Search query
            num_results: Number of results to return
            backend: Search backend to use
            time_range: Time range filter
            language: Language code
            region: Region code

        Returns:
            List of search results with title, url, snippet, position
        """
        # Check cache first
        cache_key = self._get_cache_key(query, num_results, backend, time_range, language, region)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.debug(f"Returning cached results for query: {query}")
            return cached_result

        # Select backend
        backend_enum = SearchBackend(backend)

        # Execute search based on backend
        if backend_enum == SearchBackend.DUCKDUCKGO:
            results = await self._search_duckduckgo(
                query, num_results, time_range, language, region
            )
        elif backend_enum == SearchBackend.SERPAPI:
            results = await self._search_serpapi(
                query, num_results, time_range, language, region
            )
        elif backend_enum == SearchBackend.GOOGLE:
            results = await self._search_google(
                query, num_results, language, region
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        # Cache results
        self._add_to_cache(cache_key, results)

        return results

    async def _search_duckduckgo(
        self,
        query: str,
        num_results: int,
        time_range: Optional[str],
        language: str,
        region: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo."""
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                raise RuntimeError("ddgs or duckduckgo-search not installed")

        # Map time range
        timelimit = None
        if time_range:
            time_map = {"day": "d", "week": "w", "month": "m", "year": "y"}
            timelimit = time_map.get(time_range)

        # Execute search in thread pool (blocking API)
        def search():
            with DDGS() as ddgs:
                results = list(
                    ddgs.text(
                        query,
                        max_results=num_results,
                        timelimit=timelimit,
                        region=f"{region}-{language}" if region else None,
                    )
                )
            return results

        raw_results = await asyncio.to_thread(search)

        # Format results
        formatted_results = []
        for i, result in enumerate(raw_results, 1):
            formatted_results.append({
                "title": result.get("title", ""),
                "url": result.get("href", ""),
                "snippet": result.get("body", ""),
                "position": i,
            })

        return formatted_results

    async def _search_serpapi(
        self,
        query: str,
        num_results: int,
        time_range: Optional[str],
        language: str,
        region: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Search using SerpAPI."""
        try:
            from serpapi import GoogleSearch
        except ImportError:
            raise RuntimeError("serpapi not installed")

        if "serpapi" not in self._api_keys:
            raise RuntimeError("SerpAPI key not configured")

        # Build parameters
        params = {
            "q": query,
            "num": num_results,
            "api_key": self._api_keys["serpapi"],
            "hl": language,
        }

        if region:
            params["gl"] = region

        if time_range:
            time_map = {
                "day": "qdr:d",
                "week": "qdr:w",
                "month": "qdr:m",
                "year": "qdr:y",
            }
            if time_range in time_map:
                params["tbs"] = time_map[time_range]

        # Execute search in thread pool
        def search():
            search_client = GoogleSearch(params)
            return search_client.get_dict()

        raw_results = await asyncio.to_thread(search)

        # Format results
        formatted_results = []
        organic_results = raw_results.get("organic_results", [])

        for i, result in enumerate(organic_results[:num_results], 1):
            formatted_results.append({
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "position": i,
            })

        return formatted_results

    async def _search_google(
        self,
        query: str,
        num_results: int,
        language: str,
        region: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Search using Google Custom Search API."""
        try:
            from googleapiclient.discovery import build
        except ImportError:
            raise RuntimeError("google-api-python-client not installed")

        if "google_api_key" not in self._api_keys or "google_cse_id" not in self._api_keys:
            raise RuntimeError("Google API key or CSE ID not configured")

        # Build service
        def search():
            service = build(
                "customsearch",
                "v1",
                developerKey=self._api_keys["google_api_key"],
            )
            result = service.cse().list(
                q=query,
                cx=self._api_keys["google_cse_id"],
                num=num_results,
                hl=language,
                gl=region,
            ).execute()
            return result

        raw_results = await asyncio.to_thread(search)

        # Format results
        formatted_results = []
        items = raw_results.get("items", [])

        for i, item in enumerate(items, 1):
            formatted_results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "position": i,
            })

        return formatted_results

    def _get_cache_key(
        self,
        query: str,
        num_results: int,
        backend: str,
        time_range: Optional[str],
        language: str,
        region: Optional[str],
    ) -> str:
        """Generate cache key for search parameters."""
        key_data = f"{query}:{num_results}:{backend}:{time_range}:{language}:{region}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """Get results from cache if not expired."""
        if key in self._cache:
            results, timestamp = self._cache[key]
            if datetime.now() - timestamp < self._cache_ttl:
                return results
            else:
                del self._cache[key]
        return None

    def _add_to_cache(self, key: str, results: List[Dict[str, Any]]) -> None:
        """Add results to cache."""
        self._cache[key] = (results, datetime.now())

    def clear_cache(self) -> None:
        """Clear the search cache."""
        self._cache.clear()

    async def cleanup(self) -> None:
        """Clean up resources."""
        self.clear_cache()
        await super().cleanup()
