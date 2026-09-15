"""
Web search tool wrapper with Tavily API integration.

Abstracts the search interface behind a clean interface so SerpAPI or other
providers can be swapped in later. Includes offline mock mode for testing
without requiring a live API key.
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Optional
import logging

from src.models.schemas import Source

logger = logging.getLogger(__name__)


class WebSearchProvider(ABC):
    """Abstract base class for web search providers."""
    
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[Source]:
        """
        Execute a web search.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of Source objects
            
        Raises:
            Exception: If the search fails
        """
        pass


class TavilySearchProvider(WebSearchProvider):
    """Real Tavily API provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Tavily provider.
        
        Args:
            api_key: Tavily API key (defaults to TAVILY_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TAVILY_API_KEY not provided and not set in environment"
            )
        
        try:
            # Lazy import to avoid hard dependency if not using real provider
            from tavily import Client
            self.client = Client(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "tavily package not installed. "
                "Install with: pip install tavily-python"
            )
    
    def search(self, query: str, max_results: int = 5) -> list[Source]:
        """Search using Tavily API."""
        try:
            logger.info(f"Searching Tavily for: {query}")
            response = self.client.search(query, max_results=max_results)
            
            sources = []
            for result in response.get("results", []):
                source = Source(
                    url=result.get("url", ""),
                    title=result.get("title", ""),
                    snippet=result.get("content", "")
                )
                sources.append(source)
            
            logger.info(f"Retrieved {len(sources)} results from Tavily")
            return sources
        
        except Exception as e:
            logger.error(f"Tavily search failed: {str(e)}")
            raise


class MockSearchProvider(WebSearchProvider):
    """Mock provider for offline testing."""
    
    def __init__(self, fixture_file: Optional[str] = None):
        """
        Initialize mock provider.
        
        Args:
            fixture_file: Path to a JSON fixture file with mock results.
                         If None, returns generic mock data.
        """
        self.fixture_file = fixture_file
        self.fixtures = {}
        
        if fixture_file and os.path.exists(fixture_file):
            try:
                with open(fixture_file, "r") as f:
                    self.fixtures = json.load(f)
                logger.info(f"Loaded mock fixtures from {fixture_file}")
            except Exception as e:
                logger.warning(f"Failed to load fixture file: {e}")
    
    def search(self, query: str, max_results: int = 5) -> list[Source]:
        """
        Return mock search results.
        
        If fixtures exist for the query, use them; otherwise return
        generic mock data.
        """
        logger.info(f"Mock search for: {query}")
        
        # Check if we have fixtures for this query
        if query in self.fixtures:
            results = self.fixtures[query]
            logger.info(f"Using fixture data for query: {query}")
        else:
            # Generic mock results
            results = [
                {
                    "url": f"https://example.com/article-{i}",
                    "title": f"Mock Article {i} about {query}",
                    "snippet": f"This is a mock snippet about {query}. "
                               f"Contains relevant information for testing purposes."
                }
                for i in range(1, min(max_results + 1, 4))
            ]
        
        sources = [
            Source(
                url=r.get("url", ""),
                title=r.get("title", ""),
                snippet=r.get("snippet", "")
            )
            for r in results[:max_results]
        ]
        
        logger.info(f"Returning {len(sources)} mock results")
        return sources


def create_search_provider(
    provider_type: str = "mock",
    api_key: Optional[str] = None,
    fixture_file: Optional[str] = None
) -> WebSearchProvider:
    """
    Factory function to create a search provider.
    
    Args:
        provider_type: "tavily" or "mock"
        api_key: API key for real providers
        fixture_file: Path to fixture JSON for mock provider
        
    Returns:
        WebSearchProvider instance
        
    Raises:
        ValueError: If provider_type is not recognized
    """
    if provider_type == "tavily":
        return TavilySearchProvider(api_key=api_key)
    elif provider_type == "mock":
        return MockSearchProvider(fixture_file=fixture_file)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


class WebSearchTool:
    """
    Main interface for web search used by agents.
    
    Wraps the provider and handles error logging.
    """
    
    def __init__(self, provider: WebSearchProvider):
        """
        Initialize with a provider.
        
        Args:
            provider: WebSearchProvider instance
        """
        self.provider = provider
    
    def search(self, query: str, max_results: int = 5) -> list[Source]:
        """
        Execute a web search, handling errors gracefully.
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            
        Returns:
            List of Source objects (empty list if search fails)
        """
        try:
            return self.provider.search(query, max_results=max_results)
        except Exception as e:
            error_msg = f"Web search failed for query '{query}': {str(e)}"
            logger.error(error_msg)
            # Return empty list instead of raising so researcher can adapt
            return []
