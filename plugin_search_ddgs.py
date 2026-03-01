"""
DuckDuckGo Search Plugin for MCP Agent

Provides ddgs_search tool for web search via DuckDuckGo (no API key required).
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from plugin_loader import BasePlugin


class _DdgsSearchArgs(BaseModel):
    query: str = Field(description="Search query")
    max_results: Optional[int] = Field(default=10, description="Maximum number of results to return (default: 10)")


async def search_ddgs_executor(query: str, max_results: int = 10) -> str:
    """Execute DuckDuckGo search."""
    return await _run_ddgs_search(query, max_results)


class SearchDdgsPlugin(BasePlugin):
    """DuckDuckGo web search plugin."""

    PLUGIN_NAME = "plugin_search_ddgs"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "data_tool"
    DESCRIPTION = "Web search via DuckDuckGo (no API key required)"
    DEPENDENCIES = ["ddgs"]
    ENV_VARS = []

    def __init__(self):
        self.enabled = False

    def init(self, config: dict) -> bool:
        """Initialize DuckDuckGo search plugin."""
        try:
            from ddgs import DDGS
            self.enabled = True
            return True
        except Exception as e:
            print(f"DuckDuckGo search plugin init failed: {e}")
            return False

    def shutdown(self) -> None:
        """Cleanup DuckDuckGo search resources."""
        self.enabled = False

    def get_tools(self) -> Dict[str, Any]:
        """Return DuckDuckGo search tool definitions in LangChain StructuredTool format."""
        return {
            "lc": [
                StructuredTool.from_function(
                    coroutine=search_ddgs_executor,
                    name="search_ddgs",
                    description=(
                        "Search the web using DuckDuckGo (no API key required). "
                        "Returns titles, URLs, and snippets for top results."
                    ),
                    args_schema=_DdgsSearchArgs,
                )
            ]
        }


async def _run_ddgs_search(query: str, max_results: int = 10) -> str:
    """Run a DuckDuckGo search and return formatted results."""
    import asyncio
    from ddgs import DDGS

    def _sync_search():
        with DDGS() as ddgs:
            results = ddgs.text(
                query,
                region='wt-wt',
                safesearch='moderate',
                timelimit=None,
                max_results=max_results
            )
            return list(results)

    try:
        results = await asyncio.get_event_loop().run_in_executor(None, _sync_search)

        if not results:
            return f"No results found for: {query}"

        lines = [f"DuckDuckGo search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', 'No title')}")
            lines.append(f"   URL: {r.get('href', '')}")
            snippet = r.get('body', '')
            if snippet:
                lines.append(f"   {snippet[:200]}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"DuckDuckGo search error: {e}"
