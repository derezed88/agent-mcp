"""
Google Search Plugin for MCP Agent

Provides google_search tool for web search via Gemini grounding.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from plugin_loader import BasePlugin
from search import run_google_search


class _GoogleSearchArgs(BaseModel):
    query: str = Field(description="Search query")


async def search_google_executor(query: str) -> str:
    """Execute Google search."""
    return await run_google_search(query)


class GoogleSearchPlugin(BasePlugin):
    """Google Search via Gemini grounding plugin."""

    PLUGIN_NAME = "plugin_search_google"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "data_tool"
    DESCRIPTION = "Web search via Gemini grounding"
    DEPENDENCIES = ["httpx"]
    ENV_VARS = ["GEMINI_API_KEY"]

    def __init__(self):
        self.enabled = False

    def init(self, config: dict) -> bool:
        """Initialize Google Search plugin."""
        try:
            import os
            gemini_key = os.getenv("GEMINI_API_KEY")
            if not gemini_key:
                print("Google Search plugin: GEMINI_API_KEY not set in .env")
                return False

            import httpx
            self.enabled = True
            return True
        except Exception as e:
            print(f"Google Search plugin init failed: {e}")
            return False

    def shutdown(self) -> None:
        """Cleanup Google Search resources."""
        self.enabled = False

    def get_gate_tools(self) -> Dict[str, Any]:
        """Declare search_google as a read-only gated search tool."""
        return {
            "search_google": {
                "type": "search",
                "operations": ["read"],
                "description": "web search via Gemini grounding (read-only)"
            }
        }

    def get_tools(self) -> Dict[str, Any]:
        """Return Google Search tool definitions in LangChain StructuredTool format."""
        return {
            "lc": [
                StructuredTool.from_function(
                    coroutine=search_google_executor,
                    name="search_google",
                    description="Search the web using Google Search (Gemini grounding). Third-level fallback.",
                    args_schema=_GoogleSearchArgs,
                )
            ]
        }
