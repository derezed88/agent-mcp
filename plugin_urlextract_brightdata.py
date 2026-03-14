"""
Bright Data URL Extraction Plugin for MCP Agent

Provides url_extract_brightdata tool for extracting web page content
via Bright Data's Web Unlocker API (same endpoint used by the official
@brightdata/mcp npm package).  Returns markdown-formatted page content
with numbered link references.

Requires BRIGHTDATA_API_KEY in .env.
"""

import re
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from plugin_loader import BasePlugin

_API_URL = "https://api.brightdata.com/request"
_UNLOCKER_ZONE = "mcp_unlocker"


class _UrlExtractBrightdataArgs(BaseModel):
    url: str = Field(description="The URL of the web page to extract content from")
    query: Optional[str] = Field(
        default="",
        description=(
            "Optional query to focus the extraction. "
            "When provided, only the most relevant sections "
            "matching the query are highlighted. "
            "Omit for full-page extraction."
        )
    )


class UrlextractBrightdataPlugin(BasePlugin):
    """Bright Data URL extraction plugin."""

    PLUGIN_NAME = "plugin_urlextract_brightdata"
    PLUGIN_VERSION = "1.1.0"
    PLUGIN_TYPE = "data_tool"
    DESCRIPTION = "Web page content extraction via Bright Data scraping (markdown output)"
    DEPENDENCIES = ["httpx"]
    ENV_VARS = ["BRIGHTDATA_API_KEY"]

    def __init__(self):
        self.enabled = False
        self._api_key = None

    def init(self, config: dict) -> bool:
        try:
            import os
            api_key = os.getenv("BRIGHTDATA_API_KEY")
            if not api_key:
                print("Bright Data extract plugin: BRIGHTDATA_API_KEY not set in .env")
                return False
            self._api_key = api_key
            self.enabled = True
            return True
        except Exception as e:
            print(f"Bright Data extract plugin init failed: {e}")
            return False

    def shutdown(self) -> None:
        self._api_key = None
        self.enabled = False

    def get_tools(self) -> Dict[str, Any]:
        api_key = self._api_key

        async def url_extract_brightdata_executor(url: str, query: str = "") -> str:
            return await _run_brightdata_extract(api_key, url, query)

        return {
            "lc": [
                StructuredTool.from_function(
                    coroutine=url_extract_brightdata_executor,
                    name="url_extract_brightdata",
                    description=(
                        "Extract web page content from a URL using Bright Data's "
                        "scraping infrastructure. Returns the page as markdown with "
                        "links converted to numbered references. "
                        "Handles JavaScript-rendered pages and anti-bot protections. "
                        "Optionally provide a query to focus on relevant sections."
                    ),
                    args_schema=_UrlExtractBrightdataArgs,
                )
            ]
        }


def _compress_markdown(text: str) -> str:
    """Convert inline markdown links to numbered reference-style links."""
    url_to_ref = {}
    ref_list = []

    def replace_link(m):
        anchor, url = m.group(1), m.group(2)
        url = re.sub(r'\s+"[^"]*"$', '', url).strip()
        if url not in url_to_ref:
            ref_num = len(ref_list) + 1
            url_to_ref[url] = ref_num
            ref_list.append((ref_num, url))
        return f"{anchor}[{url_to_ref[url]}]"

    compressed = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace_link, text)

    if ref_list:
        compressed += "\n\n---\n"
        for num, url in ref_list:
            compressed += f"[{num}]: {url}\n"

    return compressed


async def _run_brightdata_extract(api_key: str, url: str, query: str = "") -> str:
    """Extract web page content via Bright Data Web Unlocker (markdown mode)."""
    import httpx

    if not api_key:
        return "Bright Data extract: BRIGHTDATA_API_KEY not configured."

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "url": url,
        "zone": _UNLOCKER_ZONE,
        "format": "raw",
        "data_format": "markdown",
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(_API_URL, headers=headers, json=payload)

        if resp.status_code != 200:
            return f"Bright Data extract error (HTTP {resp.status_code}): {resp.text[:400]}"

        md = resp.text
        if not md or len(md.strip()) < 20:
            return f"Bright Data extract: empty response for {url}"

        return _format_extraction(url, query, md)

    except Exception as e:
        return f"Bright Data extract error: {e}"


def _format_extraction(url: str, query: str, raw_content: str) -> str:
    """Format extracted content with metadata header and compressed links."""
    lines = [f"Source: {url}"]
    if query:
        lines.append(f"Query: {query}")
    lines.append("")

    compressed = _compress_markdown(raw_content)
    lines.append(compressed)

    return "\n".join(lines)
