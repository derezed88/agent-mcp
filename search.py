import os
import httpx
#from .config import log
from config import log

async def run_google_search(query: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY is not set in .env"

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    payload = {
        "contents": [{"parts": [{"text": query}]}],
        "tools":    [{"google_search": {}}],
    }
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(url, json=payload, params={"key": api_key})
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        return f"Google Search failed: {exc}"

    try:
        parts = data["candidates"][0]["content"]["parts"]
        answer = "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError):
        answer = "(No answer text returned)"

    sources = []
    try:
        gm = data["candidates"][0].get("groundingMetadata", {})
        for chunk in gm.get("groundingChunks", []):
            web = chunk.get("web", {})
            uri, title = web.get("uri", ""), web.get("title", "")
            if uri: sources.append(f"  - {title + ': ' if title else ''}{uri}")
    except Exception:
        pass

    result = f"[Search: {query!r}]\n\n{answer}"
    if sources:
        result += "\n\nSources:\n" + "\n".join(sources)
    return result