"""
AgentClient — Python client library for the agent-mcp API plugin.

Usage (standalone):
    import asyncio
    from api_client import AgentClient

    async def main():
        client = AgentClient("http://localhost:8767", client_id="my-agent")
        response = await client.send("What is 2+2?")
        print(response)

    asyncio.run(main())

Usage (streaming):
    client = AgentClient("http://localhost:8767")
    async for token in client.stream("!llm_tools list"):
        print(token, end="", flush=True)

This library is also used internally by agents.py for agent_call() swarm calls.
"""

import json
import uuid
from typing import AsyncIterator

import httpx


class AgentClient:
    """
    Async HTTP client for agent-mcp API plugin (port 8767 by default).

    All methods are async and must be called from an async context.
    The client is stateless between calls — no persistent connection is kept.
    """

    def __init__(
        self,
        base_url: str,
        client_id: str = None,
        api_key: str = None,
        default_timeout: int = 60,
    ):
        """
        Args:
            base_url:           Base URL of the agent-mcp API plugin, e.g. "http://localhost:8767"
            client_id:          Session identifier. Auto-generated api-{8hex} if omitted.
                                Reuse the same client_id across calls to preserve session history.
            api_key:            Optional Bearer token if the server has API_KEY set.
            default_timeout:    Default timeout in seconds for send() calls.
        """
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id or f"api-{uuid.uuid4().hex[:8]}"
        self.default_timeout = default_timeout
        self._headers = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send(self, text: str, timeout: int = None) -> str:
        """
        Submit a message or command and wait for the complete response.

        Internally opens an SSE stream, accumulates all tokens, and returns the
        full response text when the server signals done.

        Args:
            text:    Any text, !command, or @model prompt.
            timeout: Max wait in seconds (default: self.default_timeout).

        Returns:
            Full response text as a single string.
        """
        timeout = timeout if timeout is not None else self.default_timeout
        accumulated = []

        async for token in self._stream_internal(text, timeout=timeout):
            accumulated.append(token)

        return "".join(accumulated)

    async def stream(self, text: str, timeout: int = None) -> AsyncIterator[str]:
        """
        Submit a message and yield response tokens as they arrive.

        Args:
            text:    Any text, !command, or @model prompt.
            timeout: Max seconds to wait for the stream to complete.

        Yields:
            str tokens as they stream from the server.
        """
        timeout = timeout if timeout is not None else self.default_timeout
        async for token in self._stream_internal(text, timeout=timeout):
            yield token

    async def sessions(self) -> list:
        """List all active sessions on the server."""
        async with httpx.AsyncClient(headers=self._headers, timeout=10) as http:
            resp = await http.get(f"{self.base_url}/api/v1/sessions")
            resp.raise_for_status()
            return resp.json().get("sessions", [])

    async def delete_session(self, session_id: str) -> dict:
        """Delete a session by full client_id or shorthand integer ID."""
        async with httpx.AsyncClient(headers=self._headers, timeout=10) as http:
            resp = await http.delete(f"{self.base_url}/api/v1/session/{session_id}")
            resp.raise_for_status()
            return resp.json()

    async def health(self) -> dict:
        """Check server health. Returns status dict."""
        async with httpx.AsyncClient(headers=self._headers, timeout=10) as http:
            resp = await http.get(f"{self.base_url}/api/v1/health")
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _stream_internal(self, text: str, timeout: int) -> AsyncIterator[str]:
        """
        Core streaming implementation.

        1. POST /api/v1/submit (async mode, wait=false)
        2. Open SSE stream on /api/v1/stream/{client_id}
        3. Yield tok events, stop on done/error.
        """
        # Submit the request
        submit_payload = {
            "client_id": self.client_id,
            "text": text,
            "wait": False,
        }
        async with httpx.AsyncClient(headers=self._headers, timeout=10) as http:
            resp = await http.post(
                f"{self.base_url}/api/v1/submit",
                json=submit_payload,
            )
            resp.raise_for_status()

        # Stream the response
        stream_url = f"{self.base_url}/api/v1/stream/{self.client_id}"

        async with httpx.AsyncClient(
            headers={**self._headers, "Accept": "text/event-stream"},
            timeout=httpx.Timeout(connect=10, read=timeout, write=10, pool=10),
        ) as http:
            async with http.stream("GET", stream_url) as resp:
                resp.raise_for_status()

                event_type = "message"
                data_lines = []

                async for line in resp.aiter_lines():
                    line = line.rstrip("\r")

                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].strip())
                    elif line == "":
                        # Blank line = end of event
                        raw_data = "\n".join(data_lines)
                        data_lines = []

                        if event_type in ("", "message") or event_type == "tok":
                            # Token event
                            try:
                                parsed = json.loads(raw_data)
                                token = parsed.get("text", "")
                            except (json.JSONDecodeError, ValueError):
                                token = raw_data
                            if token:
                                yield token

                        elif event_type == "done":
                            return

                        elif event_type == "error":
                            try:
                                parsed = json.loads(raw_data)
                                msg = parsed.get("message", raw_data)
                            except (json.JSONDecodeError, ValueError):
                                msg = raw_data
                            raise RuntimeError(f"Agent error: {msg}")

                        # Reset for next event
                        event_type = "message"
