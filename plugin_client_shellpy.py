"""
Shell.py Client Interface Plugin for MCP Agent

Provides MCP endpoints for shell.py client:
- /submit - Submit user messages
- /stream - SSE streaming for responses
- /gate_response - Gate approval responses
- /health - Health check
- /sessions - List sessions
- /session/{sid} - Delete session
"""

from typing import List
from starlette.routing import Route
from plugin_loader import BasePlugin
from routes import (
    endpoint_submit,
    endpoint_stream,
    endpoint_gate_response,
    endpoint_health,
    endpoint_list_sessions,
    endpoint_delete_session
)


class ShellpyClientPlugin(BasePlugin):
    """Shell.py client interface plugin."""

    PLUGIN_NAME = "plugin_client_shellpy"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "client_interface"
    DESCRIPTION = "shell.py client with SSE streaming and gate approval UI"
    DEPENDENCIES = ["sse-starlette"]
    ENV_VARS = []

    def __init__(self):
        self.enabled = False
        self.mcp_port = 8765
        self.mcp_host = "0.0.0.0"

    def init(self, config: dict) -> bool:
        """Initialize shell.py client plugin."""
        try:
            # Get configuration
            self.mcp_port = config.get('mcp_port', 8765)
            self.mcp_host = config.get('mcp_host', '0.0.0.0')

            # Verify sse-starlette is available
            import sse_starlette
            self.enabled = True
            return True
        except Exception as e:
            print(f"Shell.py client plugin init failed: {e}")
            return False

    def shutdown(self) -> None:
        """Cleanup shell.py client resources."""
        self.enabled = False

    def get_routes(self) -> List[Route]:
        """Return Starlette routes for shell.py client."""
        return [
            Route("/submit", endpoint_submit, methods=["POST"]),
            Route("/stream", endpoint_stream, methods=["GET"]),
            Route("/gate_response", endpoint_gate_response, methods=["POST"]),
            Route("/health", endpoint_health, methods=["GET"]),
            Route("/sessions", endpoint_list_sessions, methods=["GET"]),
            Route("/session/{sid}", endpoint_delete_session, methods=["DELETE"]),
        ]

    def get_config(self) -> dict:
        """Return plugin configuration for server startup."""
        return {
            "port": self.mcp_port,
            "host": self.mcp_host,
            "name": "MCP service"
        }
