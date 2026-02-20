#!/usr/bin/env python3
"""
MCP Agent with Plugin Architecture

Modular MCP agent server that loads plugins dynamically.

Core features (always enabled):
- System info (get_system_info tool)
- System prompt read/write (read_system_prompt, update_system_prompt tools)
- Session management (!session, !reset commands)
- LLM model management (!model command)
- Gate management (!autoAIdb, !autogate, !autoAISysPrompt)

Pluggable features:
- Client interfaces (shell.py, llama proxy)
- Data access tools (MySQL, Google Drive, Google Search)

Usage:
    python agent-mcp.py [--help]

Configuration:
    - plugins-enabled.json - Which plugins to load
    - plugin-manifest.json - Plugin metadata
    - .env - Credentials and environment variables
"""

import uvicorn
import argparse
import asyncio
import socket
import sys
from starlette.applications import Starlette
from starlette.routing import Route

from config import log
from plugin_loader import PluginLoader
from tools import get_core_tools
import tools as tools_module
import agents as agents_module
from tools import register_gate_tools, register_plugin_commands


def _check_port_available(host: str, port: int) -> bool:
    """Return True if the port is free to bind, False if already in use."""
    bind_host = "127.0.0.1" if host == "0.0.0.0" else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((bind_host, port))
            return True
        except OSError:
            return False


async def run_agent(host: str = "0.0.0.0"):
    """Run MCP agent with plugins."""
    import asyncio
    from uvicorn import Config, Server

    # Load plugins
    log.info("="*70)
    log.info("MCP Agent starting with plugin system")
    log.info("="*70)

    loader = PluginLoader()
    plugins = loader.load_all_enabled()

    if not plugins:
        log.warning("No plugins loaded - agent will have limited functionality")

    # Get default model from configuration
    default_model = loader.get_default_model()
    log.info(f"Default LLM model: {default_model}")

    # Collect all routes from plugins
    all_routes = []
    client_plugins = []

    for plugin_name, plugin in plugins.items():
        if plugin.PLUGIN_TYPE == "client_interface":
            routes = plugin.get_routes()
            all_routes.extend(routes)
            client_plugins.append(plugin)
            log.info(f"  + {plugin_name}: {len(routes)} routes")
        elif plugin.PLUGIN_TYPE == "data_tool":
            tool_defs = plugin.get_tools()
            # Register tools dynamically
            tools_module.register_plugin_tools(plugin_name, tool_defs)
            log.info(f"  + {plugin_name}: {len(tool_defs.get('lc', []))} tools")
            # Register gate tools (tools needing human approval)
            gate_tools = plugin.get_gate_tools()
            if gate_tools:
                register_gate_tools(plugin_name, gate_tools)
            # Register !command handlers
            commands = plugin.get_commands()
            if commands:
                register_plugin_commands(plugin_name, commands, plugin.get_help())

    # Update agents module with dynamic tools
    agents_module.update_tool_definitions()

    # Create Starlette app with all routes
    app = Starlette(routes=all_routes)

    # Determine which servers to run
    servers_to_run = []

    # Check all ports before starting any server — fail fast with a clear message
    port_conflicts = []
    seen_ports: dict = {}
    for plugin in client_plugins:
        plugin_config = plugin.get_config()
        port = plugin_config.get('port')
        host = plugin_config.get('host', '0.0.0.0')
        name = plugin_config.get('name', 'unknown')

        if port in seen_ports:
            port_conflicts.append(
                f"  Port {port}: claimed by both '{seen_ports[port]}' and '{name}'"
            )
        else:
            seen_ports[port] = name

        if not _check_port_available(host, port):
            port_conflicts.append(
                f"  Port {port} ({name}): already in use — "
                f"another process is listening on {host}:{port}"
            )

    if port_conflicts:
        log.error("=" * 70)
        log.error("STARTUP ABORTED — port conflict(s) detected:")
        for msg in port_conflicts:
            log.error(msg)
        log.error("")
        log.error("Fix options:")
        log.error("  1. Stop the process already using the port")
        log.error("  2. Change the port:  python plugin-manager.py port-set <plugin> <new_port>")
        log.error("  3. List configured ports:  python plugin-manager.py port-list")
        log.error("=" * 70)
        sys.exit(1)

    for plugin in client_plugins:
        plugin_config = plugin.get_config()
        port = plugin_config.get('port')
        host = plugin_config.get('host', '0.0.0.0')
        name = plugin_config.get('name', 'unknown')

        log.info(f"Starting {name} on {host}:{port}")
        log.info(f"  - Plugin: {plugin.PLUGIN_NAME}")

        config = Config(app, host=host, port=port, log_level="info")
        server = Server(config)
        servers_to_run.append(server.serve())

    if not servers_to_run:
        log.error("No client interface plugins enabled - cannot start any servers")
        return

    log.info("")
    log.info("="*70)
    log.info("Server startup complete!")
    log.info("="*70)

    for plugin in client_plugins:
        plugin_config = plugin.get_config()
        port = plugin_config.get('port')
        host = plugin_config.get('host', '0.0.0.0')
        name = plugin_config.get('name', 'unknown')
        log.info(f"  {name}: http://{host}:{port}")

    log.info("="*70)
    log.info("")

    # Run all servers concurrently
    await asyncio.gather(*servers_to_run)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='MCP Agent with Plugin Architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent-mcp.py                 # Start with plugins from plugins-enabled.json
  python plugin-manager.py list       # List available plugins
  python plugin-manager.py enable plugin_llama_proxy  # Enable llama proxy

Configuration Files:
  plugins-enabled.json    - Which plugins to enable
  plugin-manifest.json    - Plugin metadata
  .env                    - Environment variables and credentials
        """
    )

    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )

    args = parser.parse_args()

    # Run agent with asyncio
    try:
        asyncio.run(run_agent(host=args.host))
    except KeyboardInterrupt:
        log.info("\nShutting down...")
    except Exception as e:
        log.error(f"Fatal error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
