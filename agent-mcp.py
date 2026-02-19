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
from starlette.applications import Starlette
from starlette.routing import Route

from config import log
from plugin_loader import PluginLoader
from tools import get_core_tools
import tools as tools_module
import agents as agents_module
from tools import register_gate_tools


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

    # Update agents module with dynamic tools
    agents_module.update_tool_definitions()

    # Create Starlette app with all routes
    app = Starlette(routes=all_routes)

    # Determine which servers to run
    servers_to_run = []

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
