import datetime
import platform
from mcp.server.fastmcp import FastMCP
from langchain_core.tools import StructuredTool

from config import log
from database import execute_sql
from drive import run_drive_op
from search import run_google_search
from prompt import get_section

mcp_server = FastMCP("AIOps-DB-Tools")


@mcp_server.tool()
async def db_query(sql: str) -> str:
    """Execute SQL against mymcp MySQL database."""
    return await execute_sql(sql)


@mcp_server.tool()
async def get_system_info() -> dict:
    """Return current date/time and status."""
    return {
        "local_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "PST",
        "status": "connected",
        "platform": platform.system(),
    }


@mcp_server.tool()
async def google_search(query: str) -> str:
    """Search web using Gemini grounding."""
    return await run_google_search(query)


@mcp_server.tool()
async def google_drive(
    operation: str,
    file_id: str = "",
    file_name: str = "",
    content: str = "",
    folder_id: str = "",
) -> str:
    """
    Perform CRUD operations on Google Drive within a SPECIFIC authorized folder.

    IMPORTANT: This tool only accesses files in a pre-configured folder (FOLDER_ID from .env).
    Do NOT pass folder_id="root" or attempt to access the entire Drive.
    Leave folder_id empty to use the configured folder.
    """
    return await run_drive_op(
        operation,
        file_id or None,
        file_name or None,
        content or None,
        folder_id or None,
    )


# ---------------------------------------------------------------------------
# Dynamic Tool Registry
# ---------------------------------------------------------------------------

# Plugin tool storage — all plugins now use LangChain StructuredTool format (Step 2b)
_PLUGIN_TOOLS_LC: list = []
_PLUGIN_TOOL_EXECUTORS: dict = {}

# Gate tool registry: tool_name -> {"type": str, "operations": list[str], "description": str}
# type examples: "search", "drive", "db"
# operations: ["read"], ["write"], ["read", "write"]
_PLUGIN_GATE_TOOLS: dict = {}

# Plugin command registry: command_name -> async handler(subcommand_or_args: str) -> str
# Handlers have signature: async (args: str) -> str
# Populated at startup by register_plugin_commands(); queried by routes.py dispatch.
# Also stores optional help text per plugin: _PLUGIN_HELP[plugin_name] -> str
_PLUGIN_COMMANDS: dict = {}   # cmd_name -> async handler
_PLUGIN_HELP: dict = {}       # plugin_name -> help string (from get_help())


def register_gate_tools(plugin_name: str, gate_tools: dict):
    """
    Register gated tools from a plugin.

    Args:
        plugin_name: Name of the plugin
        gate_tools: Dict mapping tool_name -> {"type": str, "operations": list[str], "description": str}
    """
    global _PLUGIN_GATE_TOOLS
    _PLUGIN_GATE_TOOLS.update(gate_tools)
    log.info(f"Registered {len(gate_tools)} gate tool(s) from {plugin_name}: {list(gate_tools.keys())}")


def get_gate_tools_by_type(tool_type: str) -> list[str]:
    """Return all registered tool names of the given type (e.g. 'search')."""
    return [name for name, meta in _PLUGIN_GATE_TOOLS.items() if meta.get("type") == tool_type]


def get_all_gate_tools() -> dict:
    """Return the full gate tool registry."""
    return _PLUGIN_GATE_TOOLS


def register_plugin_commands(plugin_name: str, commands: dict, help_text: str = ""):
    """
    Register !command handlers contributed by a plugin.

    Args:
        plugin_name: Plugin identifier (for logging)
        commands: Dict mapping command_name -> async handler(args: str) -> str
        help_text: Optional help section string from plugin.get_help()
    """
    global _PLUGIN_COMMANDS, _PLUGIN_HELP
    _PLUGIN_COMMANDS.update(commands)
    if help_text:
        _PLUGIN_HELP[plugin_name] = help_text
    log.info(f"Registered {len(commands)} command(s) from {plugin_name}: {list(commands.keys())}")


def get_plugin_command(cmd_name: str):
    """Return the handler for a plugin-registered command, or None if not found."""
    return _PLUGIN_COMMANDS.get(cmd_name)


def get_plugin_help_sections() -> list[str]:
    """Return all plugin help section strings, in registration order."""
    return list(_PLUGIN_HELP.values())


def register_plugin_tools(plugin_name: str, tool_defs: dict):
    """
    Register tools from a plugin.

    Expected format: {'lc': [StructuredTool, ...]}

    Executors are extracted automatically from the coroutine attribute of each
    StructuredTool, so no 'executors' key is needed.
    """
    global _PLUGIN_TOOLS_LC, _PLUGIN_TOOL_EXECUTORS

    lc_tools = tool_defs.get('lc', [])
    _PLUGIN_TOOLS_LC.extend(lc_tools)
    for lc_tool in lc_tools:
        if lc_tool.coroutine and lc_tool.name not in _PLUGIN_TOOL_EXECUTORS:
            _PLUGIN_TOOL_EXECUTORS[lc_tool.name] = lc_tool.coroutine
    log.info(f"Registered {len(lc_tools)} LC tools from {plugin_name}")


def get_section_for_tool(tool_name: str) -> str:
    """
    Return the system prompt section body for a named tool.
    Tries 'tool-<name>' (hyphenated) and 'tool_<name>' (underscored) variants.
    Used by llm_clean_tool to build the target model's system prompt.
    """
    # Try hyphenated form first (canonical: tool-url-extract)
    hyphenated = "tool-" + tool_name.replace("_", "-")
    section = get_section(hyphenated)
    if section:
        return section
    # Try underscore form (tool_url_extract)
    underscored = "tool_" + tool_name
    section = get_section(underscored)
    if section:
        return section
    return ""


def get_openai_tool_schema(tool_name: str) -> dict | None:
    """
    Return the OpenAI tool schema dict for a named tool.
    Searches core and plugin LC tools, converting on the fly.
    """
    for lc_tool in CORE_LC_TOOLS + _PLUGIN_TOOLS_LC:
        if lc_tool.name == tool_name:
            return _lc_tool_to_openai_dict(lc_tool)
    return None


def get_core_tools():
    """Return core (always-enabled) tool definitions."""
    import agents as _agents
    return {
        'lc': CORE_LC_TOOLS,
        'executors': {
            'get_system_info':       get_system_info,
            'llm_clean_text':        _agents.llm_clean_text,
            'llm_clean_tool':        _agents.llm_clean_tool,
            'llm_list':              _agents.llm_list,
            'agent_call':            _agents.agent_call,
            'sysprompt_list':        _sysprompt_list_exec,
            'sysprompt_read':        _sysprompt_read_exec,
            'sysprompt_write':       _sysprompt_write_exec,
            'sysprompt_delete':      _sysprompt_delete_exec,
            'sysprompt_copy_dir':    _sysprompt_copy_dir_exec,
            'sysprompt_set_dir':     _sysprompt_set_dir_exec,
            'session':               _session_exec,
            'model':                 _model_exec,
            'reset':                 _reset_exec,
            'help':                  _help_exec,
            'llm_call':              _llm_call_exec,
            'llm_timeout':           _llm_timeout_exec,
            'stream':                _stream_exec,
            'tool_preview_length':   _tool_preview_length_exec,
            'gate_list':             _gate_list_exec,
            'limit_list':            _limit_list_exec,
            'limit_set':             _limit_set_exec,
        }
    }


def get_all_lc_tools() -> list:
    """Get all LangChain StructuredTool objects (core + plugins)."""
    return list(CORE_LC_TOOLS) + list(_PLUGIN_TOOLS_LC)


def get_all_openai_tools() -> list:
    """
    Get all tool definitions in OpenAI dict format (core + plugins).

    Used by try_force_tool_calls() for tool name validation and by
    get_openai_tool_schema() for llm_clean_tool. Derived from CORE_LC_TOOLS
    and _PLUGIN_TOOLS_LC so there is a single source of truth.
    """
    return [_lc_tool_to_openai_dict(t) for t in CORE_LC_TOOLS + _PLUGIN_TOOLS_LC]


# ---------------------------------------------------------------------------
# LangChain StructuredTool ↔ OpenAI dict helpers
# ---------------------------------------------------------------------------

def _lc_tool_to_openai_dict(tool: StructuredTool) -> dict:
    """
    Convert a LangChain StructuredTool to the OpenAI function-calling dict format.

    Used by get_all_openai_tools() so try_force_tool_calls() and
    get_openai_tool_schema() have a single source of truth (CORE_LC_TOOLS).
    """
    schema = tool.args_schema.model_json_schema() if tool.args_schema else {"type": "object", "properties": {}, "required": []}
    # Pydantic v2 puts $defs at top level — flatten for OpenAI compat
    schema.pop("$defs", None)
    schema.pop("title", None)
    # Strip 'title' from each property — Pydantic adds these but they confuse local models
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": schema,
        },
    }


# Tool type mapping for core (always-enabled) tools.
# Plugin tools declare their type via get_gate_tools() metadata.
# Used by execute_tool() for universal rate limiting.
_CORE_TOOL_TYPES: dict[str, str] = {
    "get_system_info":       "system",
    "llm_clean_text":        "llm_call",
    "llm_clean_tool":        "llm_call",
    "at_llm":                "llm_call",
    "llm_list":              "system",
    "agent_call":            "agent_call",
    "sysprompt_list":        "system",
    "sysprompt_read":        "system",
    "sysprompt_write":       "system",
    "sysprompt_delete":      "system",
    "sysprompt_copy_dir":    "system",
    "sysprompt_set_dir":     "system",
    "session":               "system",
    "model":                 "system",
    "reset":                 "system",
    "help":                  "system",
    "llm_call":              "system",
    "llm_timeout":           "system",
    "stream":                "system",
    "tool_preview_length":   "system",
    "gate_list":             "system",
    "limit_list":               "system",
    "limit_set":                "system",
    "outbound_agent_filters":   "system",
}


def get_tool_type(tool_name: str) -> str:
    """
    Return the tool type string for rate-limit bucketing.

    Core tools use _CORE_TOOL_TYPES. Plugin tools use the gate registry type.
    Falls back to 'system' (unlimited by default) for unknown tools.
    """
    if tool_name in _CORE_TOOL_TYPES:
        return _CORE_TOOL_TYPES[tool_name]
    meta = _PLUGIN_GATE_TOOLS.get(tool_name, {})
    return meta.get("type", "system")


def get_tool_executor(tool_name: str):
    """Get executor function for a tool."""
    # Lazy import to avoid circular dependency (agents imports tools, tools needs agents funcs)
    import agents as _agents

    core_executors = {
        'get_system_info':       get_system_info,
        'llm_clean_text':        _agents.llm_clean_text,
        'llm_clean_tool':        _agents.llm_clean_tool,
        'at_llm':                _agents.at_llm,
        'llm_list':              _agents.llm_list,
        'agent_call':            _agents.agent_call,
        'sysprompt_list':        _sysprompt_list_exec,
        'sysprompt_read':        _sysprompt_read_exec,
        'sysprompt_write':       _sysprompt_write_exec,
        'sysprompt_delete':      _sysprompt_delete_exec,
        'sysprompt_copy_dir':    _sysprompt_copy_dir_exec,
        'sysprompt_set_dir':     _sysprompt_set_dir_exec,
        'session':               _session_exec,
        'model':                 _model_exec,
        'reset':                 _reset_exec,
        'help':                  _help_exec,
        'llm_call':              _llm_call_exec,
        'llm_timeout':           _llm_timeout_exec,
        'stream':                _stream_exec,
        'tool_preview_length':   _tool_preview_length_exec,
        'gate_list':             _gate_list_exec,
        'limit_list':            _limit_list_exec,
        'limit_set':             _limit_set_exec,
    }

    if tool_name in core_executors:
        return core_executors[tool_name]

    # Check plugin tools
    return _PLUGIN_TOOL_EXECUTORS.get(tool_name)


# ---------------------------------------------------------------------------
# Core Tool Definitions — LangChain StructuredTool (single source of truth)
#
# Descriptions are the LLM-facing text and are preserved verbatim from the
# former CORE_OPENAI_TOOLS / CORE_GEMINI_TOOL dual-format definitions.
# get_all_openai_tools() derives OpenAI dicts from these on the fly via
# _lc_tool_to_openai_dict() so there is no longer a second copy to maintain.
# ---------------------------------------------------------------------------

from pydantic import BaseModel, Field
from typing import Optional, Literal


class _GetSystemInfoArgs(BaseModel):
    pass  # No arguments — explicit schema prevents LangChain from leaking the docstring into parameters


class _LlmCleanTextArgs(BaseModel):
    model: str = Field(description="Model key name. Use llm_list() to see valid names.")
    prompt: str = Field(description="The complete self-contained prompt. Embed all data the target model needs here.")


class _LlmCleanToolArgs(BaseModel):
    model: str = Field(description="Model key name (e.g., 'nuc11Local'). Use llm_list() to see valid names.")
    tool: str = Field(description="Exact tool name to delegate (e.g., 'url_extract', 'db_query', 'ddgs_search').")
    arguments: str = Field(description="The user request / arguments to pass as the prompt to the target model. Be specific.")


# ---------------------------------------------------------------------------
# Sysprompt tool arg schemas
# ---------------------------------------------------------------------------

class _SyspromptListArgs(BaseModel):
    model: str = Field(
        description="Model key name (use llm_list() to see valid names) or 'self' for the current model."
    )


class _SyspromptReadArgs(BaseModel):
    model: str = Field(
        description="Model key name or 'self' for the current model."
    )
    file: str = Field(
        default="",
        description=(
            "Optional: specific file to read. Omit to read the full assembled prompt. "
            "Can be a bare section name ('behavior'), a full filename ('.system_prompt_behavior'), "
            "or '.system_prompt' for the root file."
        ),
    )


class _SyspromptWriteArgs(BaseModel):
    model: str = Field(description="Model key name or 'self' for the current model.")
    file: str = Field(
        description=(
            "File to write. Can be a bare section name ('behavior'), a full filename "
            "('.system_prompt_behavior'), or '.system_prompt' for the root file."
        )
    )
    data: str = Field(description="Content to write. Overwrites the file (creates if missing).")


class _SyspromptDeleteArgs(BaseModel):
    model: str = Field(description="Model key name or 'self' for the current model.")
    file: str = Field(
        default="",
        description=(
            "File to delete. Omit to delete the ENTIRE directory (also sets system_prompt_folder='none'). "
            "Can be a bare section name or full filename."
        ),
    )


class _SyspromptCopyDirArgs(BaseModel):
    model: str = Field(description="Model key name or 'self' to copy FROM.")
    new_dir: str = Field(
        description="New directory name under system_prompt/ to copy TO (e.g., 'grok4_custom')."
    )


class _SyspromptSetDirArgs(BaseModel):
    model: str = Field(description="Model key name or 'self' for the current model.")
    dir: str = Field(
        description=(
            "Directory name under system_prompt/ to assign (e.g., '000_default', 'grok4_custom'). "
            "Use 'none' to clear the folder assignment."
        )
    )


# ---------------------------------------------------------------------------
# Session / model / reset / help tool arg schemas
# ---------------------------------------------------------------------------

class _SessionArgs(BaseModel):
    action: Literal["list", "delete"] = Field(
        description="'list' to show all sessions, 'delete' to remove a session."
    )
    session_id: str = Field(
        default="",
        description="Session shorthand ID (e.g., '101') or full session ID. Required for 'delete'.",
    )


class _ModelArgs(BaseModel):
    action: Literal["list", "set"] = Field(
        description="'list' to show available models, 'set' to switch the active model."
    )
    model_key: str = Field(
        default="",
        description="Model key to switch to. Required for 'set' action.",
    )


class _ResetArgs(BaseModel):
    pass  # No arguments


class _HelpArgs(BaseModel):
    pass  # No arguments


class _GateListArgs(BaseModel):
    pass  # No arguments


# ---------------------------------------------------------------------------
# Meta-command tool arg schemas (llm_call, llm_timeout, stream, tool_preview_length)
# ---------------------------------------------------------------------------

class _LlmCallArgs(BaseModel):
    action: Literal["list", "set"] = Field(
        description="'list' to show tool_call_available status for all models, 'set' to change it."
    )
    model_key: str = Field(
        default="",
        description="Model key to change. Omit (or use '*') to apply to ALL models. Required when action='set'.",
    )
    enabled: Optional[bool] = Field(
        default=None,
        description="True to enable tool_call_available, False to disable. Required when action='set'.",
    )


class _LlmTimeoutArgs(BaseModel):
    action: Literal["list", "set"] = Field(
        description="'list' to show current timeouts, 'set' to change."
    )
    model_key: str = Field(
        default="",
        description="Model key to set timeout for. Omit (or use '*') to apply to ALL models.",
    )
    seconds: Optional[int] = Field(
        default=None,
        description="Timeout in seconds. Required when action='set'.",
    )


class _StreamArgs(BaseModel):
    action: Literal["get", "set"] = Field(
        description="'get' to show current streaming setting, 'set' to change it."
    )
    enabled: Optional[bool] = Field(
        default=None,
        description="True to enable streaming, False to disable. Required when action='set'.",
    )


class _ToolPreviewLengthArgs(BaseModel):
    action: Literal["get", "set"] = Field(
        description="'get' to show current preview length, 'set' to change it."
    )
    length: Optional[int] = Field(
        default=None,
        description="Number of characters to show (0 = unlimited). Required when action='set'.",
    )


class _AtLlmArgs(BaseModel):
    model: str = Field(
        description="Model key name. Use llm_list() to see valid names."
    )
    prompt: str = Field(
        description="The prompt to send. The model receives the full chat history plus this prompt as a new user turn."
    )


class _LimitListArgs(BaseModel):
    pass  # No arguments


class _LimitSetArgs(BaseModel):
    key: str = Field(
        description=(
            "Limit key to update. Valid keys: 'max_at_llm_depth', 'max_agent_call_depth'. "
            "Use limit_list() first to see current values and descriptions."
        )
    )
    value: int = Field(
        description="New integer value. Must be >= 0. (1 = no recursion/nesting, 0 = fully disabled)"
    )


class _AgentCallArgs(BaseModel):
    agent_url: str = Field(
        description="Base URL of the target agent-mcp instance, e.g. 'http://localhost:8767'. "
                    "The target must have the API client plugin (plugin_client_api) enabled."
    )
    message: str = Field(
        description=(
            "The message or command to send to the target agent. "
            "Can be any text, !command, or @model prefix. "
            "The full response from the target agent is returned.\n\n"
            "CRITICAL: You are the ORCHESTRATOR. Send ONE direct conversational prompt per call. "
            "The remote agent only RESPONDS — it cannot itself call agent_call (depth guard blocks it). "
            "NEVER embed multi-turn instructions in the message (e.g. 'have a 3-turn conversation with me'). "
            "That causes Max swarm depth errors. "
            "For N-turn conversations: make N separate agent_call invocations, each with a single question."
        )
    )
    target_client_id: str = Field(
        default="",
        description="Optional: session name to use on the target agent. "
                    "Omit to auto-generate an isolated swarm session."
    )
    stream: bool = Field(
        default=True,
        description="If True (default), relay the remote agent's tokens in real-time as they "
                    "arrive so Slack and other clients see per-turn progress. "
                    "Set to False to suppress streaming and return only the final result — "
                    "useful when the intermediate tokens would be noisy or are not needed."
    )


# ---------------------------------------------------------------------------
# Sysprompt tool executors
# ---------------------------------------------------------------------------

async def _sysprompt_list_exec(model: str) -> str:
    from prompt import sp_list_files, sp_resolve_model
    from config import LLM_REGISTRY
    from state import current_client_id
    from state import sessions
    cid = current_client_id.get("")
    current_model = sessions.get(cid, {}).get("model", "") if cid else ""
    resolved = sp_resolve_model(model, current_model)
    return sp_list_files(resolved, LLM_REGISTRY)


async def _sysprompt_read_exec(model: str, file: str = "") -> str:
    from prompt import sp_read_prompt, sp_read_file, sp_resolve_model
    from config import LLM_REGISTRY
    from state import current_client_id, sessions
    cid = current_client_id.get("")
    current_model = sessions.get(cid, {}).get("model", "") if cid else ""
    resolved = sp_resolve_model(model, current_model)
    if file:
        return sp_read_file(resolved, file, LLM_REGISTRY)
    return sp_read_prompt(resolved, LLM_REGISTRY)


async def _sysprompt_write_exec(model: str, file: str, data: str) -> str:
    from prompt import sp_write_file, sp_resolve_model
    from config import LLM_REGISTRY
    from state import current_client_id, sessions
    cid = current_client_id.get("")
    current_model = sessions.get(cid, {}).get("model", "") if cid else ""
    resolved = sp_resolve_model(model, current_model)
    return sp_write_file(resolved, file, data, LLM_REGISTRY)


async def _sysprompt_delete_exec(model: str, file: str = "") -> str:
    from prompt import sp_delete_file, sp_delete_directory, sp_resolve_model
    from config import LLM_REGISTRY
    from state import current_client_id, sessions
    cid = current_client_id.get("")
    current_model = sessions.get(cid, {}).get("model", "") if cid else ""
    resolved = sp_resolve_model(model, current_model)
    if file:
        return sp_delete_file(resolved, file, LLM_REGISTRY)
    return sp_delete_directory(resolved, LLM_REGISTRY)


async def _sysprompt_copy_dir_exec(model: str, new_dir: str) -> str:
    from prompt import sp_copy_directory, sp_resolve_model
    from config import LLM_REGISTRY
    from state import current_client_id, sessions
    cid = current_client_id.get("")
    current_model = sessions.get(cid, {}).get("model", "") if cid else ""
    resolved = sp_resolve_model(model, current_model)
    return sp_copy_directory(resolved, new_dir, LLM_REGISTRY)


async def _sysprompt_set_dir_exec(model: str, dir: str) -> str:
    from prompt import sp_set_directory, sp_resolve_model
    from state import current_client_id, sessions
    cid = current_client_id.get("")
    current_model = sessions.get(cid, {}).get("model", "") if cid else ""
    resolved = sp_resolve_model(model, current_model)
    return sp_set_directory(resolved, dir)


# ---------------------------------------------------------------------------
# Session / model / reset / help tool executors
# ---------------------------------------------------------------------------

async def _session_exec(action: str, session_id: str = "") -> str:
    from state import sessions, get_or_create_shorthand_id, get_session_by_shorthand, remove_shorthand_mapping, current_client_id
    cid = current_client_id.get("")

    if action == "list":
        if not sessions:
            return "No active sessions."
        lines = ["Active sessions:"]
        for sid, data in sessions.items():
            marker = " (current)" if sid == cid else ""
            model = data.get("model", "unknown")
            history_len = len(data.get("history", []))
            shorthand_id = get_or_create_shorthand_id(sid)
            peer_ip = data.get("peer_ip")
            ip_str = f", ip={peer_ip}" if peer_ip else ""
            lines.append(f"  ID [{shorthand_id}] {sid}: model={model}, history={history_len} messages{ip_str}{marker}")
        return "\n".join(lines)

    if action == "delete":
        if not session_id:
            return "ERROR: session_id required for 'delete' action."
        # Try shorthand ID
        target_sid = None
        try:
            shorthand_id = int(session_id)
            target_sid = get_session_by_shorthand(shorthand_id)
            if not target_sid:
                return f"Session ID [{shorthand_id}] not found."
        except ValueError:
            target_sid = session_id
        if target_sid in sessions:
            shorthand_id = get_or_create_shorthand_id(target_sid)
            del sessions[target_sid]
            remove_shorthand_mapping(target_sid)
            return f"Deleted session ID [{shorthand_id}]: {target_sid}"
        return f"Session not found: {target_sid}"

    return f"Unknown action '{action}'. Valid: list, delete"


async def _model_exec(action: str, model_key: str = "") -> str:
    from config import LLM_REGISTRY
    from state import current_client_id, sessions, cancel_active_task

    if action == "list":
        lines = ["Available models:"]
        cid = current_client_id.get("")
        current = sessions.get(cid, {}).get("model", "") if cid else ""
        for key, meta in LLM_REGISTRY.items():
            model_id = meta.get("model_id", key)
            marker = " (current)" if key == current else ""
            lines.append(f"  {key:<12} {model_id}{marker}")
        return "\n".join(lines)

    if action == "set":
        if not model_key:
            return "ERROR: model_key required for 'set' action."
        cid = current_client_id.get("")
        if not cid:
            return "ERROR: No active session context for model switch."
        if model_key not in LLM_REGISTRY:
            available = ", ".join(LLM_REGISTRY.keys())
            return f"ERROR: Unknown model '{model_key}'\nAvailable: {available}"
        await cancel_active_task(cid)
        sessions[cid]["model"] = model_key
        return f"Model set to '{model_key}'."

    return f"Unknown action '{action}'. Valid: list, set"


async def _reset_exec() -> str:
    from state import current_client_id, sessions
    cid = current_client_id.get("")
    if not cid or cid not in sessions:
        return "ERROR: No active session context."
    history_len = len(sessions[cid].get("history", []))
    sessions[cid]["history"] = []
    return f"Conversation history cleared ({history_len} messages removed)."


async def _help_exec() -> str:
    from tools import get_all_gate_tools
    gate_tools = get_all_gate_tools()
    tool_lines = []
    for tool_name, meta in sorted(gate_tools.items(), key=lambda x: (x[1].get("type",""), x[0])):
        desc = meta.get("description", "")
        tool_lines.append(f"  {tool_name:<30} - {desc}")
    return (
        "Available commands: Use !help in the chat interface for full command list.\n"
        "Key tool calls (LLM-invocable):\n"
        "  sysprompt_list(model)         - list system prompt files for a model\n"
        "  sysprompt_read(model, file)   - read full prompt or specific file\n"
        "  sysprompt_write(model, file, data) - write a system prompt file\n"
        "  sysprompt_delete(model, file) - delete file or entire folder\n"
        "  sysprompt_copy_dir(model, new_dir) - copy prompt folder\n"
        "  sysprompt_set_dir(model, dir) - assign prompt folder to model\n"
        "  session(action, ...)          - list or delete sessions\n"
        "  model(action, ...)            - list or switch models\n"
        "  reset()                       - clear conversation history\n"
        "  db_query(sql)                 - run SQL\n"
        "  search_ddgs/search_tavily/search_xai/search_google(query) - web search\n"
        "  url_extract(method, url)      - extract web page content\n"
        "  google_drive(operation, ...) - Google Drive CRUD\n"
        "  get_system_info()             - date/time/status\n"
        "  llm_list()                    - list LLM models\n"
        "  llm_clean_text(model, prompt) - call LLM with no context\n"
        "  llm_clean_tool(model, tool, arguments) - delegate tool call\n"
        "  llm_call(action, ...)         - manage model delegation\n"
        "  llm_timeout(action, ...)      - manage delegation timeouts\n"
        "  stream(action, ...)           - control streaming mode\n"
        "  tool_preview_length(action, ...) - control tool output display\n"
        "  agent_call(agent_url, message) - call remote agent\n"
        "\nGated tools (listed below require human approval unless gate is off):\n"
        + "\n".join(tool_lines)
    )


async def _llm_call_exec(action: str, model_key: str = "", enabled: bool = None) -> str:
    from config import LLM_REGISTRY, save_llm_model_field
    if action == "list":
        lines = ["LLM tool_call_available status:"]
        for name, cfg in sorted(LLM_REGISTRY.items()):
            available = "YES" if cfg.get("tool_call_available", False) else "NO"
            timeout = cfg.get("llm_call_timeout", 60)
            desc = cfg.get("description", "")
            lines.append(f"  {name:<14} tool_call={available}  timeout={timeout}s  {desc}")
        return "\n".join(lines)
    if action == "set":
        if enabled is None:
            return "ERROR: 'enabled' required for action='set'."
        targets = [model_key] if model_key and model_key != "*" else list(LLM_REGISTRY.keys())
        changed = []
        for name in targets:
            if name not in LLM_REGISTRY:
                return f"ERROR: Unknown model '{name}'."
            LLM_REGISTRY[name]["tool_call_available"] = enabled
            if save_llm_model_field(name, "tool_call_available", enabled):
                changed.append(name)
        status = "enabled" if enabled else "disabled"
        return f"tool_call_available={status} for: {', '.join(changed)}. Persisted to llm-models.json."
    return f"Unknown action '{action}'. Valid: list, set"


async def _llm_timeout_exec(action: str, model_key: str = "", seconds: int = None) -> str:
    from config import LLM_REGISTRY, save_llm_model_field
    if action == "list":
        lines = ["LLM delegation timeouts:"]
        for name, cfg in sorted(LLM_REGISTRY.items()):
            t = cfg.get("llm_call_timeout", 60)
            available = "YES" if cfg.get("tool_call_available", False) else "NO"
            lines.append(f"  {name:<14} timeout={t}s  tool_call={available}")
        return "\n".join(lines)
    if action == "set":
        if seconds is None or seconds < 1:
            return "ERROR: 'seconds' must be a positive integer for action='set'."
        targets = [model_key] if model_key and model_key != "*" else list(LLM_REGISTRY.keys())
        changed = []
        for name in targets:
            if name not in LLM_REGISTRY:
                return f"ERROR: Unknown model '{name}'."
            LLM_REGISTRY[name]["llm_call_timeout"] = seconds
            if save_llm_model_field(name, "llm_call_timeout", seconds):
                changed.append(name)
        return f"llm_call_timeout={seconds}s for: {', '.join(changed)}. Persisted to llm-models.json."
    return f"Unknown action '{action}'. Valid: list, set"


async def _stream_exec(action: str, enabled: bool = None) -> str:
    from state import current_client_id, sessions
    cid = current_client_id.get("")
    if not cid or cid not in sessions:
        return "ERROR: No active session context."
    if action == "get":
        current = sessions[cid].get("agent_call_stream", True)
        return f"agent_call streaming: {'enabled' if current else 'disabled'}"
    if action == "set":
        if enabled is None:
            return "ERROR: 'enabled' required for action='set'."
        sessions[cid]["agent_call_stream"] = enabled
        return f"agent_call streaming {'enabled' if enabled else 'disabled'}."
    return f"Unknown action '{action}'. Valid: get, set"


async def _tool_preview_length_exec(action: str, length: int = None) -> str:
    from state import current_client_id, sessions
    cid = current_client_id.get("")
    if not cid or cid not in sessions:
        return "ERROR: No active session context."
    if action == "get":
        current = sessions[cid].get("tool_preview_length", 500)
        return f"Tool preview length: {'unlimited' if current == 0 else f'{current} chars'}"
    if action == "set":
        if length is None or length < 0:
            return "ERROR: 'length' must be >= 0 (0 = unlimited) for action='set'."
        sessions[cid]["tool_preview_length"] = length
        return f"Tool preview length set to {'unlimited' if length == 0 else f'{length} chars'}."
    return f"Unknown action '{action}'. Valid: get, set"


async def _gate_list_exec() -> str:
    from state import auto_aidb_state, tool_gate_state
    from tools import get_all_gate_tools

    def perm_str(is_auto: bool, perm: str) -> str:
        """Format one permission: gate=ON/OFF then effect."""
        if is_auto:
            return f"gate=OFF ({perm}=auto-allow)"
        else:
            return f"gate=ON  ({perm}=gated)"

    lines = [
        "Gate status  (gate=ON → requires approval; gate=OFF → auto-allow)",
        "  Runtime toggle: !<tool>_gate_read/write true=gate-ON  false=gate-OFF",
    ]

    # DB gates
    lines.append("\ndb_query (per-table):")
    if not auto_aidb_state:
        lines.append("  (all tables gated by default)")
    else:
        for table in sorted(auto_aidb_state.keys()):
            perms = auto_aidb_state[table]
            label = "(default *)" if table == "*" else ("(metadata)" if table == "__meta__" else table)
            r = perms.get("read", False)
            w = perms.get("write", False)
            lines.append(f"  {label:<20} read: {perm_str(r, 'read')}  write: {perm_str(w, 'write')}")

    # Tool gates — iterate all gate-able tools from registry
    lines.append("\nTool gates:")
    gate_tools = get_all_gate_tools()
    # Core gated tools not in plugin registry
    _CORE_GATED = {
        "at_llm":          ["write"],
        "sysprompt_write": ["write"],
        "session":         ["read", "write"],
        "model":           ["write"],
        "reset":           ["write"],
        "gate_list":       ["read"],
        "limit_list":      ["read"],
        "limit_set":       ["write"],
    }
    all_tools = dict(_CORE_GATED)
    for name, meta in gate_tools.items():
        ops = meta.get("operations", ["read"])
        all_tools[name] = ops
    # db_query is handled separately via auto_aidb_state (per-table), never in Tool gates
    all_tools.pop("db_query", None)
    for tool in sorted(all_tools.keys()):
        ops = all_tools[tool]
        perms = tool_gate_state.get(tool, {})
        default_perms = tool_gate_state.get("*", {})
        parts = []
        for op in ops:
            val = perms.get(op, default_perms.get(op, False))
            parts.append(f"{op}: {perm_str(val, op)}")
        lines.append(f"  {tool:<22} {' '.join(parts)}")

    return "\n".join(lines)


_LIMIT_KEY_DESCRIPTIONS: dict[str, str] = {
    "max_at_llm_depth":    "Max nested at_llm hops before rejection (1 = no recursion)",
    "max_agent_call_depth": "Max nested agent_call hops before rejection (1 = no recursion)",
}


async def _limit_list_exec() -> str:
    from config import load_limits
    limits = load_limits()
    lines = ["Depth / Iteration Limits:"]
    lines.append(f"  {'Key':<26} {'Value':>6}  Description")
    lines.append(f"  {'-'*26} {'-'*6}  {'-'*40}")
    for key, desc in sorted(_LIMIT_KEY_DESCRIPTIONS.items()):
        val = limits.get(key, "(default: 1)")
        lines.append(f"  {key:<26} {str(val):>6}  {desc}")
    lines.append("\nChanges take effect after restarting agent-mcp.py")
    return "\n".join(lines)


async def _limit_set_exec(key: str, value: int) -> str:
    from config import save_limit_field, load_limits
    if key not in _LIMIT_KEY_DESCRIPTIONS:
        valid = ", ".join(sorted(_LIMIT_KEY_DESCRIPTIONS.keys()))
        return f"ERROR: Unknown limit key '{key}'. Valid keys: {valid}"
    if value < 0:
        return "ERROR: Value must be >= 0."
    limits = load_limits()
    old = limits.get(key, 1)
    if save_limit_field(key, value):
        return f"{key}: {old} → {value}. Persisted to llm-models.json. Restart agent-mcp.py for changes to take effect."
    return f"ERROR: Failed to persist {key}={value} to llm-models.json."


async def _outbound_agent_filters_exec() -> str:
    """Return the current OUTBOUND_AGENT_ALLOWED/BLOCKED_COMMANDS configuration."""
    import json as _json, os as _os
    try:
        path = _os.path.join(_os.path.dirname(__file__), "plugins-enabled.json")
        with open(path, "r") as f:
            cfg = _json.load(f)
        api_cfg = cfg.get("plugin_config", {}).get("plugin_client_api", {})
        allowed = api_cfg.get("OUTBOUND_AGENT_ALLOWED_COMMANDS", [])
        blocked = api_cfg.get("OUTBOUND_AGENT_BLOCKED_COMMANDS", [])
    except Exception as e:
        return f"ERROR: Could not read outbound agent filters: {e}"

    lines = ["Outbound agent message filters (applied to agent_call messages):"]
    if allowed:
        lines.append(f"  OUTBOUND_AGENT_ALLOWED_COMMANDS ({len(allowed)} entries):")
        for p in allowed:
            lines.append(f"    - {p}")
        lines.append("  → Messages must start with one of the above prefixes.")
    else:
        lines.append("  OUTBOUND_AGENT_ALLOWED_COMMANDS: [] (empty — all messages permitted)")

    if blocked:
        lines.append(f"  OUTBOUND_AGENT_BLOCKED_COMMANDS ({len(blocked)} entries):")
        for p in blocked:
            lines.append(f"    - {p}")
        lines.append("  → Messages must NOT start with any of the above prefixes.")
    else:
        lines.append("  OUTBOUND_AGENT_BLOCKED_COMMANDS: [] (empty — nothing blocked)")

    return "\n".join(lines)


class _OutboundAgentFiltersArgs(BaseModel):
    pass


def _make_core_lc_tools() -> list:
    """Build CORE_LC_TOOLS after agents module is available (avoids circular import)."""
    import agents as _agents
    return [
        StructuredTool.from_function(
            coroutine=get_system_info,
            name="get_system_info",
            description="Returns current local date, time, and system status.",
            args_schema=_GetSystemInfoArgs,
        ),
        StructuredTool.from_function(
            coroutine=_agents.llm_clean_text,
            name="llm_clean_text",
            description=(
                "Call a target LLM model with a single prompt and absolutely no context. "
                "The target model receives ONLY the prompt text — no system prompt, no chat "
                "history, no tool definitions, and no session state. "
                "The response is the raw text returned by the target model. "
                "Only models with tool_call_available=true may be called. "
                "Use llm_list() first to see which models are available. "
                "Best for: summarization, analysis, or formatting of data you embed in the prompt. "
                "Subject to rate limiting (default: 3 calls per 20 seconds per session)."
            ),
            args_schema=_LlmCleanTextArgs,
        ),
        StructuredTool.from_function(
            coroutine=_agents.llm_clean_tool,
            name="llm_clean_tool",
            description=(
                "Delegate a single tool call to a target LLM with no session context. "
                "The target model receives only the named tool's definition as its system prompt "
                "and the arguments string as the user message. "
                "The server executes the tool call (normal gates apply) and returns the result "
                "to the target model for synthesis. Only the final text is returned to you. "
                "Use this to offload tool execution to a free/local model. "
                "Only models with tool_call_available=true may be called. "
                "Subject to rate limiting (same bucket as llm_clean_text)."
            ),
            args_schema=_LlmCleanToolArgs,
        ),
        StructuredTool.from_function(
            coroutine=_agents.llm_list,
            name="llm_list",
            description=(
                "List all registered LLM models with their details: type, model_id, host, "
                "max_context, tool_call_available status, timeout, and description. "
                "Use this before calling llm_clean_text or llm_clean_tool to identify a suitable target model."
            ),
        ),
        StructuredTool.from_function(
            coroutine=_agents.agent_call,
            name="agent_call",
            description=(
                "Send a single direct message to another agent-mcp instance and return its response. "
                "Use for multi-agent coordination (swarm): delegate tasks, verify answers, or "
                "gather perspectives across agent instances.\n\n"
                "ORCHESTRATION MODEL: YOU are the orchestrator. YOU make repeated agent_call "
                "invocations — one per conversation turn. The remote agent ONLY RESPONDS to the "
                "single message you send; it does NOT itself call agent_call (depth guard blocks "
                "recursion at 1 hop). For an N-turn conversation, make N separate agent_call "
                "calls, then synthesize all responses. NEVER embed multi-turn orchestration in "
                "the message field (e.g. 'have a 5-turn conversation') — that causes immediate "
                "Max swarm depth errors.\n\n"
                "Rate limited: 5 calls per 60 seconds per session. "
                "By default (stream=True) remote tokens are relayed in real-time for live Slack progress. "
                "Set stream=False to suppress streaming and return only the final result."
            ),
            args_schema=_AgentCallArgs,
        ),
        StructuredTool.from_function(
            coroutine=_agents.at_llm,
            name="at_llm",
            description=(
                "Call a named LLM model using the FULL current session context "
                "(system prompt + complete chat history + the given prompt as a new user turn). "
                "Equivalent to the @<model> prefix syntax but usable as a tool call. "
                "The result is NOT added to the session history. "
                "All tool gates are bypassed for the called model (same as @<model>). "
                "Use this to get a second opinion, delegate a sub-question, or query "
                "a specialised model mid-conversation without switching models permanently. "
                "Subject to rate limiting (same bucket as llm_clean_text). "
                "Requires at_llm write gate approval (controlled by !at_llm_gate_write). "
                "Default: gated."
            ),
            args_schema=_AtLlmArgs,
        ),
        # --- Sysprompt management tools ---
        StructuredTool.from_function(
            coroutine=_sysprompt_list_exec,
            name="sysprompt_list",
            description=(
                "List all .system_prompt* files in a model's system prompt folder. "
                "Use model='self' for the current model."
            ),
            args_schema=_SyspromptListArgs,
        ),
        StructuredTool.from_function(
            coroutine=_sysprompt_read_exec,
            name="sysprompt_read",
            description=(
                "Read the system prompt for a model. Omit 'file' to get the full assembled prompt. "
                "Specify 'file' to read a specific section file. Use model='self' for current model."
            ),
            args_schema=_SyspromptReadArgs,
        ),
        StructuredTool.from_function(
            coroutine=_sysprompt_write_exec,
            name="sysprompt_write",
            description=(
                "Overwrite or create a system prompt file for a model. "
                "Use model='self' for the current model. "
                "The file argument can be a section name ('behavior') or full filename ('.system_prompt_behavior'). "
                "WARNING: This overwrites the entire file. Requires gate approval."
            ),
            args_schema=_SyspromptWriteArgs,
        ),
        StructuredTool.from_function(
            coroutine=_sysprompt_delete_exec,
            name="sysprompt_delete",
            description=(
                "Delete a system prompt file (or the entire folder) for a model. "
                "Omit 'file' to delete the ENTIRE directory and set folder='none'. "
                "Use model='self' for the current model. Requires gate approval."
            ),
            args_schema=_SyspromptDeleteArgs,
        ),
        StructuredTool.from_function(
            coroutine=_sysprompt_copy_dir_exec,
            name="sysprompt_copy_dir",
            description=(
                "Copy a model's system prompt folder to a new directory under system_prompt/. "
                "Use model='self' for the current model. Requires gate approval."
            ),
            args_schema=_SyspromptCopyDirArgs,
        ),
        StructuredTool.from_function(
            coroutine=_sysprompt_set_dir_exec,
            name="sysprompt_set_dir",
            description=(
                "Assign a model's system_prompt_folder to an existing directory under system_prompt/. "
                "Use 'none' to clear the folder. Persists to llm-models.json. Requires gate approval."
            ),
            args_schema=_SyspromptSetDirArgs,
        ),
        # --- Session / model / reset / help tools ---
        StructuredTool.from_function(
            coroutine=_session_exec,
            name="session",
            description=(
                "Manage agent sessions. action='list' shows all active sessions. "
                "action='delete' removes a session (requires session_id, can be shorthand integer or full ID). "
                "List requires read gate approval; delete requires write gate approval."
            ),
            args_schema=_SessionArgs,
        ),
        StructuredTool.from_function(
            coroutine=_model_exec,
            name="model",
            description=(
                "Manage the active LLM model. action='list' shows all available models. "
                "action='set' switches the active model for this session (requires model_key). "
                "List is always allowed; set requires write gate approval."
            ),
            args_schema=_ModelArgs,
        ),
        StructuredTool.from_function(
            coroutine=_reset_exec,
            name="reset",
            description=(
                "Clear conversation history for the current session. "
                "Requires write gate approval."
            ),
            args_schema=_ResetArgs,
        ),
        StructuredTool.from_function(
            coroutine=_help_exec,
            name="help",
            description="Return a summary of available commands and tool calls.",
            args_schema=_HelpArgs,
        ),
        # --- Meta-command tools (also accessible via !commands) ---
        StructuredTool.from_function(
            coroutine=_llm_call_exec,
            name="llm_call",
            description=(
                "Manage the tool_call_available flag for LLM models. "
                "action='list' shows all models with their tool_call_available status. "
                "action='set' enables/disables delegation for a model (or all models if model_key is omitted). "
                "Always allowed (no gate)."
            ),
            args_schema=_LlmCallArgs,
        ),
        StructuredTool.from_function(
            coroutine=_llm_timeout_exec,
            name="llm_timeout",
            description=(
                "Manage llm_call_timeout for LLM delegation. "
                "action='list' shows current timeouts. "
                "action='set' updates the timeout in seconds (omit model_key to set all models). "
                "Always allowed (no gate)."
            ),
            args_schema=_LlmTimeoutArgs,
        ),
        StructuredTool.from_function(
            coroutine=_stream_exec,
            name="stream",
            description=(
                "Control agent_call streaming for the current session. "
                "action='get' shows current setting. "
                "action='set' enables or disables real-time token relay. "
                "Always allowed (no gate)."
            ),
            args_schema=_StreamArgs,
        ),
        StructuredTool.from_function(
            coroutine=_tool_preview_length_exec,
            name="tool_preview_length",
            description=(
                "Control how many characters of tool output are shown in chat. "
                "The full result is always sent to the LLM regardless. "
                "action='get' shows current setting. "
                "action='set' changes it (0 = unlimited). "
                "Always allowed (no gate)."
            ),
            args_schema=_ToolPreviewLengthArgs,
        ),
        StructuredTool.from_function(
            coroutine=_gate_list_exec,
            name="gate_list",
            description=(
                "List the current live gate status for all tools and DB tables. "
                "Shows whether each tool requires human approval (gated) or is auto-allowed (gate OFF). "
                "Read-only. Requires read gate approval (controlled by !gate_list_gate_read)."
            ),
            args_schema=_GateListArgs,
        ),
        StructuredTool.from_function(
            coroutine=_limit_list_exec,
            name="limit_list",
            description=(
                "List all configurable depth and iteration limits (e.g. max_at_llm_depth, max_agent_call_depth). "
                "Shows current values and descriptions. Read-only. "
                "Requires read gate approval (controlled by !limit_list_gate_read, default: gated)."
            ),
            args_schema=_LimitListArgs,
        ),
        StructuredTool.from_function(
            coroutine=_limit_set_exec,
            name="limit_set",
            description=(
                "Update a configurable depth or iteration limit and persist it to llm-models.json. "
                "Changes take effect after agent restart. "
                "Valid keys: max_at_llm_depth, max_agent_call_depth. "
                "Requires write gate approval (controlled by !limit_set_gate_write, default: gated)."
            ),
            args_schema=_LimitSetArgs,
        ),
        StructuredTool.from_function(
            coroutine=_outbound_agent_filters_exec,
            name="outbound_agent_filters",
            description=(
                "Show the current outbound agent message filter configuration. "
                "Returns OUTBOUND_AGENT_ALLOWED_COMMANDS and OUTBOUND_AGENT_BLOCKED_COMMANDS lists "
                "that are applied to messages sent via agent_call to remote agents. "
                "Call this before using agent_call if you want to ensure your message will pass the filters. "
                "Always allowed — no gate required."
            ),
            args_schema=_OutboundAgentFiltersArgs,
        ),
    ]


# Populated by agent-mcp.py after plugin registration via update_tool_definitions()
CORE_LC_TOOLS: list = []
