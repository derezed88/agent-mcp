import datetime
import platform
from mcp.server.fastmcp import FastMCP
from langchain_core.tools import StructuredTool

from config import log
from database import execute_sql
from drive import run_drive_op
from search import run_google_search
from prompt import apply_prompt_operation, get_current_prompt, get_section, list_sections

mcp_server = FastMCP("AIOps-DB-Tools")


@mcp_server.tool()
async def db_query(sql: str) -> str:
    """Execute SQL against mymcp MySQL database."""
    return await execute_sql(sql)


@mcp_server.tool()
async def update_system_prompt(
    section_name: str,
    operation: str,
    content: str = "",
    target: str = "",
    confirm_overwrite: bool = False,
) -> str:
    """
    Surgically edit a specific section of the .system_prompt file.

    section_name   : REQUIRED - one of: memory-hierarchy, tool-guardrails, tools,
                     tool-logging, time-bypass, db-guardrails, behavior
    operation      : append | prepend | replace | delete | overwrite
    content        : text to add / replacement text (not needed for delete)
    target         : exact substring to find (required for replace and delete)
    confirm_overwrite : must be True to use the overwrite operation
    """
    try:
        _, msg = apply_prompt_operation(
            section_name=section_name,
            operation=operation,
            content=content,
            target=target,
            confirm_overwrite=confirm_overwrite,
        )
        log.info("System prompt operation '%s' on section '%s': %s", operation, section_name, msg)
        return msg
    except ValueError as exc:
        return f"Prompt update rejected: {exc}"
    except Exception as exc:
        return f"Error updating system prompt: {exc}"


@mcp_server.tool()
async def read_system_prompt(section: str = "") -> str:
    """
    Return the system prompt or a specific section.

    section : optional - empty for full prompt, or specify:
              - Integer index (e.g., "0", "1", "2") for section at that position
              - Section name (e.g., "tools", "behavior") for specific section
    """
    if not section:
        return get_current_prompt()

    section_content = get_section(section)
    if section_content is None:
        sections_info = list_sections()
        section_list = "\n".join(
            f"  [{s['index']}] {s['short-section-name']}: {s['description']}"
            for s in sections_info
        )
        return f"Section '{section}' not found.\n\nAvailable sections:\n{section_list}"

    return section_content


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
            'update_system_prompt': update_system_prompt,
            'read_system_prompt': read_system_prompt,
            'get_system_info': get_system_info,
            'llm_clean_text': _agents.llm_clean_text,
            'llm_clean_tool': _agents.llm_clean_tool,
            'llm_list': _agents.llm_list,
            'agent_call': _agents.agent_call,
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
    "update_system_prompt": "system",
    "read_system_prompt":   "system",
    "get_system_info":      "system",
    "llm_clean_text":       "llm_call",
    "llm_clean_tool":       "llm_call",
    "llm_list":             "system",
    "agent_call":           "agent_call",
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
        'update_system_prompt': update_system_prompt,
        'read_system_prompt':   read_system_prompt,
        'get_system_info':      get_system_info,
        'llm_clean_text':       _agents.llm_clean_text,
        'llm_clean_tool':       _agents.llm_clean_tool,
        'llm_list':             _agents.llm_list,
        'agent_call':           _agents.agent_call,
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


class _UpdateSystemPromptArgs(BaseModel):
    section_name: Literal[
        "memory-hierarchy", "tool-guardrails", "tools",
        "tool-logging", "time-bypass", "db-guardrails", "behavior"
    ] = Field(description="REQUIRED: The section to edit.")
    operation: Literal["append", "prepend", "replace", "delete", "overwrite"] = Field(
        description="The edit operation to perform."
    )
    content: str = Field(
        default="",
        description="The text to add or the replacement text. Pass ONLY the user-specified text — do not invent additional content.",
    )
    target: str = Field(
        default="",
        description="Exact substring to find (required for replace and delete operations).",
    )
    confirm_overwrite: bool = Field(
        default=False,
        description="Must be true to use the overwrite operation. Do not set true unless the user explicitly asked to replace the entire section.",
    )


class _ReadSystemPromptArgs(BaseModel):
    section: str = Field(
        default="",
        description="Optional: Integer index or section name. Omit for full prompt.",
    )


class _GetSystemInfoArgs(BaseModel):
    pass  # No arguments — explicit schema prevents LangChain from leaking the docstring into parameters


class _LlmCleanTextArgs(BaseModel):
    model: str = Field(description="Model key name (e.g., 'nuc11Local', 'gemini25'). Use llm_list() to see valid names.")
    prompt: str = Field(description="The complete self-contained prompt. Embed all data the target model needs here.")


class _LlmCleanToolArgs(BaseModel):
    model: str = Field(description="Model key name (e.g., 'nuc11Local'). Use llm_list() to see valid names.")
    tool: str = Field(description="Exact tool name to delegate (e.g., 'url_extract', 'db_query', 'ddgs_search').")
    arguments: str = Field(description="The user request / arguments to pass as the prompt to the target model. Be specific.")


class _AgentCallArgs(BaseModel):
    agent_url: str = Field(
        description="Base URL of the target agent-mcp instance, e.g. 'http://localhost:8767'. "
                    "The target must have the API client plugin (plugin_client_api) enabled."
    )
    message: str = Field(
        description="The message or command to send to the target agent. "
                    "Can be any text, !command, or @model prefix. "
                    "The full response from the target agent is returned."
    )
    target_client_id: str = Field(
        default="",
        description="Optional: session name to use on the target agent. "
                    "Omit to auto-generate an isolated swarm session."
    )


def _make_core_lc_tools() -> list:
    """Build CORE_LC_TOOLS after agents module is available (avoids circular import)."""
    import agents as _agents
    return [
        StructuredTool.from_function(
            coroutine=update_system_prompt,
            name="update_system_prompt",
            description=(
                "Surgically edit a specific section of the system prompt file using a named operation. "
                "NEVER reconstruct the full prompt yourself — only pass the specific "
                "text the user asked to add, change, or remove.\n\n"
                "REQUIRED: section_name must be one of: memory-hierarchy, tool-guardrails, tools, "
                "tool-logging, time-bypass, db-guardrails, behavior. "
                "You CANNOT create new sections.\n\n"
                "Operations:\n"
                "  append    – add `content` to the end of the section.\n"
                "  prepend   – add `content` to the beginning of the section.\n"
                "  replace   – swap exact `target` string with `content`.\n"
                "  delete    – remove all lines containing `target` substring.\n"
                "  overwrite – replace entire section; ONLY with confirm_overwrite=true "
                "and only when user explicitly requests a full replacement.\n\n"
                "For 'add/append/include/insert' requests → use append.\n"
                "For 'change X to Y' → use replace with target=X, content=Y.\n"
                "For 'remove/delete a rule' → use delete with target=that rule text.\n"
                "NEVER use overwrite unless instructed and confirm_overwrite is true."
            ),
            args_schema=_UpdateSystemPromptArgs,
        ),
        StructuredTool.from_function(
            coroutine=read_system_prompt,
            name="read_system_prompt",
            description=(
                "Return the system prompt or a specific section. "
                "Call with no parameters for full prompt. "
                "Or specify section by integer index (0, 1, 2...) or name "
                "(e.g., 'tools', 'behavior')."
            ),
            args_schema=_ReadSystemPromptArgs,
        ),
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
                "Send a message or command to another agent-mcp instance and return its response. "
                "Use for multi-agent coordination (swarm): delegate tasks, verify answers, or "
                "parallelize work across agent instances. "
                "The target agent processes the message through its full stack (LLM, tools, gates). "
                "Swarm depth is limited to 1 hop to prevent recursion. "
                "Gate approval on the target agent follows that agent's own gate policy — "
                "configure auto_approve_gates on the AgentClient if the target needs tool access. "
                "Rate limited: 5 calls per 60 seconds per session."
            ),
            args_schema=_AgentCallArgs,
        ),
    ]


# Populated by agent-mcp.py after plugin registration via update_tool_definitions()
CORE_LC_TOOLS: list = []
