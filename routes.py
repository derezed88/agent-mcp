import asyncio
import json
from typing import AsyncGenerator
from starlette.requests import Request
from starlette.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from config import log, LLM_REGISTRY, DEFAULT_MODEL, SYSTEM_PROMPT_FILE, save_llm_model_field
from state import sessions, get_queue, push_tok, push_done, pending_gates, auto_aidb_state, tool_gate_state, active_tasks, cancel_active_task
from database import execute_sql
from prompt import get_current_prompt, load_system_prompt, get_section, list_sections
from agents import dispatch_llm
from tools import get_gate_tools_by_type, get_all_gate_tools

# Context flag for batch command processing
_batch_mode = {}  # client_id -> bool

async def conditional_push_done(client_id: str):
    """Only call push_done if not in batch mode"""
    if not _batch_mode.get(client_id, False):
        await push_done(client_id)

async def cmd_help(client_id: str):
    # Build dynamic AI tools section from gate registry
    gate_tools = get_all_gate_tools()
    tool_lines = []
    for tool_name, meta in sorted(gate_tools.items(), key=lambda x: (x[1].get("type",""), x[0])):
        desc = meta.get("description", "")
        tool_lines.append(f"  {tool_name:<30} - {desc}")
    tool_lines.append("  update_system_prompt           - edit specific section of system prompt (write gate)")
    tool_lines.append("  read_system_prompt             - read full prompt or section (auto-allowed, no gate)")
    tool_lines.append("  get_system_info                - read local date/time (auto-allowed, no gate)")

    # Search/extract tool names for the autogate hint
    search_tools = get_gate_tools_by_type("search")
    extract_tools = get_gate_tools_by_type("extract")
    search_hint = ""
    if search_tools:
        search_hint += f"  (use '!autogate search true' to toggle gate for: {', '.join(search_tools)})\n"
    if extract_tools:
        search_hint += f"  (use '!autogate extract true' to toggle gate for: {', '.join(extract_tools)})\n"

    help_text = (
        "Available commands:\n"
        "  !model                                    - list available models (current marked)\n"
        "  !model <key>                              - switch active LLM\n"
        "  !input_lines <n>                          - resize input area (client-side)\n"
        "  !reset                                    - clear conversation history\n"
        "  !db <sql>                                 - run SQL directly (no LLM, no gate)\n"
        "  !help                                     - this help\n"
        "\n"
        "Gate Management:\n"
        "  !autoAIdb <table> [read|write] <t|f>      - toggle gate for specific DB table\n"
        "  !autoAIdb <read|write> <t|f>              - set DEFAULT for all tables (wildcard)\n"
        "  !autoAIdb \\_\\_meta\\_\\_ read <t|f>             - toggle gate for metadata tables\n"
        "  !autoAIdb status                          - show current DB gate settings\n"
        "  !autogate search <t|f>                    - toggle gate for ALL search tools\n"
        "  !autogate <search_tool> <t|f>             - toggle gate for one search tool\n"
        "  !autogate extract <t|f>                   - toggle gate for ALL extract tools\n"
        "  !autogate <extract_tool> <t|f>            - toggle gate for one extract tool\n"
        "  !autogate drive [read|write] <t|f>        - toggle gate for Drive (read/write separate)\n"
        "  !autogate <read|write> <t|f>              - set DEFAULT for all tools (wildcard)\n"
        "  !autogate status                          - show current tool gate settings\n"
        "  !autoAISysPrompt [read|write] <t|f>       - toggle gate for system prompt operations\n"
        "  !autoAISysPrompt                          - show current system prompt gate settings\n"
        "\n"
        "System Prompt:\n"
        "  !sysprompt                                - show the current live system prompt\n"
        "  !sysprompt reload                         - reload .system_prompt from disk\n"
        "  !read_system_prompt                       - show the full assembled system prompt\n"
        "  !read_system_prompt <index>               - show section by index (0, 1, 2...)\n"
        "  !read_system_prompt <name>                - show section by name (tools, behavior, etc.)\n"
        "\n"
        "Session Management:\n"
        "  !session                                  - list all active sessions\n"
        "  !session <ID> delete                      - delete a session from server\n"
        "  !tool_preview_length [n]                  - get/set tool result display limit (0=unlimited, default=500)\n"
        "  !gate_preview_length [n]                  - get/set gate popup preview limit (client-side)\n"
        "\n"
        "LLM Delegation:\n"
        "  !llm_call                                 - list models with tool_call_available status\n"
        "  !llm_call <model> <true|false>            - enable/disable delegation for a model\n"
        "  !llm_call <true|false>                    - set tool_call_available for ALL models\n"
        "  !llm_timeout <model> <seconds>            - set llm_call_timeout for a model\n"
        "  !llm_timeout <seconds>                    - set timeout for ALL models\n"
        "  !llm_timeout                              - list current timeouts\n"
        "  !stream <true|false>                      - enable/disable streaming of agent_call tokens to this client\n"
        "  !stream                                   - show current agent_call streaming setting\n"
        "\n"
        "AI tools (require human gate approval unless noted):\n"
        + "\n".join(tool_lines) + "\n"
        + "  llm_clean_text(model, prompt)             - send prompt to target LLM with no context (text only)\n"
        "  llm_clean_tool(model, tool, arguments)    - delegate one tool call to target LLM (gates apply)\n"
        "  llm_list()                                - list models and tool_call_available status\n"
        + search_hint
    )
    await push_tok(client_id, help_text)
    await conditional_push_done(client_id)

async def cmd_sysprompt(client_id: str, arg: str):
    """Show or reload the system prompt."""
    arg = arg.strip()
    if arg == "reload":
        p = load_system_prompt()
        await push_tok(client_id, f"System prompt reloaded from disk ({len(p)} chars).")
    elif arg == "":
        await push_tok(client_id, f"--- Current system prompt ---\n{get_current_prompt()}")
    else:
        await push_tok(client_id,
            f"ERROR: Unknown argument '{arg}'\n"
            "Usage: !sysprompt [reload]\n"
            "  !sysprompt        - show current prompt\n"
            "  !sysprompt reload - reload from .system_prompt file")
    await conditional_push_done(client_id)

async def cmd_autoaidb(client_id: str, args: str):
    """
    Toggle human gate for database operations per table.
    Usage: !autoAIdb <table> [read|write] <true|false>
    """
    parts = args.split()

    # Validation helpers
    def is_valid_bool(s: str) -> bool:
        return s in ("true", "1", "yes", "false", "0", "no")

    def parse_bool(s: str) -> bool:
        return s in ("true", "1", "yes")

    # Show status
    if len(parts) == 1 and parts[0].lower() == "status":
        if not auto_aidb_state:
            await push_tok(client_id, "No database gate settings configured (all tables gated by default)")
        else:
            lines = ["Database gate settings (set this session — resets on restart):"]
            for key, perms in auto_aidb_state.items():
                label = "(wildcard — applies to all tables)" if key == "*" else \
                        "(metadata: SHOW/DESCRIBE)" if key == "__meta__" else key
                read_s = "auto-allow" if perms.get("read") else "gated"
                write_s = "auto-allow" if perms.get("write") else "gated"
                lines.append(f"  {label}: read={read_s}, write={write_s}")
            await push_tok(client_id, "\n".join(lines))
        await conditional_push_done(client_id)
        return

    # Format: !autoAIdb <table> <read|write> <true|false>
    if len(parts) == 3:
        table, perm_type, flag = parts[0].lower(), parts[1].lower(), parts[2].lower()

        # Validate permission type
        if perm_type not in ("read", "write"):
            await push_tok(client_id,
                f"ERROR: Permission type must be 'read' or 'write', got '{perm_type}'\n"
                f"Usage: !autoAIdb {table} <read|write> <true|false>")
            await conditional_push_done(client_id)
            return

        # Validate boolean
        if not is_valid_bool(flag):
            await push_tok(client_id,
                f"ERROR: Invalid value '{flag}'. Must be one of: true, false, 1, 0, yes, no\n"
                f"Usage: !autoAIdb {table} {perm_type} <true|false>")
            await conditional_push_done(client_id)
            return

        if table not in auto_aidb_state:
            auto_aidb_state[table] = {"read": False, "write": False}

        is_enabled = parse_bool(flag)
        auto_aidb_state[table][perm_type] = is_enabled
        status = "auto-allow (gate OFF)" if is_enabled else "gated (gate ON)"
        await push_tok(client_id, f"autoAIdb {table} {perm_type}: {status}")

    # Two arguments: Could be <table> <true|false> OR <read|write> <true|false>
    elif len(parts) == 2:
        arg1, arg2 = parts[0].lower(), parts[1].lower()

        # Check if first arg is a permission type (read/write)
        # Special case: Allow setting default for ALL tables
        if arg1 in ("read", "write"):
            # Validate boolean
            if not is_valid_bool(arg2):
                await push_tok(client_id,
                    f"ERROR: Invalid value '{arg2}'. Must be one of: true, false, 1, 0, yes, no\n"
                    f"Usage: !autoAIdb {arg1} <true|false>")
                await conditional_push_done(client_id)
                return

            # Set default for ALL tables (using special "*" key)
            perm_type = arg1
            is_enabled = parse_bool(arg2)

            # Apply to the default wildcard
            if "*" not in auto_aidb_state:
                auto_aidb_state["*"] = {"read": False, "write": False}
            auto_aidb_state["*"][perm_type] = is_enabled

            # Also apply to __meta__ (system queries like SHOW TABLES, DESCRIBE)
            # This ensures metadata queries inherit the same permission
            if "__meta__" not in auto_aidb_state:
                auto_aidb_state["__meta__"] = {"read": False, "write": False}
            auto_aidb_state["__meta__"][perm_type] = is_enabled

            status = "auto-allow (gate OFF)" if is_enabled else "gated (gate ON)"
            await push_tok(client_id,
                f"autoAIdb DEFAULT {perm_type}: {status}\n"
                f"This affects all tables AND metadata queries (SHOW/DESCRIBE) without specific settings.\n"
                f"To set specific tables: !autoAIdb <table> {perm_type} <true|false>")
            await conditional_push_done(client_id)
            return

        # Legacy format: !autoAIdb <table> <true|false> - sets both read and write
        table, flag = arg1, arg2

        # Validate boolean
        if not is_valid_bool(flag):
            await push_tok(client_id,
                f"ERROR: Invalid value '{flag}'. Must be one of: true, false, 1, 0, yes, no\n"
                f"Usage: !autoAIdb {table} <true|false>")
            await conditional_push_done(client_id)
            return

        is_enabled = parse_bool(flag)
        auto_aidb_state[table] = {"read": is_enabled, "write": is_enabled}
        status = "auto-allow (gate OFF)" if is_enabled else "gated (gate ON)"
        await push_tok(client_id, f"autoAIdb {table} read+write: {status}")

    else:
        await push_tok(client_id,
            "ERROR: Invalid arguments\n"
            "Usage: !autoAIdb <table> [read|write] <true|false>\n"
            "   or: !autoAIdb status\n"
            "Examples:\n"
            "  !autoAIdb person read true        - auto-allow SELECT on person table\n"
            "  !autoAIdb __meta__ read true      - auto-allow SHOW/DESCRIBE queries\n"
            "  !autoAIdb users write false       - require gate for INSERT/UPDATE/DELETE\n"
            "  !autoAIdb logs true               - auto-allow both read and write")

    await conditional_push_done(client_id)

async def cmd_autogate(client_id: str, args: str):
    """
    Toggle human gate for search and drive tools.

    Search tools are read-only — no read/write distinction:
      !autogate search <true|false>            - toggle gate for ALL loaded search tools
      !autogate <search_tool_name> <true|false> - toggle gate for one specific search tool
      !autogate search                          - show status of all search tools

    Drive has separate read/write gates:
      !autogate drive [read|write] <true|false> - toggle gate for drive operations
      !autogate drive                            - show drive gate status

    Other:
      !autogate status                           - show all tool gate settings
      !autogate <read|write> <true|false>        - set DEFAULT for all tools (wildcard)
    """
    parts = args.split()

    def is_valid_bool(s: str) -> bool:
        return s in ("true", "1", "yes", "false", "0", "no")

    def parse_bool(s: str) -> bool:
        return s in ("true", "1", "yes")

    # Resolve alias to list of canonical tool names.
    # Returns (tool_names, tool_group) or (None, None) if unknown.
    # tool_group: "search" | "extract" | "drive" | None (drive is not read-only)
    def resolve_tools(alias: str):
        alias = alias.lower()
        search_tools = get_gate_tools_by_type("search")
        extract_tools = get_gate_tools_by_type("extract")
        if alias == "search":
            return search_tools, "search"
        if alias in search_tools:
            return [alias], "search"
        if alias == "extract":
            return extract_tools, "extract"
        if alias in extract_tools:
            return [alias], "extract"
        if alias in ("drive", "google_drive"):
            return ["google_drive"], None
        return None, None

    def valid_tool_list() -> str:
        """Build a human-readable list of valid tool names from the registry."""
        search_tools = get_gate_tools_by_type("search")
        extract_tools = get_gate_tools_by_type("extract")
        drive_tools = get_gate_tools_by_type("drive")
        parts_list = []
        if search_tools:
            parts_list.append(f"search (expands to: {', '.join(search_tools)})")
            parts_list.extend(search_tools)
        if extract_tools:
            parts_list.append(f"extract (expands to: {', '.join(extract_tools)})")
            parts_list.extend(extract_tools)
        if drive_tools:
            parts_list.append("drive (or google_drive)")
        return ", ".join(parts_list) if parts_list else "(no tools registered)"

    # --- status ---
    if len(parts) == 1 and parts[0].lower() == "status":
        default_perms = tool_gate_state.get("*", {})
        lines = ["Tool gate settings:"]

        search_tools = get_gate_tools_by_type("search")
        extract_tools = get_gate_tools_by_type("extract")

        # Search tools
        if search_tools:
            lines.append("  Search tools (read-only):")
            for t in search_tools:
                perms = tool_gate_state.get(t, {})
                read_val = perms.get("read", default_perms.get("read", False))
                src = "(specific)" if t in tool_gate_state else "(default)"
                read_str = "auto-allow (gate OFF)" if read_val else "gated (gate ON)"
                lines.append(f"    {t} {src}: {read_str}")

        # Extract tools
        if extract_tools:
            lines.append("  Extract tools (read-only):")
            for t in extract_tools:
                perms = tool_gate_state.get(t, {})
                read_val = perms.get("read", default_perms.get("read", False))
                src = "(specific)" if t in tool_gate_state else "(default)"
                read_str = "auto-allow (gate OFF)" if read_val else "gated (gate ON)"
                lines.append(f"    {t} {src}: {read_str}")

        # Drive tool
        lines.append("  Drive tool (read+write):")
        drive_perms = tool_gate_state.get("google_drive", {})
        read_val  = drive_perms.get("read",  default_perms.get("read",  False))
        write_val = drive_perms.get("write", default_perms.get("write", False))
        src = "(specific)" if "google_drive" in tool_gate_state else "(default)"
        read_str  = "auto-allow (gate OFF)" if read_val  else "gated (gate ON)"
        write_str = "auto-allow (gate OFF)" if write_val else "gated (gate ON)"
        lines.append(f"    google_drive {src}: read={read_str}, write={write_str}")

        # System prompt tools
        lines.append("  System prompt tools:")
        for t in ("read_system_prompt", "update_system_prompt"):
            perms = tool_gate_state.get(t, {})
            r = perms.get("read",  default_perms.get("read",  False))
            w = perms.get("write", default_perms.get("write", False))
            src = "(specific)" if t in tool_gate_state else "(default)"
            r_str = "auto-allow (gate OFF)" if r else "gated (gate ON)"
            w_str = "auto-allow (gate OFF)" if w else "gated (gate ON)"
            lines.append(f"    {t} {src}: read={r_str}, write={w_str}")

        # Wildcard default (if set)
        if "*" in tool_gate_state:
            wc = tool_gate_state["*"]
            r_str = "auto-allow (gate OFF)" if wc.get("read",  False) else "gated (gate ON)"
            w_str = "auto-allow (gate OFF)" if wc.get("write", False) else "gated (gate ON)"
            lines.append(f"  DEFAULT (*): read={r_str}, write={w_str}")

        await push_tok(client_id, "\n".join(lines))
        await conditional_push_done(client_id)
        return

    # --- group status: !autogate search  OR  !autogate extract  OR  !autogate drive ---
    if len(parts) == 1 and parts[0].lower() in ("search", "extract", "drive"):
        group = parts[0].lower()
        if group == "search":
            tools_to_show = get_gate_tools_by_type("search")
        elif group == "extract":
            tools_to_show = get_gate_tools_by_type("extract")
        else:
            tools_to_show = ["google_drive"]
        default_perms = tool_gate_state.get("*", {})
        lines = [f"Gate status for {group}:"]
        for t in tools_to_show:
            perms = tool_gate_state.get(t, {})
            read_val = perms.get("read", default_perms.get("read", False))
            src = "(specific)" if t in tool_gate_state else "(default)"
            if group in ("search", "extract"):
                read_str = "auto-allow (gate OFF)" if read_val else "gated (gate ON)"
                lines.append(f"  {t} {src}: {read_str}")
            else:
                write_val = perms.get("write", default_perms.get("write", False))
                read_str  = "auto-allow (gate OFF)" if read_val  else "gated (gate ON)"
                write_str = "auto-allow (gate OFF)" if write_val else "gated (gate ON)"
                lines.append(f"  {t} {src}: read={read_str}, write={write_str}")
        await push_tok(client_id, "\n".join(lines))
        await conditional_push_done(client_id)
        return

    # --- 3 parts: !autogate <tool> <read|write> <true|false>  (drive only needs read/write) ---
    if len(parts) == 3:
        tool_alias, perm_type, flag = parts[0].lower(), parts[1].lower(), parts[2].lower()

        tool_names, tool_group = resolve_tools(tool_alias)
        if not tool_names:
            await push_tok(client_id,
                f"ERROR: Unknown tool '{tool_alias}'\n"
                f"Valid tools: {valid_tool_list()}")
            await conditional_push_done(client_id)
            return

        # Search and extract tools are read-only — reject write specification
        if tool_group in ("search", "extract"):
            await push_tok(client_id,
                f"ERROR: {tool_group.title()} tools are read-only. Use:\n"
                f"  !autogate {tool_alias} <true|false>")
            await conditional_push_done(client_id)
            return

        if perm_type not in ("read", "write"):
            await push_tok(client_id,
                f"ERROR: Permission type must be 'read' or 'write', got '{perm_type}'\n"
                f"Usage: !autogate {tool_alias} <read|write> <true|false>")
            await conditional_push_done(client_id)
            return

        if not is_valid_bool(flag):
            await push_tok(client_id,
                f"ERROR: Invalid value '{flag}'. Must be one of: true, false, 1, 0, yes, no\n"
                f"Usage: !autogate {tool_alias} {perm_type} <true|false>")
            await conditional_push_done(client_id)
            return

        is_auto = parse_bool(flag)
        for tool_name in tool_names:
            if tool_name not in tool_gate_state:
                tool_gate_state[tool_name] = {"read": False, "write": False}
            tool_gate_state[tool_name][perm_type] = is_auto

        status_str = "auto-allow (gate OFF)" if is_auto else "gated (gate ON)"
        await push_tok(client_id, f"autogate {tool_alias} {perm_type}: {status_str}")

    # --- 2 parts: !autogate <tool|read|write> <true|false> ---
    elif len(parts) == 2:
        arg1, arg2 = parts[0].lower(), parts[1].lower()

        # Wildcard default: !autogate <read|write> <true|false>
        if arg1 in ("read", "write"):
            if not is_valid_bool(arg2):
                await push_tok(client_id,
                    f"ERROR: Invalid value '{arg2}'. Must be one of: true, false, 1, 0, yes, no\n"
                    f"Usage: !autogate {arg1} <true|false>")
                await conditional_push_done(client_id)
                return

            perm_type = arg1
            is_auto = parse_bool(arg2)
            if "*" not in tool_gate_state:
                tool_gate_state["*"] = {"read": False, "write": False}
            tool_gate_state["*"][perm_type] = is_auto

            status_str = "auto-allow (gate OFF)" if is_auto else "gated (gate ON)"
            await push_tok(client_id,
                f"autogate DEFAULT {perm_type}: {status_str}\n"
                f"This affects all tools without specific settings.")
            await conditional_push_done(client_id)
            return

        # Tool toggle: !autogate <tool> <true|false>
        tool_alias, flag = arg1, arg2
        tool_names, tool_group = resolve_tools(tool_alias)
        if not tool_names:
            await push_tok(client_id,
                f"ERROR: Unknown tool '{tool_alias}'\n"
                f"Valid tools: {valid_tool_list()}")
            await conditional_push_done(client_id)
            return

        if not is_valid_bool(flag):
            await push_tok(client_id,
                f"ERROR: Invalid value '{flag}'. Must be one of: true, false, 1, 0, yes, no\n"
                f"Usage: !autogate {tool_alias} <true|false>")
            await conditional_push_done(client_id)
            return

        is_auto = parse_bool(flag)
        if tool_group in ("search", "extract"):
            # Search/extract tools: only toggle the read gate
            for tool_name in tool_names:
                if tool_name not in tool_gate_state:
                    tool_gate_state[tool_name] = {"read": False, "write": False}
                tool_gate_state[tool_name]["read"] = is_auto
            status_str = "auto-allow (gate OFF)" if is_auto else "gated (gate ON)"
            group_names = ", ".join(tool_names)
            label = f"{tool_group} ({group_names})" if tool_alias == tool_group else tool_alias
            await push_tok(client_id, f"autogate {label}: {status_str}")
        else:
            # Drive: toggle both read and write
            for tool_name in tool_names:
                tool_gate_state[tool_name] = {"read": is_auto, "write": is_auto}
            status_str = "auto-allow (gate OFF)" if is_auto else "gated (gate ON)"
            await push_tok(client_id, f"autogate {tool_alias} read+write: {status_str}")

    else:
        search_tools = get_gate_tools_by_type("search")
        extract_tools = get_gate_tools_by_type("extract")
        example_search = search_tools[0] if search_tools else "ddgs_search"
        example_extract = extract_tools[0] if extract_tools else "url_extract"
        await push_tok(client_id,
            "ERROR: Invalid arguments\n"
            "Search tools (read-only):\n"
            "  !autogate search <true|false>        - toggle gate for ALL search tools\n"
            f"  !autogate {example_search} <true|false>  - toggle gate for one search tool\n"
            "  !autogate search                     - show search gate status\n"
            "Extract tools (read-only):\n"
            "  !autogate extract <true|false>       - toggle gate for ALL extract tools\n"
            f"  !autogate {example_extract} <true|false>  - toggle gate for one extract tool\n"
            "  !autogate extract                    - show extract gate status\n"
            "Drive tool (read+write):\n"
            "  !autogate drive [read|write] <true|false>\n"
            "  !autogate drive                      - show drive gate status\n"
            "Other:\n"
            "  !autogate status                     - show all gate settings\n"
            "  !autogate <read|write> <true|false>  - set DEFAULT for all tools")

    await conditional_push_done(client_id)

async def cmd_read_system_prompt(client_id: str, arg: str = ""):
    """
    Read system prompt or specific section.
    No arg: full prompt
    Arg = index or section name: specific section
    """
    if not arg:
        await push_tok(client_id, f"--- Cached system prompt ---\n{get_current_prompt()}")
    else:
        section_content = get_section(arg)
        if section_content is None:
            sections_info = list_sections()
            section_list = "\n".join(
                f"  [{s['index']}] {s['short-section-name']}: {s['description']}"
                for s in sections_info
            )
            await push_tok(
                client_id,
                f"Section '{arg}' not found.\n\nAvailable sections:\n{section_list}"
            )
        else:
            await push_tok(client_id, f"--- Section: {arg} ---\n{section_content}")
    await conditional_push_done(client_id)


async def cmd_autoaisysprompt(client_id: str, arg: str):
    """
    Toggle human gate for update_system_prompt and read_system_prompt tools.
    Usage: !autoAISysPrompt [read|write] <true|false>
    """
    parts = arg.split()

    # Initialize if needed
    if "update_system_prompt" not in tool_gate_state:
        tool_gate_state["update_system_prompt"] = {"read": False, "write": False}
    if "read_system_prompt" not in tool_gate_state:
        tool_gate_state["read_system_prompt"] = {"read": False, "write": False}

    # No args - show current status
    if not arg:
        read_perms = tool_gate_state.get("read_system_prompt", {})
        write_perms = tool_gate_state.get("update_system_prompt", {})
        read_status = "OFF (auto-allow)" if read_perms.get("read", False) else "ON (human gates)"
        write_status = "OFF (auto-allow)" if write_perms.get("write", False) else "ON (human gates)"
        await push_tok(client_id,
            f"autoAISysPrompt read:  {read_status}\n"
            f"autoAISysPrompt write: {write_status}")
        await conditional_push_done(client_id)
        return

    # Validate boolean values
    def is_valid_bool(s: str) -> bool:
        return s in ("true", "1", "yes", "false", "0", "no")

    def parse_bool(s: str) -> bool:
        return s in ("true", "1", "yes")

    # Format: !autoAISysPrompt <read|write> <true|false>
    if len(parts) == 2:
        perm_type, flag = parts[0].lower(), parts[1].lower()

        # Validate permission type
        if perm_type not in ("read", "write"):
            await push_tok(client_id,
                "ERROR: Permission type must be 'read' or 'write'\n"
                "Usage: !autoAISysPrompt [read|write] <true|false>")
            await conditional_push_done(client_id)
            return

        # Validate boolean
        if not is_valid_bool(flag):
            await push_tok(client_id,
                f"ERROR: Invalid value '{flag}'. Must be one of: true, false, 1, 0, yes, no\n"
                f"Usage: !autoAISysPrompt {perm_type} <true|false>")
            await conditional_push_done(client_id)
            return

        is_enabled = parse_bool(flag)
        gate_status = "OFF (auto-allow)" if is_enabled else "ON (human gates)"

        if perm_type == "read":
            tool_gate_state["read_system_prompt"]["read"] = is_enabled
            await push_tok(client_id, f"autoAISysPrompt read: {gate_status}")
        else:  # write
            tool_gate_state["update_system_prompt"]["write"] = is_enabled
            await push_tok(client_id, f"autoAISysPrompt write: {gate_status}")

    # One argument: Could be <true|false> (legacy) OR <read|write> (error - missing bool)
    elif len(parts) == 1:
        arg = parts[0].lower()

        # Check if this is a permission type without a boolean
        if arg in ("read", "write"):
            await push_tok(client_id,
                f"ERROR: Missing boolean value\n"
                f"Usage: !autoAISysPrompt {arg} <true|false>\n"
                f"Example: !autoAISysPrompt {arg} true")
            await conditional_push_done(client_id)
            return

        # Legacy format: !autoAISysPrompt <true|false> (sets write only)
        flag = arg

        if not is_valid_bool(flag):
            await push_tok(client_id,
                f"ERROR: Invalid value '{flag}'. Must be one of: true, false, 1, 0, yes, no\n"
                "Usage: !autoAISysPrompt [read|write] <true|false>")
            await conditional_push_done(client_id)
            return

        is_enabled = parse_bool(flag)
        tool_gate_state["update_system_prompt"]["write"] = is_enabled
        gate_status = "OFF (auto-allow)" if is_enabled else "ON (human gates)"
        await push_tok(client_id, f"autoAISysPrompt write: {gate_status}")

    else:
        await push_tok(client_id,
            "ERROR: Invalid arguments\n"
            "Usage: !autoAISysPrompt [read|write] <true|false>\n"
            "Examples:\n"
            "  !autoAISysPrompt read true   - auto-allow read_system_prompt\n"
            "  !autoAISysPrompt write false - require gate for update_system_prompt\n"
            "  !autoAISysPrompt             - show current settings")

    await conditional_push_done(client_id)


async def cmd_list_models(client_id: str, current: str):
    lines = ["Available models:"]
    for key, meta in LLM_REGISTRY.items():
        model_id = meta.get("model_id", key)
        marker = " (current)" if key == current else ""
        lines.append(f"  {key:<12} {model_id}{marker}")
    await push_tok(client_id, "\n".join(lines))
    await conditional_push_done(client_id)

async def cmd_set_model(client_id: str, key: str, session: dict):
    """Set the active LLM model for this session."""
    if not key or not key.strip():
        await push_tok(client_id,
            "ERROR: Model name required\n"
            "Usage: !model <model_name>\n"
            "Use !model to list available models")
        await conditional_push_done(client_id)
        return

    key = key.strip()
    if key in LLM_REGISTRY:
        await cancel_active_task(client_id)
        session["model"] = key
        await push_tok(client_id, f"Model set to '{key}'.")
    else:
        available = ", ".join(LLM_REGISTRY.keys())
        await push_tok(client_id,
            f"ERROR: Unknown model '{key}'\n"
            f"Available models: {available}\n"
            f"Use !model to list all models")
    await conditional_push_done(client_id)

async def cmd_db(client_id: str, sql: str):
    """Execute SQL directly without LLM or human gate."""
    sql = sql.strip()
    if not sql:
        await push_tok(client_id,
            "ERROR: SQL query required\n"
            "Usage: !db <SQL query>\n"
            "Examples:\n"
            "  !db SELECT * FROM person\n"
            "  !db SHOW TABLES\n"
            "  !db DESCRIBE users")
        await conditional_push_done(client_id)
        return

    try:
        result = await execute_sql(sql)
        await push_tok(client_id, result)
    except Exception as exc:
        await push_tok(client_id, f"ERROR: Database query failed\n{exc}")
    await conditional_push_done(client_id)

async def cmd_reset(client_id: str, session: dict):
    """Clear conversation history for current session."""
    history_len = len(session.get("history", []))
    session["history"] = []
    await push_tok(client_id, f"Conversation history cleared ({history_len} messages removed).")
    await conditional_push_done(client_id)

async def cmd_session(client_id: str, arg: str):
    """
    Manage sessions: list, attach, or delete.
    Usage:
      !session                - list all sessions (current marked)
      !session <ID> attach    - switch to different session
      !session <ID> delete    - delete a session
    """
    from state import sessions, get_or_create_shorthand_id, get_session_by_shorthand, remove_shorthand_mapping

    parts = arg.split()

    if not arg:
        # List all sessions with shorthand IDs
        if not sessions:
            await push_tok(client_id, "No active sessions.")
        else:
            lines = ["Active sessions:"]
            for sid, data in sessions.items():
                marker = " (current)" if sid == client_id else ""
                model = data.get("model", "unknown")
                history_len = len(data.get("history", []))
                shorthand_id = get_or_create_shorthand_id(sid)
                peer_ip = data.get("peer_ip")
                ip_str = f", ip={peer_ip}" if peer_ip else ""
                lines.append(f"  ID [{shorthand_id}] {sid}: model={model}, history={history_len} messages{ip_str}{marker}")
            await push_tok(client_id, "\n".join(lines))
    elif len(parts) == 2:
        target_arg, action = parts[0], parts[1].lower()

        # Try to parse as shorthand ID (integer)
        target_sid = None
        try:
            shorthand_id = int(target_arg)
            target_sid = get_session_by_shorthand(shorthand_id)
            if not target_sid:
                await push_tok(client_id, f"Session ID [{shorthand_id}] not found.")
                await conditional_push_done(client_id)
                return
        except ValueError:
            # Not an integer, treat as full session ID
            target_sid = target_arg

        if action == "attach":
            await push_tok(client_id, f"ERROR: Session switching not supported via llama proxy.\nSession switching requires shell.py client with .aiops_session_id file.")
        elif action == "delete":
            if target_sid in sessions:
                # Get shorthand ID before deleting
                shorthand_id = get_or_create_shorthand_id(target_sid)
                del sessions[target_sid]
                remove_shorthand_mapping(target_sid)
                await push_tok(client_id, f"Deleted session ID [{shorthand_id}]: {target_sid}")
            else:
                await push_tok(client_id, f"Session not found: {target_sid}")
        else:
            await push_tok(client_id, f"Unknown action: {action}\nUse: !session <ID> attach|delete")
    else:
        await push_tok(client_id, "Usage: !session | !session <ID> attach | !session <ID> delete")

    await conditional_push_done(client_id)

async def cmd_llm_call(client_id: str, args: str):
    """
    Manage tool_call_available flag for LLM models.

    !llm_call                        - list all models with tool_call_available status
    !llm_call <true|false>           - set tool_call_available for ALL enabled models
    !llm_call <model> <true|false>   - set tool_call_available for a specific model
    """
    def is_valid_bool(s: str) -> bool:
        return s in ("true", "1", "yes", "false", "0", "no")

    def parse_bool(s: str) -> bool:
        return s in ("true", "1", "yes")

    parts = args.split()

    # No args — list all models with tool_call_available status
    if not args.strip():
        lines = ["LLM tool_call_available status:"]
        for name, cfg in sorted(LLM_REGISTRY.items()):
            available = "YES" if cfg.get("tool_call_available", False) else "NO"
            timeout = cfg.get("llm_call_timeout", 60)
            desc = cfg.get("description", "")
            lines.append(f"  {name:<14} tool_call={available}  timeout={timeout}s  {desc}")
        await push_tok(client_id, "\n".join(lines))
        await conditional_push_done(client_id)
        return

    # One arg: could be <true|false> (bulk set all) or just a model name (error)
    if len(parts) == 1:
        flag = parts[0].lower()
        if is_valid_bool(flag):
            # Bulk set all enabled models
            value = parse_bool(flag)
            changed = []
            for name, cfg in LLM_REGISTRY.items():
                cfg["tool_call_available"] = value
                if save_llm_model_field(name, "tool_call_available", value):
                    changed.append(name)
            status = "enabled" if value else "disabled"
            await push_tok(client_id,
                f"tool_call_available={status} set for all models: {', '.join(changed)}\n"
                f"Changes persisted to llm-models.json.")
        else:
            await push_tok(client_id,
                f"ERROR: Unknown argument '{parts[0]}'\n"
                "Usage:\n"
                "  !llm_call                      - list all models\n"
                "  !llm_call <true|false>         - set for all models\n"
                "  !llm_call <model> <true|false> - set for one model")
        await conditional_push_done(client_id)
        return

    # Two args: <model> <true|false>
    if len(parts) == 2:
        model_name, flag = parts[0], parts[1].lower()

        if model_name not in LLM_REGISTRY:
            available = ", ".join(sorted(LLM_REGISTRY.keys()))
            await push_tok(client_id,
                f"ERROR: Unknown model '{model_name}'\n"
                f"Available: {available}")
            await conditional_push_done(client_id)
            return

        if not is_valid_bool(flag):
            await push_tok(client_id,
                f"ERROR: Invalid value '{flag}'. Must be one of: true, false, 1, 0, yes, no")
            await conditional_push_done(client_id)
            return

        value = parse_bool(flag)
        LLM_REGISTRY[model_name]["tool_call_available"] = value
        if save_llm_model_field(model_name, "tool_call_available", value):
            status = "enabled" if value else "disabled"
            await push_tok(client_id,
                f"tool_call_available={status} for '{model_name}'. Persisted to llm-models.json.")
        else:
            await push_tok(client_id,
                f"WARNING: Updated in-memory but FAILED to persist to llm-models.json. "
                f"Change will be lost on restart.")
        await conditional_push_done(client_id)
        return

    # Too many args
    await push_tok(client_id,
        "ERROR: Too many arguments\n"
        "Usage:\n"
        "  !llm_call                      - list all models\n"
        "  !llm_call <true|false>         - set tool_call_available for all models\n"
        "  !llm_call <model> <true|false> - set for one model")
    await conditional_push_done(client_id)


async def cmd_llm_timeout(client_id: str, args: str):
    """
    Get or set llm_call_timeout for LLM delegation (llm_clean_text / llm_clean_tool).

    !llm_timeout                   - list current timeouts for all models
    !llm_timeout <seconds>         - set timeout for ALL models
    !llm_timeout <model> <seconds> - set timeout for one model
    """
    parts = args.split()

    # No args — list current timeouts
    if not args.strip():
        lines = ["LLM delegation timeouts:"]
        for name, cfg in sorted(LLM_REGISTRY.items()):
            t = cfg.get("llm_call_timeout", 60)
            available = "YES" if cfg.get("tool_call_available", False) else "NO"
            lines.append(f"  {name:<14} timeout={t}s  tool_call={available}")
        await push_tok(client_id, "\n".join(lines))
        await conditional_push_done(client_id)
        return

    # One arg: <seconds> — set all models
    if len(parts) == 1:
        try:
            secs = int(parts[0])
            if secs < 1:
                raise ValueError
        except ValueError:
            await push_tok(client_id,
                f"ERROR: Invalid value '{parts[0]}' — must be a positive integer (seconds)\n"
                "Usage: !llm_timeout [model] <seconds>")
            await conditional_push_done(client_id)
            return

        changed = []
        for name, cfg in LLM_REGISTRY.items():
            cfg["llm_call_timeout"] = secs
            if save_llm_model_field(name, "llm_call_timeout", secs):
                changed.append(name)
        await push_tok(client_id,
            f"llm_call_timeout={secs}s set for all models: {', '.join(changed)}\n"
            "Changes persisted to llm-models.json.")
        await conditional_push_done(client_id)
        return

    # Two args: <model> <seconds>
    if len(parts) == 2:
        model_name, secs_str = parts[0], parts[1]

        if model_name not in LLM_REGISTRY:
            available = ", ".join(sorted(LLM_REGISTRY.keys()))
            await push_tok(client_id,
                f"ERROR: Unknown model '{model_name}'\n"
                f"Available: {available}")
            await conditional_push_done(client_id)
            return

        try:
            secs = int(secs_str)
            if secs < 1:
                raise ValueError
        except ValueError:
            await push_tok(client_id,
                f"ERROR: Invalid value '{secs_str}' — must be a positive integer (seconds)")
            await conditional_push_done(client_id)
            return

        LLM_REGISTRY[model_name]["llm_call_timeout"] = secs
        if save_llm_model_field(model_name, "llm_call_timeout", secs):
            await push_tok(client_id,
                f"llm_call_timeout={secs}s for '{model_name}'. Persisted to llm-models.json.")
        else:
            await push_tok(client_id,
                f"WARNING: Updated in-memory but FAILED to persist to llm-models.json.")
        await conditional_push_done(client_id)
        return

    await push_tok(client_id,
        "ERROR: Too many arguments\n"
        "Usage:\n"
        "  !llm_timeout                   - list current timeouts\n"
        "  !llm_timeout <seconds>         - set for all models\n"
        "  !llm_timeout <model> <seconds> - set for one model")
    await conditional_push_done(client_id)


async def cmd_tool_preview_length(client_id: str, arg: str, session: dict):
    """
    Get or set the tool result preview length for this session.
    Controls how many characters of tool output are displayed in the chat.
    The full result is always sent to the LLM regardless of this setting.

    !tool_preview_length       - show current setting
    !tool_preview_length <n>   - set to n characters (0 = unlimited)
    """
    arg = arg.strip()
    if not arg:
        current = session.get("tool_preview_length", 500)
        limit_str = f"{current} chars" if current > 0 else "unlimited"
        await push_tok(client_id, f"Tool preview length: {limit_str}")
        await conditional_push_done(client_id)
        return

    if arg == "0" or arg.lower() in ("off", "unlimited"):
        session["tool_preview_length"] = 0
        await push_tok(client_id, "Tool preview length: unlimited (no truncation)")
    else:
        try:
            n = int(arg)
            if n < 1:
                await push_tok(client_id,
                    "ERROR: Value must be >= 1, or 0 for unlimited\n"
                    "Usage: !tool_preview_length <n>")
                await conditional_push_done(client_id)
                return
            session["tool_preview_length"] = n
            await push_tok(client_id, f"Tool preview length set to {n} chars.")
        except ValueError:
            await push_tok(client_id,
                f"ERROR: Invalid value '{arg}'\n"
                "Usage: !tool_preview_length [n]  (0 = unlimited)")
    await conditional_push_done(client_id)


async def cmd_stream(client_id: str, arg: str, session: dict):
    """
    Get or set the agent_call streaming mode for this session.

    When streaming is enabled (default), remote agent tokens are relayed via
    push_tok in real-time so Slack and other clients see per-turn progress.
    When disabled, agent_call blocks silently and returns only the final result.

    !stream               - show current setting
    !stream <true|false>  - enable or disable streaming
    """
    arg = arg.strip().lower()
    if not arg:
        current = session.get("agent_call_stream", True)
        status = "enabled" if current else "disabled"
        await push_tok(client_id, f"agent_call streaming: {status}")
        await conditional_push_done(client_id)
        return

    if arg in ("true", "1", "yes", "on"):
        session["agent_call_stream"] = True
        await push_tok(client_id, "agent_call streaming enabled — remote tokens relayed in real-time.")
    elif arg in ("false", "0", "no", "off"):
        session["agent_call_stream"] = False
        await push_tok(client_id, "agent_call streaming disabled — only final result returned.")
    else:
        await push_tok(client_id,
            f"ERROR: Invalid value '{arg}'\n"
            "Usage: !stream <true|false>")
    await conditional_push_done(client_id)


async def process_request(client_id: str, text: str, raw_payload: dict, peer_ip: str = None):
    from state import get_or_create_shorthand_id

    if client_id not in sessions:
        sessions[client_id] = {"model": raw_payload.get("default_model", DEFAULT_MODEL), "history": []}
        # Assign shorthand ID when session is created
        get_or_create_shorthand_id(client_id)
    # Store/update peer IP whenever we have it
    if peer_ip:
        sessions[client_id]["peer_ip"] = peer_ip
    session = sessions[client_id]
    stripped = text.strip()

    # Check if this is a multi-line message with multiple commands
    lines = stripped.split('\n')
    command_lines = [line.strip() for line in lines if line.strip().startswith('!')]

    # If we have multiple command lines, process them sequentially
    if len(command_lines) > 1:
        # Enable batch mode to suppress push_done in individual commands
        _batch_mode[client_id] = True
        try:
            for cmd_line in command_lines:
                parts = cmd_line[1:].split(maxsplit=1)
                cmd, arg = parts[0].lower(), parts[1].strip() if len(parts) > 1 else ""

                # Route each command
                if cmd == "help":
                    await cmd_help(client_id)
                elif cmd == "reset":
                    await cmd_reset(client_id, session)
                elif cmd == "db":
                    await cmd_db(client_id, arg)
                elif cmd == "autoaidb":
                    await cmd_autoaidb(client_id, arg)
                elif cmd == "autogate":
                    await cmd_autogate(client_id, arg)
                elif cmd == "sysprompt":
                    await cmd_sysprompt(client_id, arg)
                elif cmd == "read_system_prompt":
                    await cmd_read_system_prompt(client_id, arg)
                elif cmd == "autoaisysprompt":
                    await cmd_autoaisysprompt(client_id, arg)
                elif cmd == "model":
                    if arg:
                        await cmd_set_model(client_id, arg, session)
                    else:
                        await cmd_list_models(client_id, session["model"])
                elif cmd == "session":
                    await cmd_session(client_id, arg)
                elif cmd == "llm_call":
                    await cmd_llm_call(client_id, arg)
                elif cmd == "llm_timeout":
                    await cmd_llm_timeout(client_id, arg)
                elif cmd == "tool_preview_length":
                    await cmd_tool_preview_length(client_id, arg, session)
                elif cmd == "stream":
                    await cmd_stream(client_id, arg, session)
                else:
                    await push_tok(client_id, f"Unknown command: !{cmd}\nUse !help to see available commands.\n")

                # Add newline between command outputs for readability
                await push_tok(client_id, "\n")
        finally:
            # Disable batch mode
            _batch_mode[client_id] = False

        # After processing all commands, check if there's non-command text to send to LLM
        non_command_text = '\n'.join([line for line in lines if not line.strip().startswith('!')]).strip()
        if non_command_text:
            # Process the remaining text as a normal message to the LLM
            stripped = non_command_text
        else:
            # Only commands, no LLM interaction needed
            await conditional_push_done(client_id)
            return

    # Single command handling (original logic)
    elif stripped.startswith("!"):
        parts = stripped[1:].split(maxsplit=1)
        cmd, arg = parts[0].lower(), parts[1].strip() if len(parts) > 1 else ""

        # Command routing with validation
        if cmd == "help":
            await cmd_help(client_id)
            return
        if cmd == "reset":
            await cmd_reset(client_id, session)
            return
        if cmd == "db":
            await cmd_db(client_id, arg)
            return
        if cmd == "autoaidb":
            await cmd_autoaidb(client_id, arg)
            return
        if cmd == "autogate":
            await cmd_autogate(client_id, arg)
            return
        if cmd == "sysprompt":
            await cmd_sysprompt(client_id, arg)
            return
        if cmd == "read_system_prompt":
            await cmd_read_system_prompt(client_id, arg)
            return
        if cmd == "autoaisysprompt":
            await cmd_autoaisysprompt(client_id, arg)
            return
        if cmd == "model":
            if arg:
                await cmd_set_model(client_id, arg, session)
            else:
                await cmd_list_models(client_id, session["model"])
            return
        if cmd == "session":
            await cmd_session(client_id, arg)
            return
        if cmd == "llm_call":
            await cmd_llm_call(client_id, arg)
            return
        if cmd == "llm_timeout":
            await cmd_llm_timeout(client_id, arg)
            return
        if cmd == "tool_preview_length":
            await cmd_tool_preview_length(client_id, arg, session)
            return
        if cmd == "stream":
            await cmd_stream(client_id, arg, session)
            return

        # Catch-all for unknown commands - don't pass to LLM
        await push_tok(client_id, f"Unknown command: !{cmd}\nUse !help to see available commands.")
        await conditional_push_done(client_id)
        return

    # @<model> per-turn model switch
    # Syntax: @ModelName <prompt text>
    # - If @model is the current model: strip prefix, continue as normal
    # - If @model is unknown: return error, don't dispatch
    # - Otherwise: temporarily switch to named model for this turn, restore after
    temp_model = None
    if stripped.startswith("@"):
        first_space = stripped.find(" ")
        if first_space > 1:
            model_token = stripped[1:first_space]  # strip leading @
            rest = stripped[first_space:].strip()
        else:
            model_token = stripped[1:]
            rest = ""
        if model_token in LLM_REGISTRY:
            if model_token == session["model"]:
                # Same model — strip prefix but still grant gate bypass for this turn
                session["_temp_model_active"] = True
                temp_model = session["model"]  # set so finally block clears the flag
                stripped = rest
            else:
                # Different model — temp switch for this turn
                temp_model = session["model"]
                session["model"] = model_token
                session["_temp_model_active"] = True
                stripped = rest
        else:
            available = ", ".join(LLM_REGISTRY.keys())
            await push_tok(client_id, f"ERROR: Unknown model '@{model_token}'\nAvailable models: {available}")
            await conditional_push_done(client_id)
            return

    max_ctx = LLM_REGISTRY.get(session["model"], {}).get("max_context", 50)
    session["history"].append({"role": "user", "content": stripped})
    session["history"] = session["history"][-max_ctx:]

    try:
        final = await dispatch_llm(session["model"], session["history"], client_id)
        if final: session["history"].append({"role": "assistant", "content": final})
    except asyncio.CancelledError:
        # Remove the dangling user message so history stays consistent
        if session["history"] and session["history"][-1]["role"] == "user":
            session["history"].pop()
        raise
    finally:
        if temp_model is not None:
            session["model"] = temp_model
            session.pop("_temp_model_active", None)
        active_tasks.pop(client_id, None)

async def endpoint_submit(request: Request) -> JSONResponse:
    try: payload = await request.json()
    except: return JSONResponse({"status": "error"}, 400)

    client_id, text = payload.get("client_id"), payload.get("text", "")
    if not client_id or not text: return JSONResponse({"error": "Missing fields"}, 400)

    peer_ip = request.client.host if request.client else None
    await cancel_active_task(client_id)
    task = asyncio.create_task(process_request(client_id, text, payload, peer_ip=peer_ip))
    active_tasks[client_id] = task
    return JSONResponse({"status": "OK"})

async def endpoint_stream(request: Request):
    client_id = request.query_params.get("client_id")
    if not client_id: return JSONResponse({"error": "Missing client_id"}, 400)

    # Register session on connect so it shows up in !session before first message
    from state import get_or_create_shorthand_id
    if client_id not in sessions:
        sessions[client_id] = {"model": DEFAULT_MODEL, "history": []}
        get_or_create_shorthand_id(client_id)
    peer_ip = request.client.host if request.client else None
    if peer_ip:
        sessions[client_id]["peer_ip"] = peer_ip

    q = await get_queue(client_id)

    async def generator() -> AsyncGenerator[dict, None]:
        while True:
            if await request.is_disconnected(): break
            try:
                item = await asyncio.wait_for(q.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield {"comment": "keepalive"}
                continue
            
            t = item.get("t")
            if t == "tok": yield {"data": item["d"]}
            elif t == "done": yield {"event": "done", "data": ""}
            elif t == "err": yield {"event": "error", "data": json.dumps({"error": item["d"]})}
            elif t == "gate": yield {"event": "gate_request", "data": json.dumps(item["d"])}

    return EventSourceResponse(generator())

async def endpoint_gate_response(request: Request) -> JSONResponse:
    try: payload = await request.json()
    except: return JSONResponse({"error": "json"}, 400)
    
    gate_id, decision = payload.get("gate_id"), payload.get("decision")
    if gate_id in pending_gates:
        pending_gates[gate_id]["decision"] = decision
        pending_gates[gate_id]["event"].set()
        return JSONResponse({"status": "OK"})
    return JSONResponse({"error": "unknown gate"}, 404)

async def endpoint_health(request: Request) -> JSONResponse:
    return JSONResponse({
        "status": "healthy",
        "models": list(LLM_REGISTRY.keys()),
        "autoAIdb": auto_aidb_state,
        "autogate": tool_gate_state,
        "autoAISysPrompt": tool_gate_state.get("update_system_prompt", False),
    })

async def endpoint_list_sessions(request: Request) -> JSONResponse:
    """List all active sessions with metadata."""
    from state import sessions, sse_queues, get_or_create_shorthand_id

    client_id_filter = request.query_params.get("client_id")

    session_list = []
    for cid, data in sessions.items():
        if client_id_filter and cid != client_id_filter:
            continue
        session_list.append({
            "client_id": cid,
            "shorthand_id": get_or_create_shorthand_id(cid),
            "model": data.get("model", "unknown"),
            "history_length": len(data.get("history", [])),
            "peer_ip": data.get("peer_ip"),
        })

    return JSONResponse({"sessions": session_list})

async def endpoint_delete_session(request: Request) -> JSONResponse:
    """Delete a specific session by ID."""
    from state import sessions, sse_queues

    sid = request.path_params.get("sid")

    if sid in sessions:
        del sessions[sid]
        # Also clean up associated queue
        if sid in sse_queues:
            del sse_queues[sid]
        return JSONResponse({"status": "OK", "deleted": sid})

    return JSONResponse({"error": "session not found"}, 404)