import asyncio
import json
from typing import AsyncGenerator
from starlette.requests import Request
from starlette.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from config import log, LLM_REGISTRY, DEFAULT_MODEL, save_llm_model_field
from state import sessions, get_queue, push_tok, push_done, pending_gates, auto_aidb_state, tool_gate_state, active_tasks, cancel_active_task
from database import execute_sql
from prompt import (sp_list_files, sp_read_prompt, sp_read_file, sp_write_file,
                    sp_delete_file, sp_delete_directory, sp_copy_directory, sp_set_directory,
                    sp_resolve_model)
from agents import dispatch_llm
from tools import get_all_gate_tools, get_tool_executor

# Context flag for batch command processing
_batch_mode = {}  # client_id -> bool

async def conditional_push_done(client_id: str):
    """Only call push_done if not in batch mode"""
    if not _batch_mode.get(client_id, False):
        await push_done(client_id)

async def cmd_help(client_id: str):
    gate_tools = get_all_gate_tools()
    tool_lines = []
    for tool_name, meta in sorted(gate_tools.items(), key=lambda x: (x[1].get("type",""), x[0])):
        desc = meta.get("description", "")
        tool_lines.append(f"  {tool_name:<30} - {desc}")

    help_text = (
        "Available commands:\n"
        "  !model                                    - list available models (current marked)\n"
        "  !model <key>                              - switch active LLM\n"
        "  !reset                                    - clear conversation history\n"
        "  !help                                     - this help\n"
        "  !input_lines <n>                          - resize input area (client-side only)\n"
        "\n"
        "System Prompt:\n"
        "  !sysprompt_list <model|self>              - list system prompt files for a model\n"
        "  !sysprompt_read <model|self>              - read full assembled prompt for a model\n"
        "  !sysprompt_read <model|self> <file>       - read a specific file for a model\n"
        "  !sysprompt_write <model|self> <file> <data> - write/create a file for a model\n"
        "  !sysprompt_delete <model|self>            - delete entire prompt folder for model\n"
        "  !sysprompt_delete <model|self> <file>     - delete a specific file for model\n"
        "  !sysprompt_copy_dir <model|self> <newdir> - copy prompt folder to new directory\n"
        "  !sysprompt_set_dir <model|self> <dir>     - assign prompt folder to model\n"
        "\n"
        "Database:\n"
        "  !db_query <sql>                           - run SQL directly (no LLM)\n"
        "  !db_query_gate_read [table|*] <t|f>       - toggle read gate for DB table\n"
        "  !db_query_gate_write [table|*] <t|f>      - toggle write gate for DB table\n"
        "  !db_query_gate_status                     - show DB gate settings\n"
        "\n"
        "Search & Extract:\n"
        "  !search_ddgs <query>                      - search via DuckDuckGo\n"
        "  !search_google <query>                    - search via Google (Gemini grounding)\n"
        "  !search_tavily <query>                    - search via Tavily AI\n"
        "  !search_xai <query>                       - search via xAI Grok\n"
        "  !url_extract <url> [query]                - extract web page content\n"
        "  !search_ddgs_gate_read <t|f>              - toggle gate for search_ddgs\n"
        "  !search_google_gate_read <t|f>            - toggle gate for search_google\n"
        "  !search_tavily_gate_read <t|f>            - toggle gate for search_tavily\n"
        "  !search_xai_gate_read <t|f>               - toggle gate for search_xai\n"
        "  !url_extract_gate_read <t|f>              - toggle gate for url_extract\n"
        "\n"
        "Google Drive:\n"
        "  !google_drive <operation> [args...]       - Drive CRUD operation\n"
        "  !google_drive_gate_read <t|f>             - toggle read gate for google_drive\n"
        "  !google_drive_gate_write <t|f>            - toggle write gate for google_drive\n"
        "\n"
        "Gate Management (per-tool):\n"
        "  !sysprompt_gate_write <t|f>               - toggle write gate for sysprompt_* write ops\n"
        "  !session_gate_read <t|f>                  - toggle gate for session list\n"
        "  !session_gate_write <t|f>                 - toggle gate for session delete\n"
        "  !model_gate_write <t|f>                   - toggle gate for model set\n"
        "  !reset_gate_write <t|f>                   - toggle gate for reset\n"
        "\n"
        "Session:\n"
        "  !session                                  - list all active sessions\n"
        "  !session <ID> delete                      - delete a session from server\n"
        "  !tool_preview_length [n]                  - get/set tool result display limit (0=unlimited, default=500)\n"
        "\n"
        "Utilities:\n"
        "  !get_system_info                          - show date/time/status\n"
        "  !llm_list                                 - list LLM models\n"
        "  !llm_clean_text <model> <prompt>          - call model with clean context\n"
        "  !llm_clean_tool <model> <tool> <args>     - delegate tool call to model\n"
        "\n"
        "LLM Delegation:\n"
        "  !llm_call                                 - list models with tool_call_available status\n"
        "  !llm_call <model> <true|false>            - enable/disable delegation for a model\n"
        "  !llm_call <true|false>                    - set tool_call_available for ALL models\n"
        "  !llm_timeout <model> <seconds>            - set llm_call_timeout for a model\n"
        "  !llm_timeout <seconds>                    - set timeout for ALL models\n"
        "  !stream <true|false>                      - enable/disable agent_call streaming\n"
        "\n"
        "AI tools (gated unless noted):\n"
        + "\n".join(tool_lines) + "\n"
        + "  get_system_info()                         - auto-allowed, no gate\n"
        "  llm_clean_text(model, prompt)             - rate limited\n"
        "  llm_clean_tool(model, tool, args)         - rate limited\n"
        "  llm_list()                                - auto-allowed\n"
        "  sysprompt_list/read                       - auto-allowed\n"
        "  sysprompt_write/delete/copy_dir/set_dir   - write gate\n"
        "  session/model/reset                       - per-action gate\n"
    )
    await push_tok(client_id, help_text)
    await conditional_push_done(client_id)

async def cmd_sysprompt_list(client_id: str, model: str, session: dict):
    """List system prompt files for a model."""
    resolved = sp_resolve_model(model.strip(), session.get("model", ""))
    result = sp_list_files(resolved, LLM_REGISTRY)
    await push_tok(client_id, result)
    await conditional_push_done(client_id)


async def cmd_sysprompt_read(client_id: str, args: str, session: dict):
    """
    Read system prompt for a model (full or specific file).
    Usage: !sysprompt_read <model|self> [file]
    """
    parts = args.split(maxsplit=1)
    if not parts:
        await push_tok(client_id, "Usage: !sysprompt_read <model|self> [file]")
        await conditional_push_done(client_id)
        return
    model = parts[0].strip()
    file = parts[1].strip() if len(parts) > 1 else ""
    resolved = sp_resolve_model(model, session.get("model", ""))
    if file:
        result = sp_read_file(resolved, file, LLM_REGISTRY)
    else:
        result = sp_read_prompt(resolved, LLM_REGISTRY)
    await push_tok(client_id, result)
    await conditional_push_done(client_id)


async def cmd_sysprompt_write(client_id: str, args: str, session: dict):
    """
    Write/create a system prompt file for a model.
    Usage: !sysprompt_write <model|self> <file> <data...>
    """
    parts = args.split(maxsplit=2)
    if len(parts) < 3:
        await push_tok(client_id,
            "Usage: !sysprompt_write <model|self> <file> <data>\n"
            "  data is everything after the file argument.")
        await conditional_push_done(client_id)
        return
    model, file, data = parts[0].strip(), parts[1].strip(), parts[2]
    resolved = sp_resolve_model(model, session.get("model", ""))
    result = sp_write_file(resolved, file, data, LLM_REGISTRY)
    await push_tok(client_id, result)
    await conditional_push_done(client_id)


async def cmd_sysprompt_delete(client_id: str, args: str, session: dict):
    """
    Delete system prompt file or entire folder for a model.
    Usage: !sysprompt_delete <model|self> [file]
    """
    parts = args.split(maxsplit=1)
    if not parts:
        await push_tok(client_id, "Usage: !sysprompt_delete <model|self> [file]")
        await conditional_push_done(client_id)
        return
    model = parts[0].strip()
    file = parts[1].strip() if len(parts) > 1 else ""
    resolved = sp_resolve_model(model, session.get("model", ""))
    if file:
        result = sp_delete_file(resolved, file, LLM_REGISTRY)
    else:
        result = sp_delete_directory(resolved, LLM_REGISTRY)
    await push_tok(client_id, result)
    await conditional_push_done(client_id)


async def cmd_sysprompt_copy_dir(client_id: str, args: str, session: dict):
    """
    Copy model's prompt folder to a new directory.
    Usage: !sysprompt_copy_dir <model|self> <newdir>
    """
    parts = args.split(maxsplit=1)
    if len(parts) < 2:
        await push_tok(client_id, "Usage: !sysprompt_copy_dir <model|self> <newdir>")
        await conditional_push_done(client_id)
        return
    model, new_dir = parts[0].strip(), parts[1].strip()
    resolved = sp_resolve_model(model, session.get("model", ""))
    result = sp_copy_directory(resolved, new_dir, LLM_REGISTRY)
    await push_tok(client_id, result)
    await conditional_push_done(client_id)


async def cmd_sysprompt_set_dir(client_id: str, args: str, session: dict):
    """
    Set model's system_prompt_folder.
    Usage: !sysprompt_set_dir <model|self> <dir>
    """
    parts = args.split(maxsplit=1)
    if len(parts) < 2:
        await push_tok(client_id, "Usage: !sysprompt_set_dir <model|self> <dir>")
        await conditional_push_done(client_id)
        return
    model, dir_name = parts[0].strip(), parts[1].strip()
    resolved = sp_resolve_model(model, session.get("model", ""))
    result = sp_set_directory(resolved, dir_name)
    await push_tok(client_id, result)
    await conditional_push_done(client_id)

async def cmd_gate(client_id: str, tool_name: str, perm_type: str, flag_arg: str):
    """
    Generic gate toggle command.
    tool_name: name used in tool_gate_state or auto_aidb_state
    perm_type: "read" or "write"
    flag_arg: <table|*> <true|false>  (for db_query) or just <true|false>
    """
    def is_valid_bool(s: str) -> bool:
        return s in ("true", "1", "yes", "false", "0", "no")

    def parse_bool(s: str) -> bool:
        return s in ("true", "1", "yes")

    flag_parts = flag_arg.split()

    # db_query gate: may have table name prefix
    if tool_name == "db_query":
        if len(flag_parts) == 2:
            table, flag = flag_parts[0], flag_parts[1].lower()
        elif len(flag_parts) == 1:
            table, flag = "*", flag_parts[0].lower()
        else:
            await push_tok(client_id,
                f"Usage: !db_query_gate_{perm_type} [table|*] <true|false>\n"
                "  Omit table to set wildcard default (*) for all tables.")
            await conditional_push_done(client_id)
            return

        if not is_valid_bool(flag):
            await push_tok(client_id,
                f"ERROR: Invalid value '{flag}'. Must be true/false/1/0/yes/no")
            await conditional_push_done(client_id)
            return

        is_auto = parse_bool(flag)
        if table not in auto_aidb_state:
            auto_aidb_state[table] = {"read": False, "write": False}
        auto_aidb_state[table][perm_type] = is_auto
        if table == "*":
            # Also apply to __meta__
            if "__meta__" not in auto_aidb_state:
                auto_aidb_state["__meta__"] = {"read": False, "write": False}
            auto_aidb_state["__meta__"][perm_type] = is_auto
            label = "DEFAULT (*)"
        else:
            label = table
        status = "auto-allow (gate OFF)" if is_auto else "gated (gate ON)"
        await push_tok(client_id, f"db_query_gate_{perm_type} {label}: {status}")
        await conditional_push_done(client_id)
        return

    # status command
    if flag_arg.strip().lower() == "status":
        if tool_name == "db_query":
            lines = ["DB gate settings:"]
            for t, perms in auto_aidb_state.items():
                label = "(default)" if t == "*" else t
                r = "auto-allow" if perms.get("read") else "gated"
                w = "auto-allow" if perms.get("write") else "gated"
                lines.append(f"  {label}: read={r}, write={w}")
            await push_tok(client_id, "\n".join(lines))
        else:
            perms = tool_gate_state.get(tool_name, {})
            r = "auto-allow" if perms.get("read", False) else "gated"
            w = "auto-allow" if perms.get("write", False) else "gated"
            await push_tok(client_id, f"Gate for '{tool_name}': read={r}, write={w}")
        await conditional_push_done(client_id)
        return

    # Standard tool gate
    if len(flag_parts) != 1:
        await push_tok(client_id,
            f"Usage: !{tool_name}_gate_{perm_type} <true|false>")
        await conditional_push_done(client_id)
        return

    flag = flag_parts[0].lower()
    if not is_valid_bool(flag):
        await push_tok(client_id,
            f"ERROR: Invalid value '{flag}'. Must be true/false/1/0/yes/no")
        await conditional_push_done(client_id)
        return

    is_auto = parse_bool(flag)
    if tool_name not in tool_gate_state:
        tool_gate_state[tool_name] = {"read": False, "write": False}
    tool_gate_state[tool_name][perm_type] = is_auto
    status = "auto-allow (gate OFF)" if is_auto else "gated (gate ON)"
    await push_tok(client_id, f"{tool_name}_gate_{perm_type}: {status}")
    await conditional_push_done(client_id)


async def cmd_db_query_gate_status(client_id: str):
    """Show DB gate settings."""
    if not auto_aidb_state:
        await push_tok(client_id, "No DB gate settings configured (all tables gated by default).")
    else:
        lines = ["DB gate settings:"]
        for t, perms in auto_aidb_state.items():
            label = "(default *)" if t == "*" else ("(metadata)" if t == "__meta__" else t)
            r = "auto-allow" if perms.get("read") else "gated"
            w = "auto-allow" if perms.get("write") else "gated"
            lines.append(f"  {label}: read={r}, write={w}")
        await push_tok(client_id, "\n".join(lines))
    await conditional_push_done(client_id)


# --- New tool action commands ---

async def cmd_db_query(client_id: str, sql: str):
    """Execute SQL directly without LLM or human gate."""
    sql = sql.strip()
    if not sql:
        await push_tok(client_id,
            "Usage: !db_query <SQL>\n"
            "Examples:\n"
            "  !db_query SELECT * FROM person\n"
            "  !db_query SHOW TABLES")
        await conditional_push_done(client_id)
        return
    try:
        result = await execute_sql(sql)
        await push_tok(client_id, result)
    except Exception as exc:
        await push_tok(client_id, f"ERROR: Database query failed\n{exc}")
    await conditional_push_done(client_id)


async def cmd_search(client_id: str, engine: str, query: str):
    """Execute a search using the named engine's executor."""
    from tools import get_tool_executor
    executor = get_tool_executor(f"search_{engine}")
    if executor is None:
        await push_tok(client_id, f"ERROR: Search engine 'search_{engine}' not loaded.")
        await conditional_push_done(client_id)
        return
    if not query.strip():
        await push_tok(client_id, f"Usage: !search_{engine} <query>")
        await conditional_push_done(client_id)
        return
    try:
        result = await executor(query=query.strip())
        await push_tok(client_id, result)
    except Exception as exc:
        await push_tok(client_id, f"ERROR: search_{engine} failed: {exc}")
    await conditional_push_done(client_id)


async def cmd_url_extract(client_id: str, args: str):
    """Extract content from a URL using Tavily."""
    from tools import get_tool_executor
    parts = args.split(maxsplit=1)
    if not parts:
        await push_tok(client_id, "Usage: !url_extract <url> [query]")
        await conditional_push_done(client_id)
        return
    url = parts[0].strip()
    # Slack wraps URLs in angle brackets: <https://example.com> or <https://example.com|display>
    if url.startswith("<") and url.endswith(">"):
        url = url[1:-1].split("|")[0]
    query = parts[1].strip() if len(parts) > 1 else ""
    executor = get_tool_executor("url_extract")
    if executor is None:
        await push_tok(client_id, "ERROR: url_extract plugin not loaded.")
        await conditional_push_done(client_id)
        return
    try:
        result = await executor(method="tavily", url=url, query=query)
        await push_tok(client_id, result)
    except Exception as exc:
        await push_tok(client_id, f"ERROR: url_extract failed: {exc}")
    await conditional_push_done(client_id)


async def cmd_google_drive(client_id: str, args: str):
    """Execute a Google Drive operation."""
    parts = args.split(maxsplit=1)
    if not parts:
        await push_tok(client_id,
            "Usage: !google_drive <operation> [file_id] [file_name] [content] [folder_id]\n"
            "Operations: list, read, create, append, delete")
        await conditional_push_done(client_id)
        return
    operation = parts[0].strip()
    rest = parts[1].strip() if len(parts) > 1 else ""
    from drive import run_drive_op
    try:
        result = await run_drive_op(operation, None, rest or None, None, None)
        await push_tok(client_id, result)
    except Exception as exc:
        await push_tok(client_id, f"ERROR: google_drive failed: {exc}")
    await conditional_push_done(client_id)


async def cmd_get_system_info(client_id: str):
    """Show current date/time and system status."""
    from tools import get_system_info
    result = await get_system_info()
    await push_tok(client_id, str(result))
    await conditional_push_done(client_id)


async def cmd_llm_list(client_id: str):
    """List LLM models with their configuration."""
    from agents import llm_list
    result = await llm_list()
    await push_tok(client_id, result)
    await conditional_push_done(client_id)


async def cmd_llm_clean_text(client_id: str, args: str):
    """Call a model with clean context."""
    parts = args.split(maxsplit=1)
    if len(parts) < 2:
        await push_tok(client_id, "Usage: !llm_clean_text <model> <prompt>")
        await conditional_push_done(client_id)
        return
    model, prompt = parts[0].strip(), parts[1].strip()
    from agents import llm_clean_text
    from state import current_client_id
    token = current_client_id.set(client_id)
    try:
        result = await llm_clean_text(model=model, prompt=prompt)
        await push_tok(client_id, result)
    except Exception as exc:
        await push_tok(client_id, f"ERROR: llm_clean_text failed: {exc}")
    finally:
        current_client_id.reset(token)
    await conditional_push_done(client_id)


async def cmd_llm_clean_tool(client_id: str, args: str):
    """Delegate a tool call to a model."""
    parts = args.split(maxsplit=2)
    if len(parts) < 3:
        await push_tok(client_id, "Usage: !llm_clean_tool <model> <tool> <arguments>")
        await conditional_push_done(client_id)
        return
    model, tool, arguments = parts[0].strip(), parts[1].strip(), parts[2].strip()
    from agents import llm_clean_tool
    from state import current_client_id
    token = current_client_id.set(client_id)
    try:
        result = await llm_clean_tool(model=model, tool=tool, arguments=arguments)
        await push_tok(client_id, result)
    except Exception as exc:
        await push_tok(client_id, f"ERROR: llm_clean_tool failed: {exc}")
    finally:
        current_client_id.reset(token)
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
                elif cmd == "db_query":
                    await cmd_db_query(client_id, arg)
                elif cmd == "db_query_gate_status":
                    await cmd_db_query_gate_status(client_id)
                elif cmd in ("db_query_gate_read", "db_query_gate_write"):
                    perm = "read" if cmd == "db_query_gate_read" else "write"
                    await cmd_gate(client_id, "db_query", perm, arg)
                elif cmd == "sysprompt_list":
                    await cmd_sysprompt_list(client_id, arg, session)
                elif cmd == "sysprompt_read":
                    await cmd_sysprompt_read(client_id, arg, session)
                elif cmd == "sysprompt_write":
                    await cmd_sysprompt_write(client_id, arg, session)
                elif cmd == "sysprompt_delete":
                    await cmd_sysprompt_delete(client_id, arg, session)
                elif cmd == "sysprompt_copy_dir":
                    await cmd_sysprompt_copy_dir(client_id, arg, session)
                elif cmd == "sysprompt_set_dir":
                    await cmd_sysprompt_set_dir(client_id, arg, session)
                elif cmd in ("search_ddgs", "search_google", "search_tavily", "search_xai"):
                    engine = cmd[len("search_"):]
                    await cmd_search(client_id, engine, arg)
                elif cmd == "url_extract":
                    await cmd_url_extract(client_id, arg)
                elif cmd == "google_drive":
                    await cmd_google_drive(client_id, arg)
                elif cmd == "get_system_info":
                    await cmd_get_system_info(client_id)
                elif cmd == "llm_list":
                    await cmd_llm_list(client_id)
                elif cmd == "llm_clean_text":
                    await cmd_llm_clean_text(client_id, arg)
                elif cmd == "llm_clean_tool":
                    await cmd_llm_clean_tool(client_id, arg)
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
                elif cmd.endswith("_gate_read") or cmd.endswith("_gate_write"):
                    # Generic per-tool gate command: !<toolname>_gate_read / !<toolname>_gate_write
                    if cmd.endswith("_gate_read"):
                        tool_name = cmd[:-len("_gate_read")]
                        perm = "read"
                    else:
                        tool_name = cmd[:-len("_gate_write")]
                        perm = "write"
                    await cmd_gate(client_id, tool_name, perm, arg)
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
        if cmd == "db_query":
            await cmd_db_query(client_id, arg)
            return
        if cmd == "db_query_gate_status":
            await cmd_db_query_gate_status(client_id)
            return
        if cmd in ("db_query_gate_read", "db_query_gate_write"):
            perm = "read" if cmd == "db_query_gate_read" else "write"
            await cmd_gate(client_id, "db_query", perm, arg)
            return
        if cmd == "sysprompt_list":
            await cmd_sysprompt_list(client_id, arg, session)
            return
        if cmd == "sysprompt_read":
            await cmd_sysprompt_read(client_id, arg, session)
            return
        if cmd == "sysprompt_write":
            await cmd_sysprompt_write(client_id, arg, session)
            return
        if cmd == "sysprompt_delete":
            await cmd_sysprompt_delete(client_id, arg, session)
            return
        if cmd == "sysprompt_copy_dir":
            await cmd_sysprompt_copy_dir(client_id, arg, session)
            return
        if cmd == "sysprompt_set_dir":
            await cmd_sysprompt_set_dir(client_id, arg, session)
            return
        if cmd in ("search_ddgs", "search_google", "search_tavily", "search_xai"):
            engine = cmd[len("search_"):]
            await cmd_search(client_id, engine, arg)
            return
        if cmd == "url_extract":
            await cmd_url_extract(client_id, arg)
            return
        if cmd == "google_drive":
            await cmd_google_drive(client_id, arg)
            return
        if cmd == "get_system_info":
            await cmd_get_system_info(client_id)
            return
        if cmd == "llm_list":
            await cmd_llm_list(client_id)
            return
        if cmd == "llm_clean_text":
            await cmd_llm_clean_text(client_id, arg)
            return
        if cmd == "llm_clean_tool":
            await cmd_llm_clean_tool(client_id, arg)
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
        # Generic per-tool gate command: !<toolname>_gate_read / !<toolname>_gate_write
        if cmd.endswith("_gate_read") or cmd.endswith("_gate_write"):
            if cmd.endswith("_gate_read"):
                tool_name = cmd[:-len("_gate_read")]
                perm = "read"
            else:
                tool_name = cmd[:-len("_gate_write")]
                perm = "write"
            await cmd_gate(client_id, tool_name, perm, arg)
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