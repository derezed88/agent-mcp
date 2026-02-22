import asyncio
import json
import os
import re
import uuid

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
)

#from .config import log, MAX_TOOL_ITERATIONS, LLM_REGISTRY
#from .state import push_tok, push_done, push_err
#from .prompt import get_current_prompt
#from .gate import check_human_gate
#from .database import execute_sql
#from .tools import (
#    db_query, update_system_prompt, get_system_info, 
#    google_search, google_drive,
#    OPENAI_TOOL_DEFS, GEMINI_TOOL
#)

import time

from config import log, MAX_TOOL_ITERATIONS, LLM_REGISTRY, RATE_LIMITS, LIVE_LIMITS, save_llm_model_field
from state import push_tok, push_done, push_err, current_client_id, sessions
from prompt import get_current_prompt, load_prompt_for_folder
from gate import check_human_gate
from database import execute_sql
from tools import (
    get_system_info,
    get_all_lc_tools, get_all_openai_tools, get_tool_executor,
    get_tool_type,
)

# ---------------------------------------------------------------------------
# Outbound agent message filters
# Loaded once at startup from plugins-enabled.json
# plugin_config.plugin_client_api.OUTBOUND_AGENT_ALLOWED_COMMANDS
# plugin_config.plugin_client_api.OUTBOUND_AGENT_BLOCKED_COMMANDS
#
# These filter the *message* text sent outbound via agent_call — not tool names.
# ALLOWED (non-empty): message must start with one of the listed prefixes.
#          Empty [] = all messages permitted (no check performed).
# BLOCKED: message must not start with any of the listed prefixes.
#          Always checked when non-empty; empty [] = nothing blocked.
# Both lists are lowercased prefix strings.
# Default is empty for both — all agent-to-agent messages are permitted.
# ---------------------------------------------------------------------------

_outbound_agent_allowed: list[str] = []
_outbound_agent_blocked: list[str] = []


def _load_outbound_agent_filters() -> None:
    """Load OUTBOUND_AGENT_ALLOWED/BLOCKED_COMMANDS from plugins-enabled.json."""
    global _outbound_agent_allowed, _outbound_agent_blocked
    try:
        path = os.path.join(os.path.dirname(__file__), "plugins-enabled.json")
        with open(path, "r") as f:
            cfg = json.load(f)
        api_cfg = cfg.get("plugin_config", {}).get("plugin_client_api", {})
        raw_allow = api_cfg.get("OUTBOUND_AGENT_ALLOWED_COMMANDS", [])
        raw_block = api_cfg.get("OUTBOUND_AGENT_BLOCKED_COMMANDS", [])
        _outbound_agent_allowed = [s.strip().lower() for s in raw_allow if s.strip()]
        _outbound_agent_blocked = [s.strip().lower() for s in raw_block if s.strip()]
    except Exception:
        pass   # no filter file = no restrictions


_load_outbound_agent_filters()


def _match_outbound_pattern(msg_lower: str, pattern: str) -> bool:
    """
    Match a lowercased, stripped message against a filter pattern.

    Special commands have multiple forms:
      !reset                       (no args)
      !model / !model <name>        (optional args after space)
      !tmux new foo / !tmux ls     (subcommand + optional args)

    Rules:
    - If pattern ends with a space: raw prefix match (caller controls boundary).
    - Otherwise: match if msg == pattern OR msg starts with pattern + " ".
      This prevents "!mod" from matching "!model" while still matching
      "!model", "!model <name>", and "!model list".
    - Non-! patterns (plain text prefixes) use the same boundary logic.
    """
    if pattern.endswith(" "):
        return msg_lower.startswith(pattern)
    return msg_lower == pattern or msg_lower.startswith(pattern + " ")


def _check_outbound_agent_message(message: str) -> str | None:
    """
    Apply outbound agent message filters.
    Returns None if permitted, or an error string if blocked.

    Patterns are lowercased at load time. Matching is word-boundary aware:
    pattern '!model' matches '!model' and '!model <name>' but NOT '!modelx'.
    To match any prefix including mid-word, end the pattern with a space.
    """
    msg_lower = message.strip().lower()
    if _outbound_agent_allowed:
        if not any(_match_outbound_pattern(msg_lower, p) for p in _outbound_agent_allowed):
            return (
                f"BLOCKED by OUTBOUND_AGENT_ALLOWED_COMMANDS: message does not match "
                f"any allowed prefix. Allowed: {', '.join(_outbound_agent_allowed)}"
            )
    for pattern in _outbound_agent_blocked:
        if _match_outbound_pattern(msg_lower, pattern):
            return (
                f"BLOCKED by OUTBOUND_AGENT_BLOCKED_COMMANDS: message matches "
                f"blocked pattern '{pattern}'."
            )
    return None


# ---------------------------------------------------------------------------
# LangChain LLM Factory
# ---------------------------------------------------------------------------

def _build_lc_llm(model_key: str):
    """
    Build a LangChain chat model from LLM_REGISTRY config.

    Returns a ChatOpenAI or ChatGoogleGenerativeAI instance.
    Both expose the same .ainvoke() / .astream() interface.
    """
    cfg = LLM_REGISTRY[model_key]
    if cfg["type"] == "OPENAI":
        return ChatOpenAI(
            model=cfg["model_id"],
            base_url=cfg.get("host"),
            api_key=cfg.get("key") or "no-key-required",
            streaming=True,
            timeout=cfg.get("llm_call_timeout", 60),
        )
    if cfg["type"] == "GEMINI":
        return ChatGoogleGenerativeAI(
            model=cfg["model_id"],
            google_api_key=cfg.get("key"),
            request_timeout=cfg.get("llm_call_timeout", 60),
        )
    raise ValueError(f"Unsupported model type '{cfg['type']}' for model '{model_key}'")


def _content_to_str(content) -> str:
    """
    Normalise AIMessage.content to a plain string.

    LangChain models can return content as:
      - str  (OpenAI, most models)
      - list of dicts (Gemini multimodal, Anthropic content blocks)
        e.g. [{'type': 'text', 'text': '...'}, {'type': 'tool_use', ...}]

    Extracts all 'text' blocks and joins them. Returns "" for non-text only.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    return ""


def _to_lc_messages(system_prompt: str, messages: list[dict]) -> list[BaseMessage]:
    """
    Convert the internal message format (list of role/content dicts) to
    LangChain BaseMessage objects.

    Internal format:  [{"role": "user"|"assistant"|"system"|"tool", "content": "..."}]
    Tool messages also carry "tool_call_id" and optionally "name".
    """
    lc_msgs: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content") or ""
        if role == "system":
            lc_msgs.append(SystemMessage(content=content))
        elif role == "user":
            lc_msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            # Preserve tool_calls if present (for history replay)
            tool_calls = m.get("tool_calls")
            if tool_calls:
                lc_msgs.append(AIMessage(content=content, tool_calls=[
                    {"id": tc["id"], "name": tc["function"]["name"],
                     "args": json.loads(tc["function"]["arguments"])}
                    for tc in tool_calls
                ]))
            else:
                lc_msgs.append(AIMessage(content=content))
        elif role == "tool":
            lc_msgs.append(ToolMessage(
                content=content,
                tool_call_id=m.get("tool_call_id", ""),
            ))
    return lc_msgs


# ---------------------------------------------------------------------------
# Global variables for tool definitions (updated dynamically)
# ---------------------------------------------------------------------------

_CURRENT_LC_TOOLS: list = []   # StructuredTool objects — passed to bind_tools()
_CURRENT_OPENAI_TOOLS: list = []  # OpenAI dicts — used by try_force_tool_calls()


def update_tool_definitions():
    """
    Populate tool globals after all plugins are registered.

    Called once by agent-mcp.py after plugin loading completes.
    Also triggers CORE_LC_TOOLS construction (which needs agents imported).
    """
    global _CURRENT_LC_TOOLS, _CURRENT_OPENAI_TOOLS
    import tools as _tools_module
    # Build core LC tools now (agents is fully imported, no circular issue)
    _tools_module.CORE_LC_TOOLS = _tools_module._make_core_lc_tools()
    _CURRENT_LC_TOOLS = get_all_lc_tools()
    _CURRENT_OPENAI_TOOLS = get_all_openai_tools()


# --- Universal Rate Limiter ---

# Sliding-window call timestamps: key = "client_id:tool_type" -> [timestamps]
_rate_timestamps: dict[str, list[float]] = {}


async def check_rate_limit(client_id: str, tool_name: str, tool_type: str) -> tuple[bool, str]:
    """
    Check whether a tool call is within its rate limit.

    Returns (allowed: bool, error_msg: str).
    error_msg is empty when allowed=True.

    When auto_disable=True and the limit is breached for an llm_call tool,
    tool_call_available is set to False in the live registry (not persisted —
    user must re-enable with !llm_call <model> true).
    """
    cfg = RATE_LIMITS.get(tool_type, {})
    max_calls = cfg.get("calls", 0)
    window = cfg.get("window_seconds", 0)

    if max_calls == 0 or window == 0:
        return True, ""  # unlimited

    key = f"{client_id}:{tool_type}"
    now = time.monotonic()

    # Prune timestamps outside the window
    timestamps = _rate_timestamps.get(key, [])
    timestamps = [t for t in timestamps if now - t < window]

    if len(timestamps) >= max_calls:
        auto_disable = cfg.get("auto_disable", False)
        auto_disable_msg = ""

        if auto_disable and tool_type == "llm_call":
            # Extract model name from tool_args stored in caller context
            # We find it by scanning the tool name — llm_call_clean carries model in args,
            # but at this level we only have tool_type. Auto-disable all tool_call_available
            # models that are currently enabled to be safe.
            for model_name, model_cfg in LLM_REGISTRY.items():
                if model_cfg.get("tool_call_available", False):
                    model_cfg["tool_call_available"] = False
                    log.warning(f"Rate limit auto-disabled tool_call_available for model '{model_name}'")
            auto_disable_msg = " All llm_call models have been auto-disabled. Use !llm_call <model> true to re-enable."

        error_msg = (
            f"RATE LIMIT EXCEEDED: {tool_name} ({tool_type}) — "
            f"limit is {max_calls} calls in {window}s for this session."
            f"{auto_disable_msg}"
        )
        log.warning(f"Rate limit exceeded: client={client_id} tool={tool_name} type={tool_type}")
        return False, error_msg

    timestamps.append(now)
    _rate_timestamps[key] = timestamps
    return True, ""


# --- Tool Execution ---

async def execute_tool(client_id: str, tool_name: str, tool_args: dict) -> str:
    # Set context var so executors can read client_id without it being a parameter
    current_client_id.set(client_id)

    # Universal rate limit check (before gate — no point gating a rate-limited call)
    tool_type = get_tool_type(tool_name)
    rate_ok, rate_err = await check_rate_limit(client_id, tool_name, tool_type)
    if not rate_ok:
        await push_tok(client_id, f"\n[RATE LIMITED] {tool_name}: {rate_err}\n")
        return rate_err

    allowed = await check_human_gate(client_id, tool_name, tool_args)
    if not allowed:
        # Provide clear, actionable feedback to the LLM about the rejection
        rejection_msg = (
            f"TOOL CALL REJECTED: The user rejected your {tool_name} request.\n\n"
            f"IMPORTANT: Do NOT retry this tool call. Instead:\n"
            f"1. Acknowledge the rejection to the user\n"
            f"2. Explain what information you were trying to access\n"
            f"3. Ask the user if they want to:\n"
            f"   - Approve this specific request, or\n"
            f"   - Provide the information manually, or\n"
            f"   - Skip this step entirely\n\n"
            f"Do NOT make the same tool call again without explicit user approval."
        )
        # Also show user-facing message about rejection
        await push_tok(client_id, f"\n[REJECTED] {tool_name} call was denied by user.\n")
        return rejection_msg

    # Get executor function dynamically
    executor = get_tool_executor(tool_name)
    if not executor:
        return f"Unknown tool: {tool_name}"

    # Tool-specific logging and execution
    try:
        # Display tool call info (suppressed if tool_suppress=True)
        _tool_suppress = sessions.get(client_id, {}).get("tool_suppress", False)
        if not _tool_suppress:
            if tool_name == "db_query":
                sql = tool_args.get("sql", "")
                await push_tok(client_id, f"\n[db ▶] {sql}\n")
            elif tool_name in ("sysprompt_write", "sysprompt_delete", "sysprompt_copy_dir", "sysprompt_set_dir"):
                await push_tok(client_id, f"\n[sysprompt ▶] {tool_name}…\n")
            elif tool_name in ("sysprompt_list", "sysprompt_read"):
                await push_tok(client_id, "\n[sysprompt ▶] reading…\n")
            elif tool_name == "get_system_info":
                await push_tok(client_id, "\n[sysinfo ▶] fetching…\n")
            elif tool_name == "google_search":
                query = tool_args.get("query", "")
                await push_tok(client_id, f"\n[search google ▶] {query}\n")
            elif tool_name == "ddgs_search":
                query = tool_args.get("query", "")
                await push_tok(client_id, f"\n[search ddgs ▶] {query}\n")
            elif tool_name == "tavily_search":
                query = tool_args.get("query", "")
                await push_tok(client_id, f"\n[search tavily ▶] {query}\n")
            elif tool_name == "google_drive":
                op = tool_args.get("operation", "?")
                await push_tok(client_id, f"\n[drive ▶] {op}\n")
            else:
                await push_tok(client_id, f"\n[{tool_name} ▶] executing…\n")

        # Execute the tool
        result = await executor(**tool_args)

        # Display result with preview (length controlled by per-session tool_preview_length)
        # -1 = unlimited (no truncation), 0 = tags printed but no content, >0 = truncate to N chars
        sess = sessions.get(client_id, {})
        tool_suppress = sess.get("tool_suppress", False)
        preview_len = sess.get("tool_preview_length", 500)
        result_str = str(result)

        if not tool_suppress:
            if preview_len == 0:
                preview = ""
            elif preview_len == -1 or len(result_str) <= preview_len:
                preview = result_str
            else:
                preview = result_str[:preview_len] + "\n…(truncated)"

            if tool_name == "db_query":
                if preview:
                    await push_tok(client_id, f"[db ◀]\n{preview}\n")
                else:
                    await push_tok(client_id, "[db ◀]\n")
            elif tool_name in ("sysprompt_write", "sysprompt_delete", "sysprompt_copy_dir", "sysprompt_set_dir"):
                await push_tok(client_id, f"[sysprompt ◀] {result}\n")
            elif tool_name in ("sysprompt_list", "sysprompt_read"):
                if preview:
                    await push_tok(client_id, f"[sysprompt ◀]\n{preview}\n")
                else:
                    await push_tok(client_id, "[sysprompt ◀]\n")
            elif tool_name == "get_system_info":
                await push_tok(client_id, f"[sysinfo ◀] {result}\n")
                return json.dumps(result) if isinstance(result, dict) else str(result)
            elif tool_name in ("google_search", "ddgs_search", "tavily_search"):
                label = {"google_search": "google", "ddgs_search": "ddgs", "tavily_search": "tavily"}[tool_name]
                if preview:
                    await push_tok(client_id, f"[search {label} ◀]\n{preview}\n")
                else:
                    await push_tok(client_id, f"[search {label} ◀]\n")
            elif tool_name == "google_drive":
                if preview:
                    await push_tok(client_id, f"[drive ◀]\n{preview}\n")
                else:
                    await push_tok(client_id, "[drive ◀]\n")
            else:
                if preview:
                    await push_tok(client_id, f"[{tool_name} ◀]\n{preview}\n")
                else:
                    await push_tok(client_id, f"[{tool_name} ◀]\n")
        elif tool_name == "get_system_info":
            # get_system_info needs the return even when suppressed
            return json.dumps(result) if isinstance(result, dict) else str(result)

        return str(result)

    except Exception as exc:
        error_msg = f"{tool_name} error: {exc}"
        await push_tok(client_id, f"[{tool_name} error] {exc}\n")
        return error_msg

# --- General tool call extraction ---

_SQL_KEYWORDS = re.compile(r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|SHOW|DESCRIBE|TRUNCATE|REPLACE)\b", re.IGNORECASE | re.MULTILINE)

# Three-level nested brace matching — covers {args: {key: {val}}} which handles all real tool call shapes.
_JSON_BLOB_RE = re.compile(r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}', re.DOTALL)


def _try_parse_json_tool(raw: str) -> tuple[str, dict] | None:
    """Try to parse a JSON blob as a tool call {name, arguments/parameters}.
    Tries raw first, then with {{ }} normalization (llama.cpp template artifact)."""
    for candidate in (raw, raw.replace("{{", "{").replace("}}", "}")):
        try:
            payload = json.loads(candidate)
            if not isinstance(payload, dict):
                continue
            name = payload.get("name", "")
            if not name or not isinstance(name, str):
                continue
            args = payload.get("arguments", payload.get("parameters", payload.get("input", {})))
            return (name, args if isinstance(args, dict) else {})
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def try_force_tool_calls(text: str) -> list[tuple[str, dict, str]]:
    """Extract all tool calls from model text output.

    Handles any format a model might use:
      - <tool_call>{...}</tool_call>  (Qwen, Mistral, Hermes, etc.)
      - [TOOL_CALL]{...}              (some fine-tunes)
      - ```json\\n{...}\\n```          (markdown code block)
      - bare {"name": ..., "arguments": ...} JSON anywhere in text
      - {{ }} brace-escaping (llama.cpp template artifact)

    Tool names are validated against the live tool registry so plugin tools
    work automatically without needing model-specific regex patterns.
    Falls back to SQL keyword heuristic as a last resort.
    """
    from tools import get_all_openai_tools
    valid_tools = {t["function"]["name"] for t in get_all_openai_tools()}

    results = []
    seen_calls: set[str] = set()

    for m in _JSON_BLOB_RE.finditer(text):
        parsed = _try_parse_json_tool(m.group(0))
        if parsed is None:
            continue
        name, args = parsed
        if name not in valid_tools:
            continue
        # Deduplicate by name+args fingerprint so the same tool can be called
        # multiple times with different arguments (e.g. read file1, read file2, read file3)
        fingerprint = f"{name}:{json.dumps(args, sort_keys=True)}"
        if fingerprint in seen_calls:
            continue
        seen_calls.add(fingerprint)
        results.append((name, args, f"forced-{uuid.uuid4().hex[:8]}"))

    if results:
        return results

    # Last resort: bare SQL statement with no JSON wrapper
    stripped = text.strip()
    if _SQL_KEYWORDS.match(stripped):
        first_kw = _SQL_KEYWORDS.search(stripped)
        if first_kw and first_kw.start() <= 120:
            return [("db_query", {"sql": stripped[first_kw.start():]}, f"forced-{uuid.uuid4().hex[:8]}")]

    return []

# --- Enrichment ---

_PERSON_QUERY_RE = re.compile(r"(?:\b(?:describe|tell\s+me\s+about|who\s+is|what\s+is|profile\s+of|details?\s+(?:about|on|for)|info(?:rmation)?\s+(?:about|on|for)|about)\b.{0,50}\b(?:mark|lee|person|people|user|jimenez)\b)|(?:\b(?:mark|lee|jimenez)\b.{0,50}\b(?:person|details?|info|describe|birthday|born|relation|wife|husband|age|nickname|name|who|what|profile)\b)|\b(?:my\s+(?:details?|info|profile|person)|person\s+table)\b", re.IGNORECASE | re.VERBOSE)

async def auto_enrich_context(messages: list[dict], client_id: str) -> list[dict]:
    if not messages: return messages
    last_user = next((m for m in reversed(messages) if m["role"] == "user"), None)
    if not last_user: return messages

    text = last_user.get("content", "")
    enrichments = []
    
    if _PERSON_QUERY_RE.search(text):
        sql = "SELECT * FROM person"
        try:
            result = await execute_sql(sql)
            enrichments.append(f"[auto-retrieved via: {sql}]\n{result}")
            await push_tok(client_id, f"\n[context] Auto-queried: {sql}\n")
        except Exception: pass

    if not enrichments: return messages

    inject = {
        "role": "system",
        "content": "## Auto-retrieved database context\nBase answer on this data:\n\n" + "\n\n".join(enrichments)
    }
    return list(messages[:-1]) + [inject, messages[-1]]

# --- Agent Loop ---

async def agentic_lc(model_key: str, messages: list[dict], client_id: str) -> str:
    """
    Single agentic loop for all LLM backends using LangChain.

    Replaces the former agentic_openai() + agentic_gemini() pair.
    Tool schema format (OpenAI dicts) and executor registry are unchanged —
    this is purely an LLM abstraction swap.
    """
    try:
        llm = _build_lc_llm(model_key)

        # Bind all registered tools so the model knows what it can call.
        # StructuredTool objects carry full schema + coroutine reference.
        llm_with_tools = llm.bind_tools(_CURRENT_LC_TOOLS)

        # Load per-model system prompt (stateless; falls back to global if not configured)
        model_cfg = LLM_REGISTRY.get(model_key, {})
        sp_folder_rel = model_cfg.get("system_prompt_folder", "")
        if sp_folder_rel and sp_folder_rel.lower() != "none":
            from config import BASE_DIR
            sp_folder_abs = os.path.join(BASE_DIR, sp_folder_rel)
            system_prompt = load_prompt_for_folder(sp_folder_abs)
        else:
            system_prompt = get_current_prompt()

        # Per-model timeout for ainvoke (same setting used by llm_clean_text/tool).
        # Prevents indefinite stalls when the LLM API hangs or thinks too long.
        invoke_timeout = model_cfg.get("llm_call_timeout", 120)

        # Convert internal message format to LangChain message objects
        ctx: list[BaseMessage] = _to_lc_messages(system_prompt, messages)

        _suppress = sessions.get(client_id, {}).get("tool_suppress", False)
        for _ in range(LIVE_LIMITS.get("max_tool_iterations", MAX_TOOL_ITERATIONS)):
            if not _suppress:
                await push_tok(client_id, "\n[thinking…]\n")
            try:
                ai_msg: AIMessage = await asyncio.wait_for(
                    llm_with_tools.ainvoke(ctx),
                    timeout=invoke_timeout,
                )
            except asyncio.TimeoutError:
                await push_tok(client_id, f"\n[LLM timeout after {invoke_timeout}s — aborting turn]\n")
                await push_done(client_id)
                return ""
            ctx.append(ai_msg)

            if not ai_msg.tool_calls:
                # Check for bare/XML tool calls from local models (Qwen, Hermes, etc.)
                raw_text = _content_to_str(ai_msg.content)
                forced_calls = try_force_tool_calls(raw_text)
                if forced_calls:
                    tool_results = []
                    for tool_name, tool_args, _call_id in forced_calls:
                        if not _suppress:
                            await push_tok(client_id, f"\n[catcher] Detected bare {tool_name} call…\n")
                        result = await execute_tool(client_id, tool_name, tool_args)
                        tool_results.append(f"[Tool result for {tool_name}]: {result}")
                    # Inject results as a user turn (plain text — local models don't
                    # understand the ToolMessage format)
                    ctx.append(HumanMessage(content="\n\n".join(tool_results)))
                    continue

                # No tool calls — final answer
                final = _content_to_str(ai_msg.content)
                if final:
                    await push_tok(client_id, final)
                await push_done(client_id)
                return final

            # Execute all tool calls in this turn
            has_non_agent_call_output = False
            for tc in ai_msg.tool_calls:
                is_streaming_agent_call = (
                    tc["name"] == "agent_call"
                    and tc["args"].get("stream", True)
                )
                if tc["name"] != "agent_call":
                    has_non_agent_call_output = True
                result = await execute_tool(client_id, tc["name"], tc["args"])
                ctx.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
                # Flush immediately after each streaming agent_call so Slack posts
                # per-turn progress as it arrives rather than batching all turns.
                # Without this, when the LLM issues N agent_calls in one batch
                # (some models batch all turns), a single push_done at the end causes
                # the Slack consumer to post the entire N-turn conversation as one
                # block — the batch behaviour observed in Test 6 (5-turn).
                if is_streaming_agent_call:
                    await push_done(client_id)
            # Signal end of this tool-call round trip for any non-agent_call tools
            # (db_query, google_drive, etc.) so streaming clients see intermediate
            # results before the LLM's next thinking step.
            if has_non_agent_call_output:
                await push_done(client_id)

        await push_tok(client_id, "\n[Max iterations]\n")
        await push_done(client_id)
        return ""

    except Exception as exc:
        await push_err(client_id, str(exc))
        return ""

async def llm_clean_text(model: str, prompt: str) -> str:
    """
    Call a target LLM with a single prompt and no context.

    No system prompt, no chat history, no tools are sent to the target model.
    The prompt is the only input. Returns the raw text response.

    client_id is read from the current_client_id ContextVar (set by execute_tool).
    """
    client_id = current_client_id.get("")

    cfg = LLM_REGISTRY.get(model)
    if not cfg:
        return f"ERROR: Unknown model '{model}'. Use llm_list() to see available models."

    if not cfg.get("tool_call_available", False):
        return (
            f"ERROR: Model '{model}' is not available for tool calls "
            f"(tool_call_available=false). Use !llm_call {model} true to enable it."
        )

    timeout = cfg.get("llm_call_timeout", 60)

    if not sessions.get(client_id, {}).get("tool_suppress", False):
        await push_tok(client_id, f"\n[llm_clean_text ▶] {model}: {prompt[:80]}{'…' if len(prompt) > 80 else ''}\n")

    try:
        llm = _build_lc_llm(model)
        response = await asyncio.wait_for(
            llm.ainvoke([HumanMessage(content=prompt)]),
            timeout=timeout,
        )
        result = _content_to_str(response.content)

        _sess = sessions.get(client_id, {})
        _suppress = _sess.get("tool_suppress", False)
        if not _suppress:
            preview_len = _sess.get("tool_preview_length", 500)
            if preview_len == 0:
                await push_tok(client_id, f"[llm_clean_text ◀] {model}:\n")
            elif preview_len == -1 or len(result) <= preview_len:
                await push_tok(client_id, f"[llm_clean_text ◀] {model}:\n{result}\n")
            else:
                await push_tok(client_id, f"[llm_clean_text ◀] {model}:\n{result[:preview_len]}\n…(truncated)\n")
        return result

    except asyncio.TimeoutError:
        msg = f"ERROR: llm_clean_text timed out after {timeout}s waiting for model '{model}'."
        await push_tok(client_id, f"[llm_clean_text ✗] {model}: timeout after {timeout}s\n")
        log.warning(f"llm_clean_text timeout: model={model} client={client_id}")
        return msg
    except Exception as exc:
        msg = f"ERROR: llm_clean_text failed for model '{model}': {exc}"
        await push_tok(client_id, f"[llm_clean_text ✗] {model}: {exc}\n")
        log.error(f"llm_clean_text error: model={model} client={client_id} exc={exc}")
        return msg



async def llm_clean_tool(model: str, tool: str, arguments: str) -> str:
    """
    Delegate a single tool call to a target LLM with no session context.

    The target model receives only:
    - A system prompt containing the tool's own definition (from .system_prompt_tool-<name>)
    - The arguments string as the user message

    The server executes the tool call if the model makes one (normal gates apply).
    Returns the target model's final text synthesis of the tool result.

    Uses LangChain bind_tools() so the same two-turn exchange works for both
    OpenAI-compatible and Gemini backends without separate branches.

    client_id is read from the current_client_id ContextVar (set by execute_tool).
    """
    from tools import get_tool_executor, get_section_for_tool, get_openai_tool_schema
    client_id = current_client_id.get("")

    # Validate model
    cfg = LLM_REGISTRY.get(model)
    if not cfg:
        return f"ERROR: Unknown model '{model}'. Use llm_list() to see available models."
    if not cfg.get("tool_call_available", False):
        return (
            f"ERROR: Model '{model}' has tool_call_available=false. "
            f"Use !llm_call {model} true to enable it."
        )

    # Validate tool
    executor = get_tool_executor(tool)
    if not executor:
        return f"ERROR: Unknown tool '{tool}'. Check llm_list() or !help for available tools."

    # Get the tool's section definition to use as system prompt
    tool_system_prompt = get_section_for_tool(tool)
    if not tool_system_prompt:
        tool_system_prompt = f"You have access to one tool: {tool}. Use it to answer the user's request."

    # Build a minimal StructuredTool from the existing OpenAI schema so LangChain
    # can bind it to the model — no need to rewrite tool definitions yet.
    tool_schema = get_openai_tool_schema(tool)
    if not tool_schema:
        return f"ERROR: No schema found for tool '{tool}'. Cannot delegate to target model."

    timeout = cfg.get("llm_call_timeout", 60)

    await push_tok(client_id, f"\n[llm_clean_tool ▶] {model}/{tool}: {arguments[:60]}{'…' if len(arguments) > 60 else ''}\n")

    try:
        async def _run():
            from langchain_core.tools import StructuredTool

            # Build a LangChain tool wrapper around the existing executor
            lc_tool = StructuredTool.from_function(
                coroutine=executor,
                name=tool_schema["function"]["name"],
                description=tool_schema["function"].get("description", ""),
            )

            llm = _build_lc_llm(model)
            llm_with_tool = llm.bind_tools([lc_tool])

            # Turn 1: model decides whether to call the tool
            turn1_msgs = [
                SystemMessage(content=tool_system_prompt),
                HumanMessage(content=arguments),
            ]
            ai_msg: AIMessage = await llm_with_tool.ainvoke(turn1_msgs)

            # If model replied with text directly (no tool call), return it
            if not ai_msg.tool_calls:
                return _content_to_str(ai_msg.content)

            # Execute the first tool call (single call enforcement)
            tc = ai_msg.tool_calls[0]
            tool_result = await execute_tool(client_id, tc["name"], tc["args"])

            # Turn 2: send result back for synthesis
            turn2_msgs = turn1_msgs + [
                ai_msg,
                ToolMessage(content=str(tool_result), tool_call_id=tc["id"]),
            ]
            final_msg: AIMessage = await llm.ainvoke(turn2_msgs)
            return _content_to_str(final_msg.content) or str(tool_result)

        result = await asyncio.wait_for(_run(), timeout=timeout)

        preview_len = sessions.get(client_id, {}).get("tool_preview_length", 500)
        preview = result if (preview_len == 0 or len(result) <= preview_len) else result[:preview_len] + "\n…(truncated)"
        await push_tok(client_id, f"[llm_clean_tool ◀] {model}/{tool}:\n{preview}\n")
        return result

    except asyncio.TimeoutError:
        msg = f"ERROR: llm_clean_tool timed out after {timeout}s for model '{model}'."
        await push_tok(client_id, f"[llm_clean_tool ✗] {model}/{tool}: timeout after {timeout}s\n")
        return msg
    except Exception as exc:
        msg = f"ERROR: llm_clean_tool failed for model '{model}', tool '{tool}': {exc}"
        await push_tok(client_id, f"[llm_clean_tool ✗] {model}/{tool}: {exc}\n")
        log.error(f"llm_clean_tool error: model={model} tool={tool} client={client_id} exc={exc}")
        return msg


async def llm_list() -> str:
    """Return a formatted list of all models in LLM_REGISTRY with their details."""
    if not LLM_REGISTRY:
        return "No models registered."

    lines = ["Available LLM models:\n"]
    for name, cfg in sorted(LLM_REGISTRY.items()):
        available = "YES" if cfg.get("tool_call_available", False) else "NO"
        host = cfg.get("host") or "default"
        lines.append(
            f"  {name}\n"
            f"    type             : {cfg.get('type')}\n"
            f"    model_id         : {cfg.get('model_id')}\n"
            f"    host             : {host}\n"
            f"    max_context      : {cfg.get('max_context')}\n"
            f"    tool_call_avail  : {available}\n"
            f"    llm_call_timeout : {cfg.get('llm_call_timeout', 60)}s\n"
            f"    description      : {cfg.get('description', '')}\n"
        )
    return "\n".join(lines)


async def agent_call(
    agent_url: str,
    message: str,
    target_client_id: str = None,
    stream: bool = True,
) -> str:
    """
    Call another agent-mcp instance (swarm/multi-agent coordination).

    Sends `message` to a remote agent at `agent_url` using the API client plugin.
    The remote agent processes it through its full stack (LLM, tools, gates).
    Returns the complete text response.

    When stream=True (default), remote tokens are relayed via push_tok in real-time
    so Slack and other clients see per-turn progress as it arrives.
    When stream=False, the call blocks silently until the remote agent finishes and
    returns only the final result (original behaviour).

    Depth guard: calls originating from an api-swarm- prefixed client_id are
    rejected immediately to prevent unbounded recursion (max 1 hop).

    Session persistence: the remote session_id is derived deterministically from the
    calling session + agent URL, so repeated calls from the same human session to
    the same remote agent reuse the same remote session (history is preserved).
    Pass target_client_id to override and use a specific named session.
    """
    from api_client import AgentClient

    calling_client = current_client_id.get("")

    # Depth guard — track nesting depth in the calling session
    calling_session = sessions.get(calling_client, {})
    agent_call_depth = calling_session.get("_agent_call_depth", 0)
    _max_agent_call_depth = LIVE_LIMITS.get("max_agent_call_depth", 1)
    if agent_call_depth >= _max_agent_call_depth:
        msg = (
            f"[agent_call] Depth limit reached (max={_max_agent_call_depth}). "
            f"Call rejected to prevent recursion. Do NOT retry."
        )
        log.warning(f"agent_call depth limit: client={calling_client} depth={agent_call_depth}")
        return msg

    # Outbound agent message filter
    block_reason = _check_outbound_agent_message(message)
    if block_reason:
        log.warning(f"agent_call filter block: client={calling_client} reason={block_reason}")
        return f"ERROR: {block_reason}\nMessage was NOT sent to the remote agent."

    # Session-level streaming override: !stream true|false takes precedence over
    # the LLM-supplied stream parameter so the human always has final control.
    session_stream = sessions.get(calling_client, {}).get("agent_call_stream", None)
    if session_stream is not None:
        stream = session_stream

    # Derive a stable swarm client_id from calling session + agent URL so the
    # remote session persists across multiple agent_call invocations (same human
    # session talking to same remote agent = same remote session).
    # The LLM can still override with an explicit target_client_id.
    if target_client_id:
        swarm_client_id = target_client_id
    else:
        import hashlib
        key = f"{calling_client}:{agent_url}"
        swarm_client_id = f"api-swarm-{hashlib.md5(key.encode()).hexdigest()[:8]}"
    api_key = os.getenv("API_KEY", "") or None
    timeout = 120

    await push_tok(calling_client, f"\n[agent_call ▶] {agent_url} → {swarm_client_id}: {message[:100]}{'…' if len(message) > 100 else ''}\n")

    calling_session["_agent_call_depth"] = agent_call_depth + 1
    try:
        client = AgentClient(agent_url, client_id=swarm_client_id, api_key=api_key)

        if stream:
            # Streaming path: relay each remote token via push_tok so Slack and
            # other clients see the remote agent's output as it arrives, rather
            # than waiting for the full response before seeing anything.
            accumulated = []
            async for chunk in client.stream(message, timeout=timeout):
                accumulated.append(chunk)
                await push_tok(calling_client, chunk)
            return "".join(accumulated)
        else:
            # Non-streaming path: block until remote agent finishes, return full result.
            # The LLM narrates the result itself; no push_tok preview here to avoid
            # double-echo in Slack and other clients.
            result = await asyncio.wait_for(
                client.send(message, timeout=timeout),
                timeout=timeout + 5,
            )
            return result

    except asyncio.TimeoutError:
        msg = f"ERROR: agent_call timed out after {timeout}s waiting for {agent_url}."
        await push_tok(calling_client, f"[agent_call ✗] timeout after {timeout}s\n")
        log.warning(f"agent_call timeout: url={agent_url} swarm_client={swarm_client_id}")
        return msg
    except Exception as exc:
        msg = f"ERROR: agent_call failed for {agent_url}: {exc}"
        await push_tok(calling_client, f"[agent_call ✗] {exc}\n")
        log.error(f"agent_call error: url={agent_url} exc={exc}")
        return msg
    finally:
        calling_session["_agent_call_depth"] = agent_call_depth


async def at_llm(model: str, prompt: str) -> str:
    """
    Call a named LLM with the FULL current session context (system prompt + chat history).

    Equivalent to the @<model> prefix syntax but invocable as a tool call.
    The named model receives the assembled system prompt, the complete chat history,
    and the given prompt as the new user turn.

    The result is NOT added to the session history (same behaviour as @<model>).
    All tool gates are bypassed for this call (same as @<model> temp switch).

    client_id is read from current_client_id ContextVar (set by execute_tool).
    """
    client_id = current_client_id.get("")

    cfg = LLM_REGISTRY.get(model)
    if not cfg:
        available = ", ".join(LLM_REGISTRY.keys())
        return f"ERROR: Unknown model '{model}'. Available: {available}"

    session = sessions.get(client_id, {})

    # Depth guard — prevent unbounded recursive at_llm chains
    depth = session.get("_at_llm_depth", 0)
    _max_at_llm_depth = LIVE_LIMITS.get("max_at_llm_depth", 1)
    if depth >= _max_at_llm_depth:
        msg = (
            f"at_llm DEPTH LIMIT REACHED (max={_max_at_llm_depth}): "
            f"Cannot call at_llm from within an at_llm context. "
            f"Do NOT retry. Return your answer directly without calling at_llm again."
        )
        await push_tok(client_id, f"\n[at_llm ✗] depth limit reached (max={_max_at_llm_depth})\n")
        log.warning(f"at_llm depth limit: client={client_id} depth={depth} model={model}")
        return msg

    history = list(session.get("history", []))

    # Append the new prompt as a user turn (not stored in session history)
    history.append({"role": "user", "content": prompt})

    # Run history chain for dispatch (does not modify session["history"])
    from routes import _run_history_chain
    history = _run_history_chain(history, session, cfg)

    await push_tok(client_id, f"\n[@{model} ▶] {prompt[:80]}{'…' if len(prompt) > 80 else ''}\n")

    # Increment depth counter and set gate bypass flag for this call
    prev_temp = session.get("_temp_model_active", False)
    session["_at_llm_depth"] = depth + 1
    session["_temp_model_active"] = True

    try:
        result = await agentic_lc(model, history, client_id)
    finally:
        session["_at_llm_depth"] = depth
        session["_temp_model_active"] = prev_temp

    return result or ""


async def dispatch_llm(model_key: str, messages: list[dict], client_id: str) -> str:
    if model_key not in LLM_REGISTRY:
        await push_err(client_id, f"Unknown model: '{model_key}'")
        return ""

    messages = await auto_enrich_context(messages, client_id)
    return await agentic_lc(model_key, messages, client_id)