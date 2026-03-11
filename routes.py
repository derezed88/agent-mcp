import asyncio
import json
import importlib
import os
from typing import AsyncGenerator
from starlette.requests import Request
from starlette.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from config import log, LLM_REGISTRY, DEFAULT_MODEL, LLM_MODELS_FILE
from state import sessions, get_queue, push_tok, push_done, push_model, active_tasks, cancel_active_task
from database import execute_sql, set_model_context
from agents import dispatch_llm
from tools import get_tool_executor, get_plugin_command, get_plugin_help_sections

# ---------------------------------------------------------------------------
# History plugin chain
# ---------------------------------------------------------------------------
_PLUGINS_ENABLED_PATH = os.path.join(os.path.dirname(__file__), "plugins-enabled.json")

def _load_history_chain() -> list:
    """Load and return the ordered list of history plugin module objects."""
    try:
        with open(_PLUGINS_ENABLED_PATH) as f:
            cfg = json.load(f)
        chain_names = (
            cfg.get("plugin_config", {})
               .get("plugin_history_default", {})
               .get("chain", ["plugin_history_default"])
        )
    except Exception:
        chain_names = ["plugin_history_default"]

    chain = []
    for name in chain_names:
        try:
            mod = importlib.import_module(name)
            chain.append(mod)
        except ImportError as e:
            log.warning(f"History chain: cannot load '{name}': {e}")
    if not chain:
        # Always fall back to default to prevent unguarded raw appends
        import plugin_history_default as _phd
        chain = [_phd]
    return chain

_history_chain: list = _load_history_chain()

def _run_history_chain(history: list[dict], session: dict, model_cfg: dict) -> list[dict]:
    """Pass history through all plugins in chain order. Returns final list."""
    for plugin in _history_chain:
        history = plugin.process(history, session, model_cfg)
    return history

def _notify_chain_model_switch(session: dict, old_model: str, new_model: str,
                                old_cfg: dict, new_cfg: dict) -> list[dict]:
    """Notify each chain plugin of a model switch; return final history."""
    history = list(session.get("history", []))
    for plugin in _history_chain:
        if hasattr(plugin, "on_model_switch"):
            history = plugin.on_model_switch(session, old_model, new_model, old_cfg, new_cfg)
    return history

def _load_system_int(key: str, default: int) -> int:
    """Read a top-level integer from plugins-enabled.json."""
    try:
        with open(_PLUGINS_ENABLED_PATH) as f:
            return int(json.load(f).get(key, default))
    except Exception:
        return default

# Runtime overrides (survive until restart)
_runtime_max_users: int | None = None
_runtime_session_idle_timeout: int | None = None
_runtime_tool_preview_length: int | None = None
_runtime_tool_suppress: bool | None = None

def get_max_users() -> int:
    if _runtime_max_users is not None:
        return _runtime_max_users
    return _load_system_int("max_users", 50)

def get_session_idle_timeout() -> int:
    if _runtime_session_idle_timeout is not None:
        return _runtime_session_idle_timeout
    return _load_system_int("session_idle_timeout_minutes", 60)

def get_default_tool_preview_length() -> int:
    if _runtime_tool_preview_length is not None:
        return _runtime_tool_preview_length
    return _load_system_int("tool_preview_length", 500)

def get_default_tool_suppress() -> bool:
    if _runtime_tool_suppress is not None:
        return _runtime_tool_suppress
    try:
        with open(_PLUGINS_ENABLED_PATH) as f:
            return bool(json.load(f).get("tool_suppress", False))
    except Exception:
        return False

# Context flag for batch command processing
_batch_mode = {}  # client_id -> bool

async def conditional_push_done(client_id: str):
    """Only call push_done if not in batch mode"""
    if not _batch_mode.get(client_id, False):
        await push_done(client_id)

async def cmd_help(client_id: str):
    help_text = (
        "Available commands:\n"
        "  !model                                    - list available models (current marked)\n"
        "  !model <key>                              - switch active LLM\n"
        "  @<model> <prompt>                         - one-turn model switch (e.g. @gpt5m explain this)\n"
        "  !stop                                     - interrupt the running LLM job\n"
        "  !reset                                    - clear conversation history\n"
        "  !help                                     - this help\n"
        "  !input_lines <n>                          - resize input area (client-side only)\n"
        "  !tools                                    - show tool heat status (hot/cold) for current model\n"
        "  !tools reset                              - decay all tool subscriptions to cold\n"
        "  !tools hot <toolset,...>                  - force-subscribe toolsets immediately\n"
        "\n"
        "Database:\n"
        "  !db_query <sql>                           - run SQL directly (no LLM)\n"
        "\n"
        "Memory:\n"
        "  !memory [true|false]                      - enable/disable memory logging (MySQL+Qdrant)\n"
        "  !memory list [short|long]                 - list by tier\n"
        "  !memory show <id> [short|long]            - show one row in full\n"
        "  !memory update <id> [tier=short] [importance=N] [content=text] [topic=label]\n"
        "  !memstats                                 - memory system health dashboard\n"
        "  !membackfill                              - embed missing rows into Qdrant vector index\n"
        "  !memreconcile                             - remove orphaned Qdrant points not in MySQL\n"
        "  !memreview [approve N,N|reject N,N|clear] - AI topic review with HITL approval\n"
        "  !memreview types                          - review typed memory tables (goals, beliefs, etc.) [memory_types_enabled]\n"
        "  !memreview classify                       - classify untyped memory rows into memory types\n"
        "  !memclassify [status|approve N|reject N]  - background classify (auto-apply ≥0.80, HITL <0.80)\n"
        "  !memage                                   - manually trigger one topic-chunk aging pass\n"
        "  !memtrim [N]                              - hard-delete N oldest ST rows (escape valve; default: trim to LWM)\n"
        "\n"
        "Proactive Cognition:\n"
        "  !cogn                                     - status dashboard for all cognition timers + stats\n"
        "  !cogn on|off                              - master enable/disable (runtime, no restart)\n"
        "  !cogn contradiction|prospective|reflection on|off|run\n"
        "                                            - per-loop toggle or immediate trigger\n"
        "  !cogn interval contradiction|prospective|reflection <value>\n"
        "                                            - set loop interval (h or m) at runtime\n"
        "  !cogn model contradiction|prospective|reflection <key>\n"
        "                                            - set loop model at runtime\n"
        "  !cogn flags [clear]                       - view/retract open contradiction-flag beliefs\n"
        "  !cogn feedback reset <loop>               - reset feedback streak/conditioning for a loop\n"
        "  !cogn reset                               - revert all runtime overrides to json config\n"
        "\n"
        "Drive / Affect:\n"
        "  !drives                                   - show current drive values\n"
        "  !drives set <name> <0.0-1.0>              - set a drive value\n"
        "  !drives set <name> <0.0-1.0> <description> - set drive value and description\n"
        "  !drives baseline <name> <0.0-1.0>         - set baseline equilibrium for a drive\n"
        "  !drives decay                             - manually trigger one decay cycle\n"
        "  !drives seed                              - seed default drives from config if table empty\n"
        "\n"
        "Search & Extract:\n"
        "  !search_ddgs <query>                      - search via DuckDuckGo\n"
        "  !search_google <query>                    - search via Google (Gemini grounding)\n"
        "  !search_tavily <query>                    - search via Tavily AI\n"
        "  !search_xai <query>                       - search via xAI Grok\n"
        "  !url_extract <url> [query]                - extract web page content\n"
        "\n"
        "Google Drive:\n"
        "  !google_drive <operation> [args...]       - Drive CRUD operation\n"
        "\n"
        "VSCode Sessions:\n"
        "  !vscode list [date=YYYY-MM-DD] [project=name]   - list local Claude Code sessions\n"
        "  !vscode read <id>[,<id>...] [mode=full|summary] [model=<key>]  - pull session into context\n"
        "\n"
        + "".join(get_plugin_help_sections())
        + "\n"
        "Session & History:\n"
        "  !session                                  - list all active sessions\n"
        "  !session <ID> delete                      - delete a session from server\n"
        "\n"
        "Utilities:\n"
        "  !get_system_info                          - show date/time/status\n"
        "  !llm_list                                 - list LLM models\n"
        "  !llm_call_invoke <model> <prompt> [opts]  - call a model directly (see options below)\n"
        "    opts: mode=text|tool  sys_prompt=none|caller|target  history=none|caller  tool=<name>\n"
        "  !sleep <seconds>                          - sleep 1-300 seconds\n"
        "\n"
        "Unified Resource Commands:\n"
        "  !llm_tools [list|read|write|delete|add] [name] [tools]\n"
        "    list                                     - show all toolsets and model assignments\n"
        "    read <name>                              - show one toolset\n"
        "    write <name> <tool1,tool2,...>            - overwrite toolset\n"
        "    delete <name>                            - remove toolset\n"
        "    add <name> <tool1,tool2,...>              - add tools to toolset\n"
        "  !model_cfg [list|read|write|copy|delete|enable|disable] [name] [field] [value]\n"
        "    list                                     - show all models with toolsets\n"
        "    read <model>                             - show full model config\n"
        "    write <model> <field> <value>            - set a model field\n"
        "    copy <source> <new_name>                 - copy model\n"
        "    delete <model>                           - delete model\n"
        "    enable/disable <model>                   - enable/disable model\n"
        "  !model_cfg write <model> llm_tools_gates <entry1,entry2,...>\n"
        "    Configure tool call gates requiring human approval before execution.\n"
        "    Each entry is either a tool name or 'tool_name subcommand'.\n"
        "    Examples:\n"
        "      !model_cfg write gemini25f llm_tools_gates db_query\n"
        "      !model_cfg write gemini25f llm_tools_gates db_query,model_cfg write\n"
        "      !model_cfg write gemini25f llm_tools_gates   (empty = clear all gates)\n"
        "    Gate behavior by client:\n"
        "      shell.py : prompts user; next input is Y/N\n"
        "      llama/slack : auto-denied (cannot prompt interactively)\n"
        "      timeout 120s: auto-denied if no response\n"
        "  !sysprompt_cfg [list_dir|list|read|write|delete|copy_dir|set_dir] [model] [file|dir] [content]\n"
        "  !config [list|read|write] [key] [value]\n"
        "    list                                     - show all config settings\n"
        "    read <key>                               - show one setting\n"
        "    write <key> <value>                      - set a setting\n"
        "  !limits [list|read|write] [key] [value]\n"
        "    list                                     - show all depth/rate limits\n"
        "    read <key>                               - show one limit\n"
        "    write <key> <value>                      - set a limit\n"
        "\n"
        "Judge (LLM-as-judge enforcement):\n"
        "  !judge                                    - show judge config for current model/session\n"
        "  !judge list                               - list judge configs for all models\n"
        "  !judge on <gate|all>                      - enable gate (prompt/response/tool/memory/all)\n"
        "  !judge off <gate|all>                     - disable gate for this session\n"
        "  !judge model <name>                       - set judge model for this session\n"
        "  !judge mode block|warn                    - block (deny) or warn (log+allow)\n"
        "  !judge threshold <0.0-1.0>                - set score threshold\n"
        "  !judge reset                              - clear all session overrides\n"
        "  !judge test <text>                        - run ad-hoc evaluation\n"
        "  !judge set <model> <field> <value>        - persist judge_config to llm-models.json\n"
        "    fields: model, mode, threshold, gates\n"
        "  Note: enable plugin_history_judge in chain for tool+memory gates to be active.\n"
        "\n"
        "AI tools:\n"
        "  get_system_info()                         - show date/time/status\n"
        "  llm_call(model, prompt, ...)              - call a model (mode/sys_prompt/history/tool)\n"
        "  llm_list()                                - list LLM models\n"
        "  agent_call(agent_url, message)            - send message to remote agent\n"
        "  sleep(seconds)                            - pause 1-300s\n"
        "  session/model/reset                       - session management\n"
        "  llm_tools/model_cfg/sysprompt_cfg/config_cfg/limits_cfg - resource management\n"
    )
    await push_tok(client_id, help_text)
    await conditional_push_done(client_id)

async def cmd_db_query(client_id: str, sql: str, model_key: str = ""):
    """Execute SQL directly without LLM."""
    sql = sql.strip()
    if not sql:
        await push_tok(client_id,
            "Usage: !db_query <SQL>\n"
            "Examples:\n"
            "  !db_query SELECT * FROM users LIMIT 5\n"
            "  !db_query SHOW TABLES")
        await conditional_push_done(client_id)
        return
    set_model_context(model_key)
    try:
        result = await execute_sql(sql)
        await push_tok(client_id, result)
    except Exception as exc:
        await push_tok(client_id, f"ERROR: Database query failed\n{exc}")
    await conditional_push_done(client_id)


async def cmd_memory(client_id: str, arg: str):
    """
    !memory                         - list all short-term memories
    !memory list [short|long]       - list short or long-term memories
    !memory show <id> [short|long]  - show one row in full
    !memory update <id> [tier=short] [importance=N] [content=...] [topic=...]
    """
    from memory import load_short_term, load_long_term, update_memory, _mem_plugin_cfg
    parts = arg.split(maxsplit=1)
    subcmd = parts[0].lower() if parts else "list"
    rest = parts[1].strip() if len(parts) > 1 else ""

    if subcmd in ("list", "short", "long", ""):
        tier = "long" if subcmd == "long" else "short"
        if subcmd == "list" and rest in ("long", "short"):
            tier = rest
        _mem_flag = sessions[client_id].get("memory_enabled", None)
        _global_mem = _mem_plugin_cfg().get("enabled", True) if _mem_flag is None else None
        _mem_active = _mem_flag if _mem_flag is not None else _global_mem
        _mem_src = "session" if _mem_flag is not None else "global"
        await push_tok(client_id, f"Memory logging: {'ON' if _mem_active else 'OFF'} ({_mem_src})")
        rows = await (load_long_term(limit=100) if tier == "long" else load_short_term(limit=100, min_importance=1))
        if not rows:
            await push_tok(client_id, f"No {tier}-term memories.")
        else:
            lines = [f"{tier}-term memory ({len(rows)} rows):\n"]
            for r in rows:
                imp = r.get("importance", "?")
                lines.append(f"  [{r['id']:>3}] imp={imp} [{r.get('topic','')}] {r.get('content','')}")
            await push_tok(client_id, "\n".join(lines))

    elif subcmd == "show":
        tparts = rest.split()
        if not tparts:
            await push_tok(client_id, "Usage: !memory show <id> [short|long]")
            await conditional_push_done(client_id)
            return
        try:
            row_id = int(tparts[0])
        except ValueError:
            await push_tok(client_id, f"Invalid id: {tparts[0]}")
            await conditional_push_done(client_id)
            return
        tier = tparts[1] if len(tparts) > 1 and tparts[1] in ("short", "long") else "short"
        rows = await (load_long_term(limit=1000) if tier == "long" else load_short_term(limit=1000, min_importance=1))
        row = next((r for r in rows if str(r.get("id")) == str(row_id)), None)
        if not row:
            await push_tok(client_id, f"No {tier}-term row with id={row_id}.")
        else:
            await push_tok(client_id, "\n".join(f"  {k}: {v}" for k, v in row.items()))

    elif subcmd in ("true", "false", "on", "off"):
        # Per-session on/off switch — stored in session dict, not global config
        val = subcmd in ("true", "on")
        sessions[client_id]["memory_enabled"] = val
        from state import save_session_config
        save_session_config(client_id, sessions[client_id])
        state_str = "ON" if val else "OFF"
        await push_tok(client_id,
            f"Memory logging {state_str} for this session.\n"
            + ("  conv_log writes to MySQL+Qdrant re-enabled." if val else
               "  conv_log writes to MySQL+Qdrant suppressed. !reset will wipe history without summarizing.")
        )

    elif subcmd == "update":
        tparts = rest.split(maxsplit=1)
        if not tparts:
            await push_tok(client_id, "Usage: !memory update <id> [tier=short] [importance=N] [content=...] [topic=...]")
            await conditional_push_done(client_id)
            return
        try:
            row_id = int(tparts[0])
        except ValueError:
            await push_tok(client_id, f"Invalid id: {tparts[0]}")
            await conditional_push_done(client_id)
            return
        kwargs_str = tparts[1] if len(tparts) > 1 else ""
        # Parse key=value pairs; content/topic may contain spaces so grab them last
        import re as _re
        tier = "short"
        importance = None
        content = None
        topic = None
        m = _re.search(r'\btier=(short|long)\b', kwargs_str)
        if m:
            tier = m.group(1)
            kwargs_str = kwargs_str[:m.start()] + kwargs_str[m.end():]
        m = _re.search(r'\bimportance=(\d+)\b', kwargs_str)
        if m:
            importance = int(m.group(1))
            kwargs_str = kwargs_str[:m.start()] + kwargs_str[m.end():]
        m = _re.search(r'\btopic=(\S+)', kwargs_str)
        if m:
            topic = m.group(1)
            kwargs_str = kwargs_str[:m.start()] + kwargs_str[m.end():]
        m = _re.search(r'\bcontent=(.+)', kwargs_str, _re.DOTALL)
        if m:
            content = m.group(1).strip()
        result = await update_memory(row_id=row_id, tier=tier,
                                     importance=importance, content=content, topic=topic)
        await push_tok(client_id, result)

    else:
        await push_tok(client_id,
            "Usage:\n"
            "  !memory [true|false]                 - enable/disable memory logging (MySQL+Qdrant)\n"
            "  !memory list [short|long]            - list by tier\n"
            "  !memory show <id> [short|long]       - show one row\n"
            "  !memory update <id> [tier=short] [importance=N] [content=text] [topic=label]")

    await conditional_push_done(client_id)


async def cmd_memstats(client_id: str, model_key: str = ""):
    """
    !memstats — memory system health dashboard.
    Shows row counts, topic distribution, importance spread, aging history,
    summarizer runs, dedup config, and last activity timestamps.
    """
    set_model_context(model_key)
    from database import execute_sql
    from memory import _ST, _LT, _SUM, _COLLECTION, _age_cfg, _mem_plugin_cfg, get_retrieval_stats, get_typed_metrics, _GOALS, _PLANS, _BELIEFS

    lines = ["## Memory System Stats\n"]

    async def q(sql: str) -> str:
        try:
            return await execute_sql(sql)
        except Exception as e:
            return f"(error: {e})"

    def _int_from(raw: str) -> int:
        for line in raw.strip().splitlines():
            line = line.strip()
            if line.isdigit():
                return int(line)
            parts = line.split()
            if parts and parts[-1].isdigit():
                return int(parts[-1])
        return 0

    def _rows_from(raw: str) -> list[list[str]]:
        """Parse pipe-separated output into rows of stripped strings."""
        result = []
        for line in raw.strip().splitlines()[1:]:  # skip header
            if not line.strip() or set(line.strip()) <= set("-+|"):
                continue
            result.append([c.strip() for c in line.split("|")])
        return result

    # ── Tier counts ──────────────────────────────────────────────────────────
    st_count = _int_from(await q(f"SELECT COUNT(*) FROM {_ST()}"))
    lt_count = _int_from(await q(f"SELECT COUNT(*) FROM {_LT()}"))
    sum_count = _int_from(await q(f"SELECT COUNT(*) FROM {_SUM()}"))
    lines.append(f"**Tier counts**")
    lines.append(f"  short-term : {st_count} rows")
    lines.append(f"  long-term  : {lt_count} rows")
    lines.append(f"  summaries  : {sum_count} rows\n")

    # ── Short-term: topic breakdown ───────────────────────────────────────────
    st_topics_raw = await q(
        f"SELECT topic, COUNT(*) as n, ROUND(AVG(importance),1) as avg_imp "
        f"FROM {_ST()} GROUP BY topic ORDER BY n DESC LIMIT 20"
    )
    st_topic_rows = _rows_from(st_topics_raw)
    if st_topic_rows:
        lines.append(f"**Short-term by topic** (top {len(st_topic_rows)})")
        for row in st_topic_rows:
            if len(row) >= 3:
                lines.append(f"  {row[0]:<30} {row[1]:>4} rows  avg_imp={row[2]}")
        lines.append("")

    # ── Long-term: topic breakdown ────────────────────────────────────────────
    lt_topics_raw = await q(
        f"SELECT topic, COUNT(*) as n, ROUND(AVG(importance),1) as avg_imp "
        f"FROM {_LT()} GROUP BY topic ORDER BY n DESC LIMIT 20"
    )
    lt_topic_rows = _rows_from(lt_topics_raw)
    if lt_topic_rows:
        lines.append(f"**Long-term by topic** (top {len(lt_topic_rows)})")
        for row in lt_topic_rows:
            if len(row) >= 3:
                lines.append(f"  {row[0]:<30} {row[1]:>4} rows  avg_imp={row[2]}")
        lines.append("")

    # ── Importance distribution (short-term) ─────────────────────────────────
    imp_raw = await q(
        f"SELECT importance, COUNT(*) as n FROM {_ST()} GROUP BY importance ORDER BY importance"
    )
    imp_rows = _rows_from(imp_raw)
    if imp_rows:
        lines.append("**Short-term importance distribution**")
        dist = "  " + "  ".join(f"[{r[0]}]={r[1]}" for r in imp_rows if len(r) >= 2)
        lines.append(dist)
        lines.append("")

    # ── Source breakdown (short-term) ─────────────────────────────────────────
    src_raw = await q(
        f"SELECT source, COUNT(*) as n FROM {_ST()} GROUP BY source ORDER BY n DESC"
    )
    src_rows = _rows_from(src_raw)
    if src_rows:
        lines.append("**Short-term by source**")
        for row in src_rows:
            if len(row) >= 2:
                lines.append(f"  {row[0]:<20} {row[1]:>4} rows")
        lines.append("")

    # ── Type distribution (short-term + long-term) ───────────────────────────
    st_type_raw = await q(
        f"SELECT type, COUNT(*) as n FROM {_ST()} GROUP BY type ORDER BY n DESC"
    )
    lt_type_raw = await q(
        f"SELECT type, COUNT(*) as n FROM {_LT()} GROUP BY type ORDER BY n DESC"
    )
    st_type_rows = _rows_from(st_type_raw)
    lt_type_rows = _rows_from(lt_type_raw)
    if st_type_rows or lt_type_rows:
        lines.append("**Memory type distribution**")
        all_types = {}
        for row in st_type_rows:
            if len(row) >= 2:
                all_types.setdefault(row[0], [0, 0])[0] = int(row[1])
        for row in lt_type_rows:
            if len(row) >= 2:
                all_types.setdefault(row[0], [0, 0])[1] = int(row[1])
        for t, (sc, lc) in sorted(all_types.items(), key=lambda x: -(x[1][0]+x[1][1])):
            lines.append(f"  {t:<20} ST={sc:>4}  LT={lc:>4}")
        lines.append("")

    # ── Aging: long-term source of truth ─────────────────────────────────────
    lt_from_st_raw = await q(
        f"SELECT COUNT(*) FROM {_LT()} WHERE shortterm_id IS NOT NULL AND shortterm_id > 0"
    )
    lt_aged = _int_from(lt_from_st_raw)
    lt_direct_raw = await q(
        f"SELECT COUNT(*) FROM {_LT()} WHERE shortterm_id IS NULL OR shortterm_id = 0"
    )
    lt_direct = _int_from(lt_direct_raw)
    lines.append("**Long-term origin**")
    lines.append(f"  aged from short-term : {lt_aged} rows")
    lines.append(f"  direct saves         : {lt_direct} rows\n")

    # ── Last activity timestamps ──────────────────────────────────────────────
    st_newest_raw = await q(f"SELECT MAX(created_at) FROM {_ST()}")
    st_oldest_raw = await q(f"SELECT MIN(created_at) FROM {_ST()}")
    st_lru_raw    = await q(f"SELECT MIN(last_accessed) FROM {_ST()}")
    lt_newest_raw = await q(f"SELECT MAX(created_at) FROM {_LT()}")
    sum_newest_raw = await q(f"SELECT MAX(created_at) FROM {_SUM()}")

    def _scalar(raw: str) -> str:
        for line in raw.strip().splitlines():
            line = line.strip()
            if line and not line.startswith("MAX") and not line.startswith("MIN") and not set(line).issubset(set("-+| ")):
                return line
        return "(none)"

    lines.append("**Timestamps**")
    lines.append(f"  ST newest created    : {_scalar(st_newest_raw)}")
    lines.append(f"  ST oldest created    : {_scalar(st_oldest_raw)}")
    lines.append(f"  ST least recently    : {_scalar(st_lru_raw)}")
    lines.append(f"  LT newest created    : {_scalar(lt_newest_raw)}")
    lines.append(f"  Summaries newest     : {_scalar(sum_newest_raw)}\n")

    # ── Summarizer runs ───────────────────────────────────────────────────────
    if sum_count > 0:
        sum_model_raw = await q(
            f"SELECT model_used, COUNT(*) as n FROM {_SUM()} GROUP BY model_used ORDER BY n DESC"
        )
        sum_model_rows = _rows_from(sum_model_raw)
        if sum_model_rows:
            lines.append("**Summarizer runs by model**")
            for row in sum_model_rows:
                if len(row) >= 2:
                    lines.append(f"  {row[0]:<30} {row[1]:>3} runs")
            lines.append("")

    # ── Vector index stats ────────────────────────────────────────────────────
    try:
        from plugin_memory_vector_qdrant import get_vector_api
        vec = get_vector_api()
        if vec:
            from memory import _COLLECTION
            vstats = await vec.get_stats(collection=_COLLECTION())
            vcfg   = vec.cfg()
            qd = vstats.get("qdrant", {})
            em = vstats.get("embed", {})

            lines.append("**Vector Index (Qdrant)**")
            if "error" in qd:
                lines.append(f"  status               : ERROR — {qd['error']}")
            else:
                mysql_total = st_count + lt_count
                qdrant_pts  = qd.get("points_count", "?")
                drift = ""
                if isinstance(qdrant_pts, int) and mysql_total > 0:
                    diff = mysql_total - qdrant_pts
                    if diff != 0:
                        drift = f"  !! drift: MySQL={mysql_total} Qdrant={qdrant_pts} (diff={diff:+d})"
                lines.append(f"  status               : {qd.get('status', '?')}")
                lines.append(f"  points_count         : {qdrant_pts}  (MySQL ST+LT={mysql_total})")
                if drift:
                    lines.append(drift)
                lines.append(f"  indexed_vectors      : {qd.get('indexed_vectors_count', '?')}")
                lines.append(f"  segments             : {qd.get('segments_count', '?')}")
                lines.append(f"  optimizer            : {qd.get('optimizer_status', '?')}")
                lines.append(f"  collection           : {vcfg.get('collection')}  @{vcfg.get('qdrant_host')}:{vcfg.get('qdrant_port')}")
                lines.append(f"  top_k / min_score    : {vcfg.get('top_k')} / {vcfg.get('min_score')}")
                lines.append(f"  always_inject_imp>=  : {vcfg.get('min_importance_always')}")
            lines.append("")

            lines.append("**Embedding Server (nomic-embed-text)**")
            lines.append(f"  health               : {em.get('health', '?')}")
            if "kv_cache_pct" in em:
                lines.append(f"  kv_cache_usage       : {em['kv_cache_pct']}")
            if "kv_cache_tokens" in em:
                lines.append(f"  kv_cache_tokens      : {em['kv_cache_tokens']}")
            if "embed_tps" in em:
                lines.append(f"  embed_throughput     : {em['embed_tps']}")
            if "requests_processing" in em:
                lines.append(f"  requests_processing  : {em['requests_processing']}")
            if "requests_deferred" in em:
                lines.append(f"  requests_deferred    : {em['requests_deferred']}")
            if "metrics" in em:
                lines.append(f"  metrics endpoint     : {em['metrics']}")
            lines.append(f"  url                  : {vcfg.get('embed_url')}")
            lines.append("")
        else:
            lines.append("**Vector Index**: not loaded\n")
    except Exception as _vec_err:
        lines.append(f"**Vector Index**: error — {_vec_err}\n")

    # ── Retrieval stats (two-pass) ───────────────────────────────────────────
    rstats = get_retrieval_stats()
    if rstats["total"] > 0:
        single = rstats["single_pass_sufficient"]
        two = rstats["two_pass_needed"]
        fallback = rstats["fallback_no_vec"]
        total = rstats["total"]
        single_pct = int(single / total * 100) if total else 0
        two_pct = int(two / total * 100) if total else 0
        lines.append("**Retrieval Stats (since restart)**")
        lines.append(f"  total retrievals     : {total}")
        lines.append(f"  single-pass sufficient: {single} ({single_pct}%)")
        lines.append(f"  two-pass needed      : {two} ({two_pct}%)")
        lines.append(f"  fallback (no vector) : {fallback}")
        qfloor = float(_mem_plugin_cfg().get("two_pass_quality_floor", 0.75))
        lines.append(f"  pass-1 avg quality hits: {rstats['pass1_avg_hits']:.1f} (score ≥ {qfloor})")
        if two > 0:
            lines.append(f"  pass-2 avg extra hits: {rstats['pass2_avg_extra']:.1f}")
        lines.append("")

    # ── Typed tables (goals/plans/beliefs) ───────────────────────────────────
    try:
        goals_total   = _int_from(await q(f"SELECT COUNT(*) FROM {_GOALS()}"))
        goals_active  = _int_from(await q(f"SELECT COUNT(*) FROM {_GOALS()} WHERE status='active'"))
        plans_total   = _int_from(await q(f"SELECT COUNT(*) FROM {_PLANS()}"))
        plans_pending = _int_from(await q(f"SELECT COUNT(*) FROM {_PLANS()} WHERE status IN ('pending','in_progress')"))
        beliefs_total  = _int_from(await q(f"SELECT COUNT(*) FROM {_BELIEFS()}"))
        beliefs_active = _int_from(await q(f"SELECT COUNT(*) FROM {_BELIEFS()} WHERE status='active'"))
        tm = get_typed_metrics()
        lines.append("**Typed Tables (goals / plans / beliefs)**")
        lines.append(f"  goals    : {goals_active} active / {goals_total} total  writes={tm['goals']['writes']}  reads={tm['goals']['reads']}")
        lines.append(f"  plans    : {plans_pending} pending / {plans_total} total  writes={tm['plans']['writes']}  reads={tm['plans']['reads']}")
        lines.append(f"  beliefs  : {beliefs_active} active / {beliefs_total} total  writes={tm['beliefs']['writes']}  reads={tm['beliefs']['reads']}")
        lines.append("  (writes/reads reset on restart)")
        lines.append("")
    except Exception as _te:
        lines.append(f"**Typed Tables**: error — {_te}\n")

    # ── Config snapshot ───────────────────────────────────────────────────────
    mem_cfg = _mem_plugin_cfg()
    age_cfg = _age_cfg()
    lines.append("**Config (plugins-enabled.json)**")
    master_on = mem_cfg.get("enabled", True)
    lines.append(f"  {'enabled (master)':<28}: {'on' if master_on else 'OFF'}")
    _sess_mem = sessions[client_id].get("memory_enabled", None) if client_id in sessions else None
    if _sess_mem is not None:
        lines.append(f"  {'enabled (this session)':<28}: {'on' if _sess_mem else 'OFF'} (overrides master)")
    features = ("context_injection", "reset_summarize", "post_response_scan", "fuzzy_dedup", "vector_search_qdrant")
    for f in features:
        val = mem_cfg.get(f, True)
        lines.append(f"  {f:<28}: {'on' if val else 'OFF'}{' (inactive—master off)' if not master_on else ''}")
    lines.append(f"  {'fuzzy_dedup_threshold':<28}: {mem_cfg.get('fuzzy_dedup_threshold', 0.78):.2f}")
    lines.append(f"  {'summarizer_model':<28}: {mem_cfg.get('summarizer_model', '(not set)')}")
    hwm = age_cfg["short_hwm"]
    lwm = age_cfg["short_lwm"]
    lines.append(f"  {'auto_memory_age':<28}: {'on' if age_cfg['auto_memory_age'] else 'OFF'}")
    lines.append(f"  {'short_hwm (aging trigger)':<28}: {hwm} rows")
    lines.append(f"  {'short_lwm (aging target)':<28}: {lwm} rows")
    lines.append(f"  {'recent_turns_protect':<28}: {age_cfg['recent_turns_protect']} turns")
    lines.append(f"  {'staleness_override_minutes':<28}: {age_cfg['staleness_override_minutes']} min ({age_cfg['staleness_override_minutes']//60}h)")
    lines.append(f"  {'chunk_importance_threshold':<28}: {age_cfg['chunk_importance_threshold']}")

    def _timer(v: int) -> str:
        return "disabled" if v == -1 else f"{v} min"

    lines.append(f"  {'memory_age_count_timer':<28}: {_timer(age_cfg['memory_age_count_timer'])}")
    lines.append(f"  {'memory_age_trigger_minutes':<28}: {age_cfg['memory_age_trigger_minutes']} min ({age_cfg['memory_age_trigger_minutes']//60}h)")
    lines.append(f"  {'memory_age_minutes_timer':<28}: {_timer(age_cfg['memory_age_minutes_timer'])}")

    # Pressure bar vs HWM/LWM
    pct_hwm = int(st_count / hwm * 100) if hwm > 0 else 0
    filled  = min(20, pct_hwm // 5)
    lwm_pos = min(20, int(lwm / hwm * 20)) if hwm > 0 else 0
    bar_chars = list("." * 20)
    for i in range(filled):
        bar_chars[i] = "#"
    if 0 <= lwm_pos < 20:
        bar_chars[lwm_pos] = "|"  # LWM marker
    bar = "".join(bar_chars)
    status = "AGING NOW" if st_count >= hwm else ("near HWM" if pct_hwm >= 80 else "ok")
    lines.append(f"\n  ST pressure  [{bar}] {st_count}/{hwm} rows ({pct_hwm}%)  LWM={lwm}  [{status}]")

    # ── Proactive Cognition timers (summary) ──────────────────────────────
    try:
        from contradiction import get_contradiction_stats, _cogn_cfg as _ccfg
        from prospective import get_prospective_stats
        from reflection import get_reflection_stats
        cogn_cfg = _ccfg()
        master_on = cogn_cfg["enabled"]
        lines.append("\n**Proactive Cognition (use !cogn for full detail)**")
        lines.append(f"  master: {'ON' if master_on else 'OFF'}")

        cs = get_contradiction_stats()
        cscan_on = cogn_cfg["contradiction_enabled"]
        lines.append(
            f"  contradiction  {'ON' if master_on and cscan_on else 'OFF'}  "
            f"scans={cs['scans_run']}  flags={cs['flags_written']}  "
            f"last={cs['last_scan_at'] or 'never'}"
        )

        ps = get_prospective_stats()
        pscan_on = cogn_cfg.get("prospective_enabled", True)
        lines.append(
            f"  prospective    {'ON' if master_on and pscan_on else 'OFF'}  "
            f"checks={ps['checks_run']}  fired={ps['rows_fired']}  "
            f"last={ps['last_check_at'] or 'never'}"
        )

        rs = get_reflection_stats()
        rscan_on = cogn_cfg.get("reflection_enabled", True)
        lines.append(
            f"  reflection     {'ON' if master_on and rscan_on else 'OFF'}  "
            f"runs={rs['runs']}  saved={rs['memories_saved']}  "
            f"last={rs['last_run_at'] or 'never'}"
        )
    except ImportError:
        pass
    except Exception as _cogn_err:
        lines.append(f"\n**Proactive Cognition**: error — {_cogn_err}")

    await push_tok(client_id, "\n".join(lines))
    await conditional_push_done(client_id)


async def cmd_membackfill(client_id: str, model_key: str = ""):
    """
    !membackfill — embed and upsert any MySQL memory rows missing from Qdrant.
    Compares all MySQL row IDs against Qdrant point IDs; only processes the gap.
    """
    if await _guard_utility_model(client_id, model_key, "membackfill"):
        return
    set_model_context(model_key)
    from database import fetch_dicts
    from memory import _ST, _LT, _COLLECTION

    try:
        from plugin_memory_vector_qdrant import get_vector_api
        vec = get_vector_api()
        if not vec:
            await push_tok(client_id, "Vector plugin not available.")
            await conditional_push_done(client_id)
            return
    except Exception as e:
        await push_tok(client_id, f"Vector plugin error: {e}")
        await conditional_push_done(client_id)
        return

    await push_tok(client_id, "Scanning for missing Qdrant points...\n")

    coll = _COLLECTION()
    try:
        qdrant_ids = vec.get_all_point_ids(collection=coll)
    except Exception as e:
        await push_tok(client_id, f"Failed to fetch Qdrant point IDs: {e}")
        await conditional_push_done(client_id)
        return

    st_rows = await fetch_dicts(f"SELECT id, topic, content, importance FROM {_ST()}")
    lt_rows = await fetch_dicts(f"SELECT id, topic, content, importance FROM {_LT()}")

    mysql_ids = {int(r["id"]) for r in st_rows} | {int(r["id"]) for r in lt_rows}
    missing_st = [r for r in st_rows if int(r["id"]) not in qdrant_ids]
    missing_lt = [r for r in lt_rows if int(r["id"]) not in qdrant_ids]
    total_missing = len(missing_st) + len(missing_lt)
    orphan_count = len(qdrant_ids - mysql_ids)

    # Report metrics
    report = (
        f"Collection:     {coll}\n"
        f"Qdrant points:  {len(qdrant_ids)}\n"
        f"MySQL rows:     {len(mysql_ids)} ({len(st_rows)} ST, {len(lt_rows)} LT)\n"
        f"In sync:        {len(qdrant_ids & mysql_ids)}\n"
        f"Missing from Q: {total_missing} ({len(missing_st)} ST, {len(missing_lt)} LT)\n"
        f"Orphans in Q:   {orphan_count} (use !memreconcile to clean)\n"
    )
    await push_tok(client_id, report)

    if total_missing == 0:
        await push_tok(client_id, "No missing points — Qdrant has all MySQL rows.")
        await conditional_push_done(client_id)
        return

    await push_tok(client_id, f"Backfilling {total_missing} missing rows...")

    saved_st = await vec.backfill(missing_st, tier="short", collection=coll) if missing_st else 0
    saved_lt = await vec.backfill(missing_lt, tier="long",  collection=coll) if missing_lt else 0

    await push_tok(client_id, f"Done. Upserted {saved_st} short-term + {saved_lt} long-term rows into Qdrant.")
    await conditional_push_done(client_id)


async def cmd_memreconcile(client_id: str, model_key: str = ""):
    """
    !memreconcile — remove orphaned Qdrant points whose MySQL rows no longer exist.
    Inverse of !membackfill: Qdrant has points that MySQL doesn't → delete from Qdrant.
    Reports metrics: total Qdrant points, MySQL rows, orphans found, orphans deleted.
    """
    if await _guard_utility_model(client_id, model_key, "memreconcile"):
        return
    set_model_context(model_key)
    from database import fetch_dicts
    from memory import _ST, _LT, _COLLECTION

    try:
        from plugin_memory_vector_qdrant import get_vector_api
        vec = get_vector_api()
        if not vec:
            await push_tok(client_id, "Vector plugin not available.")
            await conditional_push_done(client_id)
            return
    except Exception as e:
        await push_tok(client_id, f"Vector plugin error: {e}")
        await conditional_push_done(client_id)
        return

    await push_tok(client_id, "Scanning for orphaned Qdrant points...\n")

    coll = _COLLECTION()
    try:
        qdrant_ids = vec.get_all_point_ids(collection=coll)
    except Exception as e:
        await push_tok(client_id, f"Failed to fetch Qdrant point IDs: {e}")
        await conditional_push_done(client_id)
        return

    # Collect all valid MySQL IDs from both tiers
    st_ids = await fetch_dicts(f"SELECT id FROM {_ST()}")
    lt_ids = await fetch_dicts(f"SELECT id FROM {_LT()}")
    mysql_ids = {int(r["id"]) for r in st_ids} | {int(r["id"]) for r in lt_ids}

    orphan_ids = qdrant_ids - mysql_ids
    in_sync = qdrant_ids & mysql_ids

    # Report metrics
    report = (
        f"Collection:     {coll}\n"
        f"Qdrant points:  {len(qdrant_ids)}\n"
        f"MySQL rows:     {len(mysql_ids)}\n"
        f"In sync:        {len(in_sync)}\n"
        f"Orphans found:  {len(orphan_ids)}\n"
    )
    await push_tok(client_id, report)

    if not orphan_ids:
        await push_tok(client_id, "No orphans — Qdrant is fully reconciled with MySQL.")
        await conditional_push_done(client_id)
        return

    # Delete orphaned points in batches
    deleted = 0
    batch_size = 500
    orphan_list = sorted(orphan_ids)
    for i in range(0, len(orphan_list), batch_size):
        batch = orphan_list[i : i + batch_size]
        try:
            vec._qc.delete(
                collection_name=coll,
                points_selector=batch,
            )
            deleted += len(batch)
        except Exception as e:
            await push_tok(client_id, f"Delete batch failed at offset {i}: {e}\n")

    await push_tok(client_id, f"Deleted {deleted}/{len(orphan_ids)} orphaned Qdrant points.")
    await conditional_push_done(client_id)


# ---------------------------------------------------------------------------
# Utility model guard — prevents !mem* LLM commands from running when the
# session model is a service model (summarizer, reviewer, classifier, judge).
# These models are never end-user session models; running mem commands from
# them would cause recursive llm_call loops.
# ---------------------------------------------------------------------------

_UTILITY_MODEL_PREFIXES = ("summarizer-", "reviewer-", "judge-", "extractor-")


def _is_utility_model(model_key: str) -> bool:
    """Return True if model_key is a service/utility model that should not run !mem* LLM commands."""
    if not model_key:
        return False
    return any(model_key.startswith(p) for p in _UTILITY_MODEL_PREFIXES)


async def _guard_utility_model(client_id: str, model_key: str, cmd_name: str) -> bool:
    """Push an error and return True if the session is a utility model. Caller should return early."""
    if _is_utility_model(model_key):
        await push_tok(
            client_id,
            f"!{cmd_name} cannot run from a utility model session ({model_key}). "
            f"Switch to an interactive model (e.g. samaritan-reasoning) to use this command."
        )
        await conditional_push_done(client_id)
        return True
    return False


# ---------------------------------------------------------------------------
# !memreview — AI-assisted topic review with HITL approval
# ---------------------------------------------------------------------------
_pending_reviews: dict[str, list[dict]] = {}          # client_id → topic review proposals
_pending_type_reviews: dict[str, list[dict]] = {}     # client_id → typed-memory review proposals
_pending_classify_reviews: dict[str, list[dict]] = {} # client_id → memory reclassification proposals


async def cmd_memreview(client_id: str, arg: str = "", model_key: str = ""):
    """
    !memreview [types] [approve N,N,...] [reject N,N,...] [clear]

    No args: Calls reviewer model to analyse topic slugs and propose merges/renames.
    types: Review typed memory tables (goals, beliefs, prospective, conditioned, etc.)
           for stale/duplicate/status issues — when memory_types_enabled is active on the model.
    approve N,N,...: Execute approved proposals by number.
    reject N,N,...: Remove proposals from the pending list.
    clear: Discard all pending proposals.
    """
    if await _guard_utility_model(client_id, model_key, "memreview"):
        return
    set_model_context(model_key)
    from database import fetch_dicts, execute_sql
    from memory import _ST, _LT, _COLLECTION

    parts = arg.strip().split(maxsplit=1) if arg.strip() else []
    subcmd = parts[0].lower() if parts else ""

    # ── approve / reject / clear ─────────────────────────────────────────
    if subcmd == "clear":
        _pending_reviews.pop(client_id, None)
        _pending_type_reviews.pop(client_id, None)
        _pending_classify_reviews.pop(client_id, None)
        await push_tok(client_id, "Pending review proposals cleared.")
        await conditional_push_done(client_id)
        return

    if subcmd == "types":
        await cmd_memreview_types(client_id, model_key)
        return

    if subcmd == "classify":
        await cmd_memreview_classify(client_id, model_key)
        return

    if subcmd in ("approve", "reject"):
        # Route to classify approval if only classify proposals are pending
        if not _pending_reviews.get(client_id) and not _pending_type_reviews.get(client_id) \
                and _pending_classify_reviews.get(client_id):
            await _apply_classify_proposals(client_id, subcmd, parts)
            return
        # Route to typed-memory approval if topic proposals are absent but type proposals exist
        if not _pending_reviews.get(client_id) and _pending_type_reviews.get(client_id):
            await _apply_type_proposals(client_id, subcmd, parts)
            return

        proposals = _pending_reviews.get(client_id, [])
        if not proposals:
            await push_tok(client_id, "No pending proposals. Run !memreview or !memreview types first.")
            await conditional_push_done(client_id)
            return

        nums_str = parts[1] if len(parts) > 1 else ""
        try:
            nums = [int(n.strip()) for n in nums_str.split(",") if n.strip()]
        except ValueError:
            await push_tok(client_id, "Usage: !memreview approve 1,2,3 or !memreview reject 1,2")
            await conditional_push_done(client_id)
            return

        if not nums:
            await push_tok(client_id, "Specify proposal numbers: !memreview approve 1,3")
            await conditional_push_done(client_id)
            return

        if subcmd == "reject":
            rejected = []
            for n in sorted(nums, reverse=True):
                if 1 <= n <= len(proposals):
                    p = proposals.pop(n - 1)
                    rejected.append(f"#{n} ({p.get('action', '?')} {p.get('from', '?')})")
            if not proposals:
                _pending_reviews.pop(client_id, None)
            await push_tok(client_id, f"Rejected: {', '.join(rejected) if rejected else 'none matched'}")
            await conditional_push_done(client_id)
            return

        # ── approve: execute proposals ───────────────────────────────
        applied = []
        errors = []
        for n in sorted(nums):
            if n < 1 or n > len(proposals):
                errors.append(f"#{n}: out of range")
                continue
            p = proposals[n - 1]
            action = p.get("action", "")
            old_topic = p.get("from", "")
            new_topic = p.get("to", "")
            if not old_topic or not new_topic:
                errors.append(f"#{n}: invalid proposal")
                continue

            try:
                # Collect affected row IDs before updating
                affected_ids = []
                for table in (_ST(), _LT()):
                    rows = await fetch_dicts(
                        f"SELECT id FROM {table} WHERE topic = '{old_topic}'"
                    )
                    affected_ids.extend(int(r["id"]) for r in rows)

                # Update MySQL (both ST and LT tables)
                for table in (_ST(), _LT()):
                    await execute_sql(
                        f"UPDATE {table} SET topic = '{new_topic}' "
                        f"WHERE topic = '{old_topic}'"
                    )

                # Update Qdrant payload to match
                if affected_ids:
                    try:
                        from plugin_memory_vector_qdrant import get_vector_api
                        vec = get_vector_api()
                        if vec:
                            vec._qc.set_payload(
                                collection_name=_COLLECTION(),
                                payload={"topic": new_topic},
                                points=affected_ids,
                            )
                    except Exception as qe:
                        errors.append(f"#{n}: MySQL OK but Qdrant update failed: {qe}")

                applied.append(f"#{n}: {action} '{old_topic}' → '{new_topic}' ({len(affected_ids)} rows)")
            except Exception as e:
                errors.append(f"#{n}: {e}")

        # Remove applied proposals (reverse order to preserve indices)
        for n in sorted(nums, reverse=True):
            if 1 <= n <= len(proposals):
                proposals.pop(n - 1)
        if not proposals:
            _pending_reviews.pop(client_id, None)

        result_lines = []
        if applied:
            result_lines.append("Applied:\n" + "\n".join(f"  {a}" for a in applied))
        if errors:
            result_lines.append("Errors:\n" + "\n".join(f"  {e}" for e in errors))
        remaining = len(_pending_reviews.get(client_id, []))
        if remaining:
            result_lines.append(f"\n{remaining} proposal(s) still pending.")
        await push_tok(client_id, "\n".join(result_lines) if result_lines else "No valid proposals to apply.")
        await conditional_push_done(client_id)
        return

    # ── show pending proposals if any exist ───────────────────────────────
    if subcmd == "" and client_id in _pending_reviews and _pending_reviews[client_id]:
        proposals = _pending_reviews[client_id]
        lines = [f"**Pending Topic Review ({len(proposals)} proposals)**\n"]
        for i, p in enumerate(proposals, 1):
            reason = p.get("reason", "")
            lines.append(
                f"  {i}. [{p.get('action', '?')}] "
                f"'{p.get('from', '?')}' → '{p.get('to', '?')}'"
                f"  ({p.get('from_count', '?')} rows → {p.get('to_count', '?')} rows)"
            )
            if reason:
                lines.append(f"     Reason: {reason}")
        lines.append("\nUsage: !memreview approve 1,3  |  !memreview reject 2  |  !memreview clear")
        await push_tok(client_id, "\n".join(lines))
        await conditional_push_done(client_id)
        return

    # ── generate new review ──────────────────────────────────────────────
    from memory import _mem_plugin_cfg
    from config import get_model_role
    _review_model = _mem_plugin_cfg().get("reviewer_model") or get_model_role("reviewer")
    await push_tok(client_id, f"Analysing topics with {_review_model}...\n")

    # Gather topic stats + sample content from both tiers
    st_rows = await fetch_dicts(
        f"SELECT topic, content, importance FROM {_ST()} ORDER BY topic, id DESC"
    )
    lt_rows = await fetch_dicts(
        f"SELECT topic, content, importance FROM {_LT()} ORDER BY topic, id DESC"
    )

    # Build topic summary with sample content (max 3 samples per topic per tier)
    topic_data: dict[str, dict] = {}
    for tier_name, rows in [("ST", st_rows), ("LT", lt_rows)]:
        for r in rows:
            t = r.get("topic", "unknown")
            if t not in topic_data:
                topic_data[t] = {"st_count": 0, "lt_count": 0, "samples": []}
            topic_data[t][f"{tier_name.lower()}_count"] = topic_data[t].get(f"{tier_name.lower()}_count", 0) + 1
            # Keep max 3 samples per topic (truncated)
            samples = topic_data[t]["samples"]
            if len([s for s in samples if s.startswith(f"[{tier_name}]")]) < 3:
                content = str(r.get("content", ""))[:200]
                samples.append(f"[{tier_name}] {content}")

    if not topic_data:
        await push_tok(client_id, "No topics found in memory.")
        await conditional_push_done(client_id)
        return

    # Build the review prompt
    topic_lines = []
    for slug, data in sorted(topic_data.items()):
        total = data.get("st_count", 0) + data.get("lt_count", 0)
        topic_lines.append(
            f"\n### {slug} (ST={data.get('st_count', 0)}, LT={data.get('lt_count', 0)}, total={total})"
        )
        for s in data["samples"]:
            topic_lines.append(f"  {s}")

    prompt = (
        "You are a topic hygiene reviewer for a personal AI memory system.\n"
        "Below is a list of all topic slugs with row counts and sample content.\n\n"
        "Analyse the topics and propose:\n"
        "1. **merge**: Two topics that represent the same real-world subject but have different slugs. "
        "The 'to' field should be the more established topic (higher row count or better name).\n"
        "2. **rename**: A single topic with a poor or inconsistent slug name.\n\n"
        "Rules:\n"
        "- Only propose merges when content clearly overlaps (same real-world subject).\n"
        "- Do NOT merge topics that merely share a word prefix (e.g. 'memory-system' and 'memory-roadmap' are distinct).\n"
        "- Do NOT propose changes for topics that are already well-named and distinct.\n"
        "- Prefer shorter, clearer slugs.\n"
        "- If no changes are needed, return an empty proposals list.\n\n"
        "Return ONLY valid JSON (no markdown fences, no commentary):\n"
        '{"proposals": [{"action": "merge"|"rename", "from": "old-slug", "to": "new-slug", "reason": "brief explanation"}]}\n\n'
        "## Topics\n"
        + "\n".join(topic_lines)
    )

    # Call reviewer model
    try:
        from agents import llm_call as _llm_call
        from state import current_client_id

        token = current_client_id.set(client_id)
        try:
            result = await _llm_call(
                model=_review_model,
                prompt=prompt,
                mode="text",
                sys_prompt="none",
                history="none",
            )
        finally:
            current_client_id.reset(token)
    except Exception as e:
        await push_tok(client_id, f"Review failed: {e}")
        await conditional_push_done(client_id)
        return

    # Parse JSON response
    import json as _json
    try:
        # Strip markdown fences if model adds them despite instructions
        raw = result.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = "\n".join(raw.split("\n")[:-1])
        parsed = _json.loads(raw.strip())
        proposals = parsed.get("proposals", [])
    except (_json.JSONDecodeError, AttributeError) as e:
        await push_tok(client_id, f"Failed to parse reviewer response:\n{result[:500]}\n\nError: {e}")
        await conditional_push_done(client_id)
        return

    if not proposals:
        await push_tok(client_id, "Reviewer found no changes needed. Topics look clean.")
        await conditional_push_done(client_id)
        return

    # Enrich proposals with row counts
    for p in proposals:
        from_slug = p.get("from", "")
        to_slug = p.get("to", "")
        from_data = topic_data.get(from_slug, {})
        to_data = topic_data.get(to_slug, {})
        p["from_count"] = from_data.get("st_count", 0) + from_data.get("lt_count", 0)
        p["to_count"] = to_data.get("st_count", 0) + to_data.get("lt_count", 0)

    # Store and display
    _pending_reviews[client_id] = proposals

    lines = [f"**Topic Review ({len(proposals)} proposals)**\n"]
    for i, p in enumerate(proposals, 1):
        lines.append(
            f"  {i}. [{p.get('action', '?')}] "
            f"'{p.get('from', '?')}' → '{p.get('to', '?')}'"
            f"  ({p.get('from_count', '?')} rows → {p.get('to_count', '?')} rows)"
        )
        reason = p.get("reason", "")
        if reason:
            lines.append(f"     Reason: {reason}")
    lines.append("\nUsage: !memreview approve 1,3  |  !memreview reject 2  |  !memreview clear\n"
                 "       !memreview types  — review typed memory tables (goals, beliefs, etc.)")
    await push_tok(client_id, "\n".join(lines))
    await conditional_push_done(client_id)


# ---------------------------------------------------------------------------
# !memreview types — review typed memory tables for staleness / duplicates
# ---------------------------------------------------------------------------

# Tables reviewed and their display label + key columns for the prompt
_TYPE_REVIEW_TABLES = [
    ("goals",          "title",   ["status", "description"]),
    ("beliefs",        "topic",   ["status", "content", "confidence"]),
    ("prospective",    "topic",   ["status", "content", "due_at"]),
    ("conditioned",    "topic",   ["status", "trigger", "reaction", "strength"]),
    ("episodic",       "topic",   ["content"]),
    ("semantic",       "topic",   ["content"]),
    ("procedural",     "topic",   ["content"]),
    ("autobiographical", "topic", ["content"]),
]

# Actions the reviewer can propose for typed memory rows
_TYPE_ACTIONS = {
    "archive":       "Set status to 'done'/'retracted'/'extinguished' (soft-delete)",
    "update_status": "Change status field (e.g. pending→done, active→retracted)",
    "merge":         "Merge duplicate row into another (keeps to_id, removes from_id)",
    "delete":        "Permanently delete row (use only for clearly invalid/test data)",
}


async def cmd_memreview_types(client_id: str, model_key: str = ""):
    """Generate typed-memory review proposals via reviewer model."""
    from database import fetch_dicts, get_tables_for_model
    from memory import _mem_plugin_cfg
    from config import get_model_role

    set_model_context(model_key)

    # Check memory_types_enabled on the model
    mcfg = LLM_REGISTRY.get(model_key, {}) if model_key else {}
    if not mcfg.get("memory_types_enabled", False):
        await push_tok(client_id,
            "memory_types_enabled is not set for this model. "
            "!memreview types is only available when memory_types_enabled: true.")
        await conditional_push_done(client_id)
        return

    tables = get_tables_for_model(model_key)
    _review_model = _mem_plugin_cfg().get("reviewer_model") or get_model_role("reviewer")
    await push_tok(client_id, f"Analysing typed memory tables with {_review_model}...\n")

    # Gather rows from each typed table
    table_sections = []
    for logical_name, title_col, detail_cols in _TYPE_REVIEW_TABLES:
        table_name = tables.get(logical_name)
        if not table_name:
            continue
        try:
            cols = ["id", title_col] + [c for c in detail_cols if c != title_col]
            col_list = ", ".join(cols)
            rows = await fetch_dicts(f"SELECT {col_list} FROM {table_name} LIMIT 100")
        except Exception:
            continue
        if not rows:
            continue

        section_lines = [f"\n### {logical_name} ({len(rows)} rows)"]
        for r in rows:
            row_summary = f"  id={r.get('id')} {title_col}={str(r.get(title_col, ''))[:60]!r}"
            for col in detail_cols:
                val = r.get(col)
                if val is not None:
                    row_summary += f" {col}={str(val)[:40]!r}"
            section_lines.append(row_summary)
        table_sections.append("\n".join(section_lines))

    if not table_sections:
        await push_tok(client_id, "No typed memory rows found.")
        await conditional_push_done(client_id)
        return

    action_descriptions = "\n".join(f"  - {k}: {v}" for k, v in _TYPE_ACTIONS.items())
    prompt = (
        "You are a memory hygiene reviewer for a personal AI typed-memory system.\n"
        "Below are rows from typed memory tables (goals, beliefs, prospective intentions, conditioned responses, etc.).\n\n"
        "Analyse the rows and propose actions for any that are:\n"
        "- Clearly completed or obsolete (goals/plans/prospective)\n"
        "- Contradicted by other rows (beliefs)\n"
        "- Duplicates of another row\n"
        "- Invalid or test data\n\n"
        f"Available actions:\n{action_descriptions}\n\n"
        "Rules:\n"
        "- Only propose changes where there is a clear reason.\n"
        "- Do NOT propose changes for rows that look healthy and current.\n"
        "- For merge: from_id is removed, to_id is kept.\n"
        "- If no changes are needed, return an empty proposals list.\n\n"
        "Return ONLY valid JSON (no markdown fences, no commentary):\n"
        '{"proposals": [{"action": "archive"|"update_status"|"merge"|"delete", '
        '"table": "<table_logical_name>", "from_id": N, "to_id": N_or_null, '
        '"new_status": "value_or_null", "reason": "brief explanation"}]}\n\n'
        "## Typed Memory Tables\n"
        + "\n".join(table_sections)
    )

    try:
        from agents import llm_call as _llm_call
        from state import current_client_id
        token = current_client_id.set(client_id)
        try:
            result = await _llm_call(
                model=_review_model,
                prompt=prompt,
                mode="text",
                sys_prompt="none",
                history="none",
            )
        finally:
            current_client_id.reset(token)
    except Exception as e:
        await push_tok(client_id, f"Type review failed: {e}")
        await conditional_push_done(client_id)
        return

    import json as _json
    try:
        raw = result.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = "\n".join(raw.split("\n")[:-1])
        parsed = _json.loads(raw.strip())
        proposals = parsed.get("proposals", [])
    except (_json.JSONDecodeError, AttributeError) as e:
        await push_tok(client_id, f"Failed to parse reviewer response:\n{result[:500]}\n\nError: {e}")
        await conditional_push_done(client_id)
        return

    if not proposals:
        await push_tok(client_id, "Reviewer found no changes needed. Typed memory looks clean.")
        await conditional_push_done(client_id)
        return

    _pending_type_reviews[client_id] = proposals

    lines = [f"**Typed Memory Review ({len(proposals)} proposals)**\n"]
    for i, p in enumerate(proposals, 1):
        action = p.get("action", "?")
        table = p.get("table", "?")
        from_id = p.get("from_id", "?")
        to_id = p.get("to_id")
        new_status = p.get("new_status")
        reason = p.get("reason", "")
        detail = f"id={from_id}"
        if to_id:
            detail += f" → id={to_id}"
        if new_status:
            detail += f" status→{new_status!r}"
        lines.append(f"  {i}. [{action}] {table} {detail}")
        if reason:
            lines.append(f"     Reason: {reason}")
    lines.append("\nUsage: !memreview approve 1,3  |  !memreview reject 2  |  !memreview clear")
    await push_tok(client_id, "\n".join(lines))
    await conditional_push_done(client_id)


async def _apply_type_proposals(client_id: str, subcmd: str, parts: list[str]):
    """Execute approve/reject against _pending_type_reviews."""
    from database import fetch_dicts, execute_sql, get_tables_for_model

    proposals = _pending_type_reviews.get(client_id, [])
    if not proposals:
        await push_tok(client_id, "No pending type proposals. Run !memreview types first.")
        await conditional_push_done(client_id)
        return

    nums_str = parts[1] if len(parts) > 1 else ""
    try:
        nums = [int(n.strip()) for n in nums_str.split(",") if n.strip()]
    except ValueError:
        await push_tok(client_id, "Usage: !memreview approve 1,2,3 or !memreview reject 1,2")
        await conditional_push_done(client_id)
        return

    if not nums:
        await push_tok(client_id, "Specify proposal numbers: !memreview approve 1,3")
        await conditional_push_done(client_id)
        return

    tables = get_tables_for_model()

    if subcmd == "reject":
        rejected = []
        for n in sorted(nums, reverse=True):
            if 1 <= n <= len(proposals):
                p = proposals.pop(n - 1)
                rejected.append(f"#{n} ({p.get('action','?')} {p.get('table','?')} id={p.get('from_id','?')})")
        if not proposals:
            _pending_type_reviews.pop(client_id, None)
        await push_tok(client_id, f"Rejected: {', '.join(rejected) if rejected else 'none matched'}")
        await conditional_push_done(client_id)
        return

    # ── approve ───────────────────────────────────────────────────────────
    applied = []
    errors = []

    # Status column names per table type (for archive/update_status actions)
    _status_cols = {
        "goals":          ("status", "done"),
        "plans":          ("status", "done"),
        "beliefs":        ("status", "retracted"),
        "prospective":    ("status", "done"),
        "conditioned":    ("status", "extinguished"),
    }

    for n in sorted(nums):
        if n < 1 or n > len(proposals):
            errors.append(f"#{n}: out of range")
            continue
        p = proposals[n - 1]
        action     = p.get("action", "")
        logical    = p.get("table", "")
        from_id    = p.get("from_id")
        to_id      = p.get("to_id")
        new_status = p.get("new_status")
        table_name = tables.get(logical)

        if not table_name:
            errors.append(f"#{n}: unknown table '{logical}'")
            continue
        if not from_id:
            errors.append(f"#{n}: missing from_id")
            continue

        try:
            if action == "delete":
                await execute_sql(f"DELETE FROM {table_name} WHERE id = {int(from_id)}")
                applied.append(f"#{n}: deleted {logical} id={from_id}")

            elif action == "archive":
                status_col, archive_val = _status_cols.get(logical, ("status", "done"))
                await execute_sql(
                    f"UPDATE {table_name} SET {status_col} = '{archive_val}' WHERE id = {int(from_id)}"
                )
                applied.append(f"#{n}: archived {logical} id={from_id} → {status_col}='{archive_val}'")

            elif action == "update_status":
                if not new_status:
                    errors.append(f"#{n}: update_status requires new_status")
                    continue
                status_col = _status_cols.get(logical, ("status", None))[0]
                await execute_sql(
                    f"UPDATE {table_name} SET {status_col} = '{new_status}' WHERE id = {int(from_id)}"
                )
                applied.append(f"#{n}: updated {logical} id={from_id} {status_col}→'{new_status}'")

            elif action == "merge":
                if not to_id:
                    errors.append(f"#{n}: merge requires to_id")
                    continue
                await execute_sql(f"DELETE FROM {table_name} WHERE id = {int(from_id)}")
                applied.append(f"#{n}: merged {logical} id={from_id} into id={to_id} (removed {from_id})")

            else:
                errors.append(f"#{n}: unknown action '{action}'")

        except Exception as e:
            errors.append(f"#{n}: {e}")

    for n in sorted(nums, reverse=True):
        if 1 <= n <= len(proposals):
            proposals.pop(n - 1)
    if not proposals:
        _pending_type_reviews.pop(client_id, None)

    result_lines = []
    if applied:
        result_lines.append("Applied:\n" + "\n".join(f"  {a}" for a in applied))
    if errors:
        result_lines.append("Errors:\n" + "\n".join(f"  {e}" for e in errors))
    remaining = len(_pending_type_reviews.get(client_id, []))
    if remaining:
        result_lines.append(f"\n{remaining} proposal(s) still pending.")
    await push_tok(client_id, "\n".join(result_lines) if result_lines else "No valid proposals to apply.")
    await conditional_push_done(client_id)


# ---------------------------------------------------------------------------
# !memreview classify — reclassify unclassified (type='context') memory rows
# ---------------------------------------------------------------------------

_CLASSIFY_TYPES = [
    "context", "goal", "plan", "belief",
    "episodic", "semantic", "procedural",
    "autobiographical", "prospective", "conditioned",
]
_CLASSIFY_BATCH = 50  # rows per LLM call


async def cmd_memreview_classify(client_id: str, model_key: str = ""):
    """
    Scan ST+LT memory rows with type='context' (default/unclassified) and
    propose type reclassifications via reviewer model with HITL approval.
    """
    from database import fetch_dicts
    from memory import _ST, _LT, _mem_plugin_cfg
    from config import get_model_role

    set_model_context(model_key)

    _review_model = _mem_plugin_cfg().get("reviewer_model") or get_model_role("reviewer")

    # Gather unclassified rows from both tiers
    st_rows = await fetch_dicts(
        f"SELECT id, topic, content, importance, source, 'short' AS tier "
        f"FROM {_ST()} WHERE type = 'context' ORDER BY id DESC LIMIT {_CLASSIFY_BATCH * 2}"
    )
    lt_rows = await fetch_dicts(
        f"SELECT id, topic, content, importance, source, 'long' AS tier "
        f"FROM {_LT()} WHERE type = 'context' ORDER BY id DESC LIMIT {_CLASSIFY_BATCH * 2}"
    )
    all_rows = st_rows + lt_rows

    if not all_rows:
        await push_tok(client_id, "No unclassified memory rows found (all rows already have a type).")
        await conditional_push_done(client_id)
        return

    # Work in batches to stay within context limits
    total_proposals: list[dict] = []
    batch_count = (len(all_rows) + _CLASSIFY_BATCH - 1) // _CLASSIFY_BATCH
    await push_tok(client_id,
        f"Classifying {len(all_rows)} unclassified rows "
        f"({batch_count} batch{'es' if batch_count > 1 else ''}) with {_review_model}...\n")

    types_list = ", ".join(f'"{t}"' for t in _CLASSIFY_TYPES if t != "context")

    for batch_idx in range(batch_count):
        batch = all_rows[batch_idx * _CLASSIFY_BATCH : (batch_idx + 1) * _CLASSIFY_BATCH]
        row_lines = []
        for r in batch:
            content_preview = str(r.get("content", ""))[:200]
            row_lines.append(
                f"  id={r['id']} tier={r['tier']} topic={r.get('topic','')!r} "
                f"importance={r.get('importance',5)} source={r.get('source','?')!r}\n"
                f"    content: {content_preview!r}"
            )

        prompt = (
            "You are a memory classification assistant for a personal AI memory system.\n"
            "Each row below is currently tagged as generic 'context'. Classify each row into "
            "the most specific type that fits. Only reclassify rows where the type is clearly "
            "not generic context — leave ambiguous rows as 'context'.\n\n"
            f"Available types: {types_list}\n\n"
            "Type definitions:\n"
            "  goal: An active objective the system or user wants to achieve\n"
            "  plan: An ordered step or task toward a goal\n"
            "  belief: An asserted world-state fact (e.g. 'user prefers X', 'X is true about the world')\n"
            "  episodic: A specific personal experience, event, or situation that occurred\n"
            "  semantic: General knowledge, facts, concepts (not tied to a specific event)\n"
            "  procedural: A skill, habit, or how-to sequence\n"
            "  autobiographical: Identity-defining information about the system or a specific person\n"
            "  prospective: A future intention, reminder, or planned action\n"
            "  conditioned: A learned trigger→reaction association\n"
            "  context: Generic conversation context (default — use when nothing more specific fits)\n\n"
            "Return ONLY valid JSON, no markdown, no commentary:\n"
            '{"classifications": [{"id": N, "tier": "short"|"long", "type": "<type>", "reason": "brief"}]}\n'
            "Only include rows you are reclassifying (omit rows that should remain 'context').\n\n"
            "## Rows\n" + "\n".join(row_lines)
        )

        try:
            from agents import llm_call as _llm_call
            from state import current_client_id
            token = current_client_id.set(client_id)
            try:
                result = await _llm_call(
                    model=_review_model,
                    prompt=prompt,
                    mode="text",
                    sys_prompt="none",
                    history="none",
                )
            finally:
                current_client_id.reset(token)
        except Exception as e:
            await push_tok(client_id, f"Batch {batch_idx+1} failed: {e}")
            continue

        import json as _json
        try:
            raw = result.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
            if raw.endswith("```"):
                raw = "\n".join(raw.split("\n")[:-1])
            parsed = _json.loads(raw.strip())
            batch_proposals = parsed.get("classifications", [])
            total_proposals.extend(batch_proposals)
        except (_json.JSONDecodeError, AttributeError) as e:
            await push_tok(client_id, f"Batch {batch_idx+1}: failed to parse response: {e}")

    if not total_proposals:
        await push_tok(client_id, "Classifier found no reclassifications needed — all rows look like generic context.")
        await conditional_push_done(client_id)
        return

    # Filter out proposals that suggest keeping 'context' or have invalid types
    valid_proposals = [
        p for p in total_proposals
        if p.get("type") in _CLASSIFY_TYPES and p.get("type") != "context" and p.get("id")
    ]

    if not valid_proposals:
        await push_tok(client_id, "No reclassifications proposed after filtering.")
        await conditional_push_done(client_id)
        return

    _pending_classify_reviews[client_id] = valid_proposals

    lines = [f"**Memory Classification Review ({len(valid_proposals)} proposals)**\n"]
    for i, p in enumerate(valid_proposals, 1):
        tier_label = f"[{p.get('tier','?')}]"
        lines.append(f"  {i}. {tier_label} id={p.get('id','?')} → type={p.get('type','?')!r}")
        reason = p.get("reason", "")
        if reason:
            lines.append(f"     Reason: {reason}")
    lines.append("\nUsage: !memreview approve 1,3  |  !memreview reject 2  |  !memreview clear")
    await push_tok(client_id, "\n".join(lines))
    await conditional_push_done(client_id)


async def _apply_classify_proposals(client_id: str, subcmd: str, parts: list[str]):
    """Execute approve/reject against _pending_classify_reviews."""
    from database import execute_sql
    from memory import _ST, _LT

    proposals = _pending_classify_reviews.get(client_id, [])
    if not proposals:
        await push_tok(client_id, "No pending classification proposals. Run !memreview classify first.")
        await conditional_push_done(client_id)
        return

    nums_str = parts[1] if len(parts) > 1 else ""
    try:
        nums = [int(n.strip()) for n in nums_str.split(",") if n.strip()]
    except ValueError:
        await push_tok(client_id, "Usage: !memreview approve 1,2,3 or !memreview reject 1,2")
        await conditional_push_done(client_id)
        return

    if not nums:
        await push_tok(client_id, "Specify proposal numbers: !memreview approve 1,3")
        await conditional_push_done(client_id)
        return

    if subcmd == "reject":
        rejected = []
        for n in sorted(nums, reverse=True):
            if 1 <= n <= len(proposals):
                p = proposals.pop(n - 1)
                rejected.append(f"#{n} (id={p.get('id','?')} → {p.get('type','?')})")
        if not proposals:
            _pending_classify_reviews.pop(client_id, None)
        await push_tok(client_id, f"Rejected: {', '.join(rejected) if rejected else 'none matched'}")
        await conditional_push_done(client_id)
        return

    # ── approve ───────────────────────────────────────────────────────────
    applied = []
    errors = []

    for n in sorted(nums):
        if n < 1 or n > len(proposals):
            errors.append(f"#{n}: out of range")
            continue
        p = proposals[n - 1]
        row_id  = p.get("id")
        tier    = p.get("tier", "short")
        new_type = p.get("type")

        if not row_id or not new_type or new_type not in _CLASSIFY_TYPES:
            errors.append(f"#{n}: invalid proposal (id={row_id} type={new_type})")
            continue

        table = _ST() if tier == "short" else _LT()
        try:
            await execute_sql(f"UPDATE {table} SET type = '{new_type}' WHERE id = {int(row_id)}")
            applied.append(f"#{n}: [{tier}] id={row_id} → type='{new_type}'")
        except Exception as e:
            errors.append(f"#{n}: {e}")

    for n in sorted(nums, reverse=True):
        if 1 <= n <= len(proposals):
            proposals.pop(n - 1)
    if not proposals:
        _pending_classify_reviews.pop(client_id, None)

    result_lines = []
    if applied:
        result_lines.append("Applied:\n" + "\n".join(f"  {a}" for a in applied))
    if errors:
        result_lines.append("Errors:\n" + "\n".join(f"  {e}" for e in errors))
    remaining = len(_pending_classify_reviews.get(client_id, []))
    if remaining:
        result_lines.append(f"\n{remaining} proposal(s) still pending.")
    await push_tok(client_id, "\n".join(result_lines) if result_lines else "No valid proposals to apply.")
    await conditional_push_done(client_id)


# ---------------------------------------------------------------------------
# !memclassify — Option B: auto-apply high-confidence, HITL for low-confidence
# ---------------------------------------------------------------------------

_AUTO_APPLY_THRESHOLD = 0.80   # confidence ≥ this → apply immediately
_CLASSIFY_MODEL       = "summarizer-gemini"
_CLASSIFY_BIG_BATCH   = 40     # rows per LLM call — keeps each call well under 30s

# Per-client background task state
_classify_tasks: dict[str, asyncio.Task]  = {}  # client_id → running task
_classify_status: dict[str, dict]         = {}  # client_id → progress dict
_classify_hitl: dict[str, list[dict]]     = {}  # client_id → low-confidence proposals


async def cmd_memclassify(client_id: str, arg: str = "", model_key: str = ""):
    """
    !memclassify [status|approve N,N|reject N,N|cancel] [model=<key>]

    No args / 'start': Launch background classification of all type='context' rows.
      - High confidence (≥ 0.80): auto-applied immediately, logged.
      - Low confidence (< 0.80):  queued for HITL approval.
    status:      Show progress of running or completed classification.
    approve N,N: Apply queued low-confidence proposals.
    reject N,N:  Discard queued low-confidence proposals.
    cancel:      Abort a running background classification.

    model=<key>: Override the classification model for this run.
      e.g. !memclassify model=samaritan-reasoning   (higher accuracy, slower)
           !memclassify model=summarizer-gemini      (default, faster)
    batch=N: Rows per LLM call (1–100, default 40). Larger = fewer round-trips.
      e.g. !memclassify model=samaritan-reasoning batch=80
    """
    if await _guard_utility_model(client_id, model_key, "memclassify"):
        return
    from database import fetch_dicts, execute_sql
    from memory import _ST, _LT, _mem_plugin_cfg
    from config import get_model_role

    set_model_context(model_key)

    # Parse optional model=<key> and batch=N overrides anywhere in arg, e.g.:
    #   !memclassify model=samaritan-reasoning batch=150
    #   !memclassify start model=samaritan-reasoning
    import re as _re_arg
    _model_override = None
    _batch_override = None
    _arg_clean = arg.strip()
    _mo = _re_arg.search(r'\bmodel=(\S+)', _arg_clean)
    if _mo:
        _model_override = _mo.group(1)
        _arg_clean = (_arg_clean[:_mo.start()] + _arg_clean[_mo.end():]).strip()
    _bo = _re_arg.search(r'\bbatch=(\d+)', _arg_clean)
    if _bo:
        _batch_override = max(1, min(int(_bo.group(1)), 100))
        _arg_clean = (_arg_clean[:_bo.start()] + _arg_clean[_bo.end():]).strip()
    _classify_model = _model_override or _CLASSIFY_MODEL
    _classify_batch = _batch_override or _CLASSIFY_BIG_BATCH

    parts = _arg_clean.split(maxsplit=1) if _arg_clean else []
    subcmd = parts[0].lower() if parts else "start"

    # ── status ───────────────────────────────────────────────────────────
    if subcmd == "status":
        st = _classify_status.get(client_id)
        if not st:
            await push_tok(client_id, "No classification job for this session. Run !memclassify to start.")
            await conditional_push_done(client_id)
            return
        running = client_id in _classify_tasks and not _classify_tasks[client_id].done()
        state   = "running" if running else ("failed" if st.get("error") else "complete")
        lines = [
            f"**Classification {state}**",
            f"  Total unclassified: {st.get('total', '?')}",
            f"  Processed: {st.get('processed', 0)}",
            f"  Auto-applied: {st.get('auto_applied', 0)}",
            f"  Pending HITL: {len(_classify_hitl.get(client_id, []))}",
            f"  Kept as context: {st.get('kept_context', 0)}",
        ]
        if st.get("error"):
            lines.append(f"  Error: {st['error']}")
        if running:
            lines.append(f"\n  Batch {st.get('batch', 0)}/{st.get('total_batches', '?')} in progress...")
        elif _classify_hitl.get(client_id):
            lines.append(f"\n{len(_classify_hitl[client_id])} low-confidence rows need review:")
            for i, p in enumerate(_classify_hitl[client_id], 1):
                lines.append(
                    f"  {i}. [{p.get('tier','?')}] id={p.get('id','?')} "
                    f"→ {p.get('type','?')!r}  conf={p.get('confidence',0):.2f}"
                )
                if p.get("reason"):
                    lines.append(f"     {p['reason']}")
            lines.append("\nUsage: !memclassify approve 1,3  |  !memclassify reject 2")
        await push_tok(client_id, "\n".join(lines))
        await conditional_push_done(client_id)
        return

    # ── cancel ────────────────────────────────────────────────────────────
    if subcmd == "cancel":
        task = _classify_tasks.get(client_id)
        if task and not task.done():
            task.cancel()
            _classify_tasks.pop(client_id, None)
            await push_tok(client_id, "Classification job cancelled.")
        else:
            await push_tok(client_id, "No running classification job to cancel.")
        await conditional_push_done(client_id)
        return

    # ── approve / reject (HITL queue) ────────────────────────────────────
    if subcmd in ("approve", "reject"):
        proposals = _classify_hitl.get(client_id, [])
        if not proposals:
            await push_tok(client_id, "No pending low-confidence proposals. Run !memclassify status.")
            await conditional_push_done(client_id)
            return

        nums_str = parts[1] if len(parts) > 1 else ""
        try:
            nums = [int(n.strip()) for n in nums_str.split(",") if n.strip()]
        except ValueError:
            await push_tok(client_id, "Usage: !memclassify approve 1,2,3 or !memclassify reject 1,2")
            await conditional_push_done(client_id)
            return
        if not nums:
            await push_tok(client_id, "Specify proposal numbers: !memclassify approve 1,3")
            await conditional_push_done(client_id)
            return

        if subcmd == "reject":
            rejected = []
            for n in sorted(nums, reverse=True):
                if 1 <= n <= len(proposals):
                    p = proposals.pop(n - 1)
                    rejected.append(f"#{n} (id={p.get('id','?')} {p.get('type','?')})")
            if not proposals:
                _classify_hitl.pop(client_id, None)
            await push_tok(client_id, f"Rejected: {', '.join(rejected) if rejected else 'none matched'}")
            await conditional_push_done(client_id)
            return

        applied, errors = [], []
        for n in sorted(nums):
            if n < 1 or n > len(proposals):
                errors.append(f"#{n}: out of range")
                continue
            p = proposals[n - 1]
            table = _ST() if p.get("tier") == "short" else _LT()
            try:
                await execute_sql(
                    f"UPDATE {table} SET type = '{p['type']}' WHERE id = {int(p['id'])}"
                )
                applied.append(f"#{n}: [{p.get('tier','?')}] id={p['id']} → '{p['type']}'")
            except Exception as e:
                errors.append(f"#{n}: {e}")

        for n in sorted(nums, reverse=True):
            if 1 <= n <= len(proposals):
                proposals.pop(n - 1)
        if not proposals:
            _classify_hitl.pop(client_id, None)

        result_lines = []
        if applied:
            result_lines.append("Applied:\n" + "\n".join(f"  {a}" for a in applied))
        if errors:
            result_lines.append("Errors:\n" + "\n".join(f"  {e}" for e in errors))
        remaining = len(_classify_hitl.get(client_id, []))
        if remaining:
            result_lines.append(f"\n{remaining} proposal(s) still pending.")
        await push_tok(client_id, "\n".join(result_lines) if result_lines else "No valid proposals.")
        await conditional_push_done(client_id)
        return

    # ── start (or re-start) ───────────────────────────────────────────────
    # Block if the classify model is the same as the session model — calling
    # llm_call with the session's own model would be a self-referential call.
    if model_key and _classify_model == model_key:
        await push_tok(
            client_id,
            f"!memclassify cannot use model={_classify_model!r} because that is the current session model. "
            f"Specify a different model, e.g. !memclassify model={_CLASSIFY_MODEL}"
        )
        await conditional_push_done(client_id)
        return

    task = _classify_tasks.get(client_id)
    if task and not task.done():
        await push_tok(client_id,
            "Classification already running. Use !memclassify status to check progress "
            "or !memclassify cancel to abort.")
        await conditional_push_done(client_id)
        return

    # Count unclassified rows
    st_count = await fetch_dicts(f"SELECT COUNT(*) AS n FROM {_ST()} WHERE type = 'context'")
    lt_count = await fetch_dicts(f"SELECT COUNT(*) AS n FROM {_LT()} WHERE type = 'context'")
    total = (st_count[0]["n"] if st_count else 0) + (lt_count[0]["n"] if lt_count else 0)

    if total == 0:
        await push_tok(client_id, "No unclassified rows found — all rows already have a specific type.")
        await conditional_push_done(client_id)
        return

    total_batches = (total + _classify_batch - 1) // _classify_batch
    _classify_status[client_id] = {
        "total": total,
        "total_batches": total_batches,
        "processed": 0,
        "auto_applied": 0,
        "kept_context": 0,
        "batch": 0,
    }
    _classify_hitl[client_id] = []

    # Snapshot model_key, classify model, and batch size for the background task
    _model_key = model_key
    _classify_model_snap = _classify_model
    _classify_batch_snap = _classify_batch

    async def _run_classification():
        from database import fetch_dicts as _fetch, execute_sql as _exec
        from memory import _ST as ST, _LT as LT
        from agents import llm_call as _llm_call
        from state import current_client_id
        import json as _json

        st_state = _classify_status[client_id]
        set_model_context(_model_key)

        types_list = ", ".join(f'"{t}"' for t in _CLASSIFY_TYPES if t != "context")
        type_defs = (
            "  goal: An active objective to achieve\n"
            "  plan: An ordered step or task toward a goal\n"
            "  belief: An asserted world-state fact (e.g. 'user prefers X')\n"
            "  episodic: A specific personal experience, event, or situation that occurred\n"
            "  semantic: General knowledge, facts, or concepts (not tied to a specific event)\n"
            "  procedural: A skill, habit, or how-to sequence\n"
            "  autobiographical: Identity-defining information about the system or a person\n"
            "  prospective: A future intention, reminder, or planned action\n"
            "  conditioned: A learned trigger→reaction association\n"
            "  context: Generic conversation context (use when nothing more specific fits)"
        )

        offset = 0
        batch_idx = 0

        try:
            while True:
                set_model_context(_model_key)
                # Fetch next batch across both tiers (ST first, then LT)
                batch_rows = await _fetch(
                    f"SELECT id, topic, content, importance, source, 'short' AS tier "
                    f"FROM {ST()} WHERE type = 'context' "
                    f"ORDER BY id ASC LIMIT {_classify_batch_snap}"
                )
                remaining_slots = _classify_batch_snap - len(batch_rows)
                if remaining_slots > 0:
                    lt_rows = await _fetch(
                        f"SELECT id, topic, content, importance, source, 'long' AS tier "
                        f"FROM {LT()} WHERE type = 'context' "
                        f"ORDER BY id ASC LIMIT {remaining_slots}"
                    )
                    batch_rows.extend(lt_rows)

                if not batch_rows:
                    break

                batch_idx += 1
                st_state["batch"] = batch_idx

                row_lines = []
                for r in batch_rows:
                    content_preview = str(r.get("content", ""))[:300]
                    row_lines.append(
                        f"id={r['id']} tier={r['tier']} topic={r.get('topic','')!r} "
                        f"imp={r.get('importance',5)}\n  {content_preview!r}"
                    )

                prompt = (
                    "You are a memory classification engine. Classify each memory row into "
                    "the most specific type. For each row return a confidence score (0.0–1.0) "
                    "reflecting how certain you are. Only deviate from 'context' when the type "
                    "is clearly more specific — ambiguous rows should remain 'context' with "
                    "high confidence.\n\n"
                    f"Available types: {types_list}, context\n\n"
                    f"Type definitions:\n{type_defs}\n\n"
                    "Return ONLY valid JSON, no markdown:\n"
                    '{"results": [{"id": N, "tier": "short"|"long", "type": "<type>", '
                    '"confidence": 0.0-1.0, "reason": "brief"}]}\n'
                    "Include ALL rows — even those staying as 'context'.\n\n"
                    "## Rows\n" + "\n".join(row_lines)
                )

                try:
                    token = current_client_id.set(client_id)
                    try:
                        result = await _llm_call(
                            model=_classify_model_snap,
                            prompt=prompt,
                            mode="text",
                            sys_prompt="none",
                            history="none",
                        )
                    finally:
                        current_client_id.reset(token)
                except Exception as e:
                    st_state["error"] = f"LLM call failed batch {batch_idx}: {e}"
                    break

                # llm_call returns "ERROR: ..." strings on timeout/failure instead of raising
                if result.startswith("ERROR:"):
                    st_state["error"] = f"Batch {batch_idx}: {result}"
                    break

                try:
                    # Extract first {...} block — robust against markdown fences or preamble
                    import re as _re
                    m = _re.search(r'\{.*\}', result, _re.DOTALL)
                    if not m:
                        raise ValueError("no JSON object found in response")
                    parsed = _json.loads(m.group(0))
                    classifications = parsed.get("results", [])
                except (ValueError, _json.JSONDecodeError, AttributeError) as e:
                    st_state["error"] = f"Parse failed batch {batch_idx}: {e!r} — raw: {result[:200]!r}"
                    break

                auto_applied = 0
                kept_context = 0
                for c in classifications:
                    row_id    = c.get("id")
                    tier      = c.get("tier", "short")
                    new_type  = c.get("type", "context")
                    conf      = float(c.get("confidence", 0.0))

                    if not row_id or new_type not in _CLASSIFY_TYPES:
                        continue

                    if new_type == "context":
                        kept_context += 1
                        continue

                    table = ST() if tier == "short" else LT()

                    if conf >= _AUTO_APPLY_THRESHOLD:
                        try:
                            await _exec(
                                f"UPDATE {table} SET type = '{new_type}' WHERE id = {int(row_id)}"
                            )
                            auto_applied += 1
                        except Exception as _ue:
                            st_state.setdefault("db_errors", []).append(str(_ue))
                            if len(st_state["db_errors"]) == 1:
                                # Surface first error immediately so it shows in status
                                st_state["error"] = f"DB update failed: {_ue}"
                    else:
                        # Queue for HITL
                        _classify_hitl[client_id].append({
                            "id":         row_id,
                            "tier":       tier,
                            "type":       new_type,
                            "confidence": conf,
                            "reason":     c.get("reason", ""),
                        })

                st_state["processed"]   += len(batch_rows)
                st_state["auto_applied"] += auto_applied
                st_state["kept_context"] += kept_context

                # If we got fewer rows than the batch size, we're done
                if len(batch_rows) < _classify_batch_snap:
                    break

                # Brief pause between batches to avoid rate-limiting
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            st_state["error"] = "Cancelled"
            raise
        except Exception as e:
            st_state["error"] = str(e)

    task = asyncio.create_task(_run_classification())
    _classify_tasks[client_id] = task

    await push_tok(client_id,
        f"Classification started in background: {total} unclassified rows, "
        f"{total_batches} batch{'es' if total_batches > 1 else ''} of {_classify_batch}, "
        f"model={_classify_model}, auto-apply threshold={_AUTO_APPLY_THRESHOLD}.\n"
        f"Use !memclassify status to check progress and review low-confidence proposals."
    )
    await conditional_push_done(client_id)


async def cmd_memage(client_id: str, model_key: str = ""):
    """
    !memage — manually trigger one full aging pass (count-pressure + staleness).
    Uses the same logic as the background tasks but runs immediately.
    """
    from memory import age_by_count, age_by_minutes, _age_cfg, _st_count
    set_model_context(model_key)

    cfg     = _age_cfg()
    before  = await _st_count()
    await push_tok(client_id, f"ST rows before: {before} (HWM={cfg['short_hwm']}, LWM={cfg['short_lwm']})\n")
    await push_tok(client_id, "Running count-pressure aging pass...")

    deleted_count = await age_by_count()
    mid = await _st_count()
    await push_tok(client_id, f" done ({deleted_count} rows deleted, ST now={mid})\n")

    await push_tok(client_id, "Running staleness aging pass...")
    deleted_min = await age_by_minutes(trigger_minutes=cfg["memory_age_trigger_minutes"])
    after = await _st_count()
    await push_tok(
        client_id,
        f" done ({deleted_min} rows deleted, ST now={after})\n"
        f"Total deleted: {deleted_count + deleted_min} rows.\n"
    )
    await conditional_push_done(client_id)


async def cmd_memtrim(client_id: str, arg: str, model_key: str = ""):
    """
    !memtrim [N] — hard-delete N oldest/least-important ST rows without summarizing.
    If N is omitted, trims to short_lwm.
    Escape valve for when topic-chunk aging can't make progress.
    """
    set_model_context(model_key)
    from memory import trim_st_to_lwm, _st_count, _age_cfg, _COLLECTION
    from database import execute_sql, fetch_dicts
    from memory import _ST

    cfg    = _age_cfg()
    before = await _st_count()

    if arg.strip().isdigit():
        n = int(arg.strip())
        await push_tok(client_id, f"Hard-trimming {n} oldest ST rows (ST before={before})...\n")
        # Fetch target rows and delete them
        try:
            from plugin_memory_vector_qdrant import get_vector_api
            vec = get_vector_api()
        except Exception:
            vec = None
        try:
            import asyncio
            rows = await fetch_dicts(
                f"SELECT id FROM {_ST()} ORDER BY importance ASC, last_accessed ASC LIMIT {n}"
            )
            deleted = 0
            for r in rows:
                rid = r.get("id")
                if rid:
                    await execute_sql(f"DELETE FROM {_ST()} WHERE id = {int(rid)}")
                    deleted += 1
                    if vec:
                        asyncio.create_task(vec.delete_memory(int(rid), collection=_COLLECTION()))
            after = await _st_count()
            await push_tok(client_id, f"Deleted {deleted} rows. ST now={after}.\n")
        except Exception as e:
            await push_tok(client_id, f"Error during trim: {e}\n")
    else:
        await push_tok(client_id, f"Trimming ST to LWM={cfg['short_lwm']} (ST before={before})...\n")
        deleted = await trim_st_to_lwm()
        after   = await _st_count()
        await push_tok(client_id, f"Deleted {deleted} rows. ST now={after}.\n")

    await conditional_push_done(client_id)


# ---------------------------------------------------------------------------
# !cogn — proactive cognition timer control + stats
# ---------------------------------------------------------------------------

async def cmd_cogn(client_id: str, arg: str, model_key: str = ""):
    """
    !cogn — proactive cognition control panel.

    Subcommands:
      !cogn                          show status of all timers + stats
      !cogn on / off                 master enable/disable (runtime)
      !cogn contradiction on|off|run scanner toggle / trigger now
      !cogn prospective on|off|run   prospective loop toggle / trigger now
      !cogn reflection on|off|run    reflection loop toggle / trigger now
      !cogn interval contradiction <h>   set scan interval (hours, float)
      !cogn interval prospective <m>     set check interval (minutes, int)
      !cogn interval reflection <h>      set reflection interval (hours, float)
      !cogn model <loop> <key>       set model for a loop (contradiction|prospective|reflection)
      !cogn flags [clear]            view/retract open contradiction-flag beliefs
      !cogn feedback reset <loop>    reset feedback streak/strength for a loop
      !cogn reset                    clear all runtime overrides (revert to json)
    """
    set_model_context(model_key)

    try:
        from contradiction import (
            get_contradiction_stats, get_runtime_overrides,
            set_runtime_override, clear_runtime_overrides,
            _cogn_cfg,
        )
        from memory import _BELIEFS
    except ImportError as e:
        await push_tok(client_id, f"contradiction module not available: {e}")
        await conditional_push_done(client_id)
        return

    parts = arg.strip().split() if arg.strip() else []

    # ── !cogn flags [clear] ────────────────────────────────────────────────
    if parts and parts[0] == "flags":
        if len(parts) > 1 and parts[1] == "clear":
            try:
                tbl = _BELIEFS()
                await execute_sql(
                    f"UPDATE {tbl} SET status='retracted' "
                    f"WHERE topic='contradiction-flag' AND status='active'"
                )
                await push_tok(client_id, "All open contradiction-flag beliefs retracted.\n")
            except Exception as e:
                await push_tok(client_id, f"Error clearing flags: {e}\n")
        else:
            try:
                from database import fetch_dicts
                rows = await fetch_dicts(
                    f"SELECT id, content, confidence, created_at "
                    f"FROM {_BELIEFS()} "
                    f"WHERE topic='contradiction-flag' AND status='active' "
                    f"ORDER BY created_at DESC"
                )
                if not rows:
                    await push_tok(client_id, "No open contradiction-flag beliefs.\n")
                else:
                    flines = [f"## Open Contradiction Flags ({len(rows)})\n"]
                    for r in rows:
                        flines.append(
                            f"  [id={r['id']}] {r.get('content','')[:160]}  "
                            f"(created {r.get('created_at','')})"
                        )
                    await push_tok(client_id, "\n".join(flines) + "\n")
            except Exception as e:
                await push_tok(client_id, f"Error fetching flags: {e}\n")
        await conditional_push_done(client_id)
        return

    # ── !cogn feedback reset <loop> ───────────────────────────────────────
    if parts and parts[0] == "feedback":
        if len(parts) > 1 and parts[1] == "reset":
            loop = parts[2] if len(parts) > 2 else ""
            valid = ("contradiction", "prospective", "reflection")
            if loop not in valid:
                await push_tok(client_id, "Usage: !cogn feedback reset contradiction|prospective|reflection\n")
            else:
                try:
                    from cogn_feedback import reset_feedback_state
                    reset_feedback_state(loop)
                    await push_tok(client_id, f"Feedback state reset for {loop}. Streak and strength cleared.\n")
                except Exception as e:
                    await push_tok(client_id, f"Error: {e}\n")
        else:
            await push_tok(client_id, "Usage: !cogn feedback reset contradiction|prospective|reflection\n")
        await conditional_push_done(client_id)
        return

    # ── !cogn reset ────────────────────────────────────────────────────────
    if parts and parts[0] == "reset":
        clear_runtime_overrides()
        await push_tok(client_id, "Runtime overrides cleared. Config reverts to plugins-enabled.json.\n")
        await conditional_push_done(client_id)
        return

    # ── !cogn on / off ─────────────────────────────────────────────────────
    if parts and parts[0] == "on":
        set_runtime_override("enabled", True)
        await push_tok(client_id, "Proactive cognition master switch: ON (runtime)\n")
        await conditional_push_done(client_id)
        return
    if parts and parts[0] == "off":
        set_runtime_override("enabled", False)
        await push_tok(client_id, "Proactive cognition master switch: OFF (runtime)\n")
        await conditional_push_done(client_id)
        return

    # ── !cogn <loop> on|off|run ────────────────────────────────────────────
    _loop_map = {
        "contradiction": {
            "flag":     "contradiction_enabled",
            "run_fn":   lambda: __import__("contradiction").run_scan,
            "trig_fn":  lambda: __import__("contradiction").trigger_now,
            "label":    "Contradiction scanner",
        },
        "prospective": {
            "flag":     "prospective_enabled",
            "run_fn":   lambda: __import__("prospective").run_check,
            "trig_fn":  lambda: __import__("prospective").trigger_now,
            "label":    "Prospective memory loop",
        },
        "reflection": {
            "flag":     "reflection_enabled",
            "run_fn":   lambda: __import__("reflection").run_reflection,
            "trig_fn":  lambda: __import__("reflection").trigger_now,
            "label":    "Reflection loop",
        },
    }
    if parts and parts[0] in _loop_map:
        loop_name = parts[0]
        lm = _loop_map[loop_name]
        sub = parts[1] if len(parts) > 1 else ""
        if sub == "on":
            set_runtime_override(lm["flag"], True)
            await push_tok(client_id, f"{lm['label']}: ON (runtime)\n")
        elif sub == "off":
            set_runtime_override(lm["flag"], False)
            await push_tok(client_id, f"{lm['label']}: OFF (runtime)\n")
        elif sub == "run":
            await push_tok(client_id, f"Running {lm['label']} now...\n")
            try:
                lm["trig_fn"]()()
                run_fn = lm["run_fn"]()
                summary = await run_fn()
                err = summary.get("error")
                skip = summary.get("skipped_reason") or summary.get("skipped")
                if skip:
                    await push_tok(client_id, f"Skipped: {skip}\n")
                elif err:
                    await push_tok(client_id, f"Error: {err}\n")
                else:
                    parts_out = []
                    for k, v in summary.items():
                        if k != "error" and not k.startswith("skipped"):
                            parts_out.append(f"{k}={v}")
                    await push_tok(client_id, f"Done — {', '.join(parts_out)}\n")
            except Exception as e:
                await push_tok(client_id, f"Failed: {e}\n")
        else:
            await push_tok(client_id, f"Usage: !cogn {loop_name} on|off|run\n")
        await conditional_push_done(client_id)
        return

    # ── !cogn interval <loop> <value> ─────────────────────────────────────
    if parts and parts[0] == "interval":
        _interval_keys = {
            "contradiction": ("contradiction_interval_h", "hours", float),
            "prospective":   ("prospective_interval_m",   "minutes", int),
            "reflection":    ("reflection_interval_h",    "hours", float),
        }
        if len(parts) < 3 or parts[1] not in _interval_keys:
            await push_tok(client_id,
                "Usage: !cogn interval contradiction <h>  |  "
                "!cogn interval prospective <m>  |  !cogn interval reflection <h>\n")
            await conditional_push_done(client_id)
            return
        key, unit, cast = _interval_keys[parts[1]]
        try:
            val = cast(parts[2])
            if val < 0:
                raise ValueError("must be >= 0")
            set_runtime_override(key, val)
            await push_tok(client_id, f"{parts[1]} interval set to {val} {unit} (runtime)\n")
        except ValueError as e:
            await push_tok(client_id, f"Invalid value: {e}\n")
        await conditional_push_done(client_id)
        return

    # ── !cogn model <loop> <key> ───────────────────────────────────────────
    if parts and parts[0] == "model":
        _model_keys = {
            "contradiction": "contradiction_model",
            "prospective":   "prospective_model",
            "reflection":    "reflection_model",
        }
        if len(parts) < 3 or parts[1] not in _model_keys:
            await push_tok(client_id,
                "Usage: !cogn model contradiction|prospective|reflection <model_key>\n")
            await conditional_push_done(client_id)
            return
        set_runtime_override(_model_keys[parts[1]], parts[2])
        await push_tok(client_id, f"{parts[1]} model set to {parts[2]!r} (runtime)\n")
        await conditional_push_done(client_id)
        return

    # ── !cogn (status dashboard) ──────────────────────────────────────────
    cfg = _cogn_cfg()
    ovr = get_runtime_overrides()

    def _bool(v: bool) -> str:
        return "ON" if v else "OFF"

    def _timer_h(h: float) -> str:
        return f"{h}h" if h > 0 else "disabled"

    def _timer_m(m: int) -> str:
        return f"{m}m" if m > 0 else "disabled"

    def _ovr(key: str) -> str:
        return " [runtime]" if key in ovr else ""

    master = cfg["enabled"]
    lines = ["## Proactive Cognition Status\n"]

    lines.append("**Timers**")
    lines.append(f"  {'master':<30}: {_bool(master)}{_ovr('enabled')}")

    suffix = "" if master else "  (master OFF)"
    # Contradiction
    cscan = cfg["contradiction_enabled"]
    lines.append(
        f"  {'contradiction':<30}: {_bool(master and cscan)}{_ovr('contradiction_enabled')}  "
        f"every {_timer_h(cfg['contradiction_interval_h'])}{_ovr('contradiction_interval_h')}  "
        f"model={cfg['contradiction_model']}{_ovr('contradiction_model')}{suffix}"
    )
    # Prospective
    pscan = cfg.get("prospective_enabled", True)
    lines.append(
        f"  {'prospective':<30}: {_bool(master and pscan)}{_ovr('prospective_enabled')}  "
        f"every {_timer_m(cfg.get('prospective_interval_m', 5))}{_ovr('prospective_interval_m')}  "
        f"model={cfg.get('prospective_model','?')}{_ovr('prospective_model')}{suffix}"
    )
    # Reflection
    rscan = cfg.get("reflection_enabled", True)
    lines.append(
        f"  {'reflection':<30}: {_bool(master and rscan)}{_ovr('reflection_enabled')}  "
        f"every {_timer_h(cfg.get('reflection_interval_h', 6))}{_ovr('reflection_interval_h')}  "
        f"model={cfg.get('reflection_model','?')}{_ovr('reflection_model')}{suffix}"
    )
    lines.append("")

    # Feedback state helper
    def _fb_line(fb: dict | None) -> str:
        if not fb:
            return "  feedback: (not yet evaluated)"
        v = fb.get("verdict", "?")
        r = fb.get("ratio")
        s = fb.get("strength", 0)
        streak = fb.get("streak", 0)
        ratio_s = f"{r:.2f}" if r is not None else "n/a"
        _verdict_icon = {
            "useful": "✓", "neutral": "~", "low": "↓",
            "throttle": "⚠ THROTTLED", "extinguish": "✗ EXTINGUISHED",
            "insufficient_data": "?",
        }
        icon = _verdict_icon.get(v, v)
        return f"  feedback: {icon}  ratio={ratio_s}  strength={s}/10  streak={streak}"

    # Contradiction stats
    try:
        from contradiction import get_contradiction_stats
        cs = get_contradiction_stats()
        lines.append("**Contradiction Scanner** (since restart)")
        lines.append(
            f"  scans={cs['scans_run']}  pairs={cs['pairs_evaluated']}  "
            f"found={cs['contradictions_found']}  flags={cs['flags_written']}  "
            f"auto_retracted={cs['auto_retracted']}"
        )
        lines.append(
            f"  last={cs['last_scan_at'] or 'never'}  "
            f"dur={cs['last_scan_duration_s']}s  "
            f"last_pairs={cs['last_scan_pairs']}  last_flags={cs['last_scan_flags']}"
        )
        lines.append(_fb_line(cs.get("last_feedback")))
        if cs["last_error"]:
            lines.append(f"  last_error={cs['last_error']}")
    except ImportError:
        pass
    lines.append("")

    # Prospective stats
    try:
        from prospective import get_prospective_stats
        ps = get_prospective_stats()
        lines.append("**Prospective Memory Loop** (since restart)")
        lines.append(
            f"  checks={ps['checks_run']}  evaluated={ps['rows_evaluated']}  "
            f"fired={ps['rows_fired']}  reminders={ps['reminders_written']}"
        )
        lines.append(f"  last={ps['last_check_at'] or 'never'}")
        lines.append(_fb_line(ps.get("last_feedback")))
        if ps["last_error"]:
            lines.append(f"  last_error={ps['last_error']}")
    except ImportError:
        pass
    lines.append("")

    # Reflection stats
    try:
        from reflection import get_reflection_stats
        rs = get_reflection_stats()
        lines.append("**Reflection Loop** (since restart)")
        lines.append(
            f"  runs={rs['runs']}  turns_processed={rs['turns_processed']}  "
            f"memories_saved={rs['memories_saved']}  skipped={rs['memories_skipped']}"
        )
        lines.append(
            f"  last={rs['last_run_at'] or 'never'}  "
            f"dur={rs['last_run_duration_s']}s  last_saved={rs['last_run_saved']}"
        )
        lines.append(_fb_line(rs.get("last_feedback")))
        if rs["last_error"]:
            lines.append(f"  last_error={rs['last_error']}")
    except ImportError:
        pass
    lines.append("")

    # Feedback config
    try:
        from cogn_feedback import get_feedback_state, _fb_cfg
        fb_cfg = _fb_cfg()
        fb_state = get_feedback_state()
        lines.append("**Feedback Evaluator Config**")
        lines.append(
            f"  throttle_at_strength={fb_cfg['feedback_strength_throttle']}  "
            f"extinguish_at_strength={fb_cfg['feedback_strength_extinguish']}  "
            f"low_ratio<{fb_cfg['feedback_low_ratio']}  "
            f"high_ratio>={fb_cfg['feedback_high_ratio']}  "
            f"min_rows={fb_cfg['feedback_min_rows']}"
        )
        for lname, st in fb_state.items():
            cid = st.get("conditioned_id", 0)
            lines.append(
                f"  {lname:<14} conditioned_id={cid}  "
                f"streak={st.get('consecutive_low',0)}  "
                f"strength={st.get('current_strength',0)}  "
                f"last_ratio={st.get('last_ratio','n/a')}"
            )
    except ImportError:
        pass
    lines.append("")

    # Open contradiction flags
    try:
        open_flags_raw = await execute_sql(
            f"SELECT COUNT(*) FROM {_BELIEFS()} "
            f"WHERE topic='contradiction-flag' AND status='active'"
        )
        open_flags = 0
        for ln in open_flags_raw.strip().splitlines():
            ln = ln.strip()
            if ln.isdigit():
                open_flags = int(ln)
                break
            lp = ln.split()
            if lp and lp[-1].isdigit():
                open_flags = int(lp[-1])
                break
        lines.append(f"**Open contradiction flags**: {open_flags}"
                     + ("  (!cogn flags to view  |  !cogn flags clear)" if open_flags else ""))
    except Exception:
        pass

    if ovr:
        lines.append("")
        lines.append(f"**Active runtime overrides**: {list(ovr.keys())}  (!cogn reset to revert)")

    lines.append("")
    lines.append(
        "**Commands**:\n"
        "  !cogn on|off\n"
        "  !cogn contradiction|prospective|reflection on|off|run\n"
        "  !cogn interval contradiction|prospective|reflection <value>\n"
        "  !cogn model contradiction|prospective|reflection <key>\n"
        "  !cogn flags [clear]  |  !cogn reset"
    )

    await push_tok(client_id, "\n".join(lines))
    await conditional_push_done(client_id)


# ---------------------------------------------------------------------------
# !drives — affect/motivational drive control
# ---------------------------------------------------------------------------

async def cmd_drives(client_id: str, arg: str, model_key: str = ""):
    """
    !drives                            — show current drive values
    !drives set <name> <val>           — set drive value (0.0-1.0)
    !drives set <name> <val> <desc...> — set drive value and description
    !drives baseline <name> <val>      — set baseline equilibrium
    !drives decay                      — manually trigger one decay cycle
    !drives seed                       — seed default drives from config
    """
    set_model_context(model_key)
    from memory import load_drives, update_drive, decay_drives, _DRIVES
    from database import execute_sql, fetch_dicts, execute_insert

    parts = arg.strip().split() if arg.strip() else []

    # ── !drives set <name> <val> [description...] ──────────────────────────
    if parts and parts[0] == "set":
        if len(parts) < 3:
            await push_tok(client_id, "Usage: !drives set <name> <0.0-1.0> [description]\n")
            await conditional_push_done(client_id)
            return
        name = parts[1].lower()
        try:
            val = float(parts[2])
        except ValueError:
            await push_tok(client_id, f"Invalid value: {parts[2]} (must be 0.0-1.0)\n")
            await conditional_push_done(client_id)
            return
        desc = " ".join(parts[3:]) if len(parts) > 3 else None
        ok = await update_drive(name, val, source="user")
        if ok and desc is not None:
            tbl = _DRIVES()
            desc_esc = desc.replace("'", "''")
            try:
                await execute_sql(
                    f"UPDATE {tbl} SET description='{desc_esc}' WHERE name='{name}'"
                )
            except Exception as e:
                await push_tok(client_id, f"Value set but description update failed: {e}\n")
                await conditional_push_done(client_id)
                return
        await push_tok(client_id, f"Drive '{name}' set to {val:.2f}\n" if ok
                       else f"Failed to update drive '{name}'\n")
        await conditional_push_done(client_id)
        return

    # ── !drives baseline <name> <val> ──────────────────────────────────────
    if parts and parts[0] == "baseline":
        if len(parts) < 3:
            await push_tok(client_id, "Usage: !drives baseline <name> <0.0-1.0>\n")
            await conditional_push_done(client_id)
            return
        name = parts[1].lower()
        try:
            val = max(0.0, min(1.0, float(parts[2])))
        except ValueError:
            await push_tok(client_id, f"Invalid value: {parts[2]}\n")
            await conditional_push_done(client_id)
            return
        tbl = _DRIVES()
        try:
            await execute_sql(
                f"UPDATE {tbl} SET baseline={val:.4f} WHERE name='{name}'"
            )
            await push_tok(client_id, f"Drive '{name}' baseline set to {val:.2f}\n")
        except Exception as e:
            await push_tok(client_id, f"Failed: {e}\n")
        await conditional_push_done(client_id)
        return

    # ── !drives decay ───────────────────────────────────────────────────────
    if parts and parts[0] == "decay":
        n = await decay_drives()
        await push_tok(client_id, f"Decay cycle complete — {n} drives adjusted toward baseline\n")
        await conditional_push_done(client_id)
        return

    # ── !drives seed ────────────────────────────────────────────────────────
    if parts and parts[0] == "seed":
        from memory import _mem_plugin_cfg
        import os, json as _json
        try:
            path = os.path.join(os.path.dirname(__file__), "plugins-enabled.json")
            with open(path) as f:
                cfg_data = _json.load(f)
            defaults = cfg_data.get("plugin_config", {}).get("drives", {}).get("defaults", [])
        except Exception as e:
            await push_tok(client_id, f"Failed to load drive defaults from plugins-enabled.json: {e}\n")
            await conditional_push_done(client_id)
            return

        if not defaults:
            await push_tok(client_id, "No drive defaults found in plugins-enabled.json → plugin_config.drives.defaults\n")
            await conditional_push_done(client_id)
            return

        tbl = _DRIVES()
        seeded = 0
        skipped = 0
        for d in defaults:
            name = str(d.get("name", "")).strip().lower()[:64]
            if not name:
                continue
            desc = str(d.get("description", "")).replace("'", "''")
            val  = float(d.get("value", 0.5))
            base = float(d.get("baseline", val))
            dr   = float(d.get("decay_rate", 0.05))
            try:
                existing = await fetch_dicts(
                    f"SELECT id FROM {tbl} WHERE name='{name}' LIMIT 1"
                )
                if existing:
                    skipped += 1
                    continue
                await execute_insert(
                    f"INSERT INTO {tbl} (name, description, value, baseline, decay_rate, source) "
                    f"VALUES ('{name}', '{desc}', {val:.4f}, {base:.4f}, {dr:.4f}, 'system')"
                )
                seeded += 1
            except Exception as e:
                await push_tok(client_id, f"  Failed to seed '{name}': {e}\n")
        await push_tok(client_id, f"Seeded {seeded} drives, skipped {skipped} (already exist)\n")
        await conditional_push_done(client_id)
        return

    # ── !drives (status) ────────────────────────────────────────────────────
    drives = await load_drives()
    if not drives:
        await push_tok(client_id,
            "No drives configured. Use `!drives seed` to seed defaults, "
            "or `!drives set <name> <0.0-1.0> <description>` to create one.\n")
        await conditional_push_done(client_id)
        return

    lines = ["**Active Drives**\n"]
    lines.append(f"{'Name':<22} {'Value':>6}  {'Baseline':>8}  Bar                    Description")
    lines.append("-" * 90)
    for d in drives:
        v = float(d.get("value", 0.5))
        b = float(d.get("baseline", 0.5))
        bar = int(round(v * 20))
        arrow = "▲" if v > b + 0.05 else ("▼" if v < b - 0.05 else "─")
        lines.append(
            f"{d['name']:<22} {v:>6.2f}  {b:>8.2f}  "
            f"{'█' * bar}{'░' * (20 - bar)} {arrow}  "
            f"{(d.get('description') or '')[:40]}"
        )
    lines.append("")
    lines.append("**Commands**: !drives set <name> <val>  |  !drives baseline <name> <val>  "
                 "|  !drives decay  |  !drives seed")
    await push_tok(client_id, "\n".join(lines) + "\n")
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


async def cmd_plugin_command(client_id: str, cmd: str, args: str):
    """
    Generic dispatcher for plugin-registered !commands.
    Looks up cmd in the _PLUGIN_COMMANDS registry and calls the handler.
    Handler signature: async (args: str) -> str
    Sets current_client_id ContextVar so handlers can access session/state.
    """
    handler = get_plugin_command(cmd)
    if handler is None:
        await push_tok(client_id, f"Unknown command: !{cmd}\nUse !help to see available commands.")
        await conditional_push_done(client_id)
        return
    from state import current_client_id as _ccid
    token = _ccid.set(client_id)
    try:
        result = await handler(args)
        await push_tok(client_id, result)
    except Exception as exc:
        await push_tok(client_id, f"ERROR: !{cmd} failed: {exc}")
    finally:
        _ccid.reset(token)
    await conditional_push_done(client_id)


async def cmd_get_system_info(client_id: str):
    """Show current date/time and system status."""
    from tools import get_system_info
    from state import current_client_id
    token = current_client_id.set(client_id)
    try:
        result = await get_system_info()
    finally:
        current_client_id.reset(token)
    await push_tok(client_id, str(result))
    await conditional_push_done(client_id)


async def cmd_llm_list(client_id: str):
    """List LLM models with their configuration."""
    from agents import llm_list
    result = await llm_list()
    await push_tok(client_id, result)
    await conditional_push_done(client_id)


async def cmd_llm_call_invoke(client_id: str, args: str):
    """
    Invoke a target LLM directly.

    Usage: !llm_call_invoke <model> <prompt> [key=value ...]

    Keyword options (all optional, defaults shown):
      mode=text          text | tool
      sys_prompt=none    none | caller | target
      history=none       none | caller
      tool=<name>        required when mode=tool

    Examples:
      !llm_call_invoke nuc11Local Summarize this text for me.
      !llm_call_invoke grok4 What's the weather? mode=text sys_prompt=caller
      !llm_call_invoke nuc11Local Find the top result tool=ddgs_search mode=tool
    """
    # Parse: first token = model, remaining tokens parsed for key=value overrides,
    # everything else concatenated as the prompt.
    parts = args.split()
    if len(parts) < 2:
        await push_tok(client_id,
            "Usage: !llm_call_invoke <model> <prompt> [mode=text|tool] [sys_prompt=none|caller|target] "
            "[history=none|caller] [tool=<name>]")
        await conditional_push_done(client_id)
        return

    model = parts[0].strip()
    kwargs = {"mode": "text", "sys_prompt": "none", "history": "none", "tool": ""}
    kw_keys = set(kwargs.keys())
    prompt_parts = []

    for token in parts[1:]:
        if "=" in token:
            k, _, v = token.partition("=")
            if k in kw_keys:
                kwargs[k] = v
                continue
        prompt_parts.append(token)

    prompt = " ".join(prompt_parts).strip()
    if not prompt:
        await push_tok(client_id, "ERROR: prompt is required.\nUsage: !llm_call_invoke <model> <prompt> [options]")
        await conditional_push_done(client_id)
        return

    from agents import llm_call as _llm_call
    from state import current_client_id
    token = current_client_id.set(client_id)
    try:
        result = await _llm_call(
            model=model,
            prompt=prompt,
            mode=kwargs["mode"],
            sys_prompt=kwargs["sys_prompt"],
            history=kwargs["history"],
            tool=kwargs["tool"],
        )
        await push_tok(client_id, result)
    except Exception as exc:
        await push_tok(client_id, f"ERROR: llm_call_invoke failed: {exc}")
    finally:
        current_client_id.reset(token)
    await conditional_push_done(client_id)






async def cmd_list_models(client_id: str, current: str):
    lines = ["Available models:"]
    # Enabled models from in-memory registry
    for key, meta in LLM_REGISTRY.items():
        model_id = meta.get("model_id", key)
        marker = " (current)" if key == current else ""
        lines.append(f"  {key:<12} {model_id}{marker}")
    # Disabled models from llm-models.json (not in LLM_REGISTRY)
    try:
        import json as _json
        with open(LLM_MODELS_FILE, "r") as _f:
            _data = _json.load(_f)
        for key, cfg in _data.get("models", {}).items():
            if not cfg.get("enabled", True) and key not in LLM_REGISTRY:
                model_id = cfg.get("model_id", key)
                lines.append(f"  {key:<12} {model_id}  [disabled]")
    except Exception:
        pass
    await push_tok(client_id, "\n".join(lines))
    await conditional_push_done(client_id)

async def cmd_stop(client_id: str):
    """Cancel the currently running LLM job for this client, if any."""
    cancelled = await cancel_active_task(client_id)
    if cancelled:
        await push_tok(client_id, "Job stopped.")
    else:
        await push_tok(client_id, "No job running.")
    await push_done(client_id)


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
        old_model = session["model"]
        old_cfg = LLM_REGISTRY.get(old_model, {})
        new_cfg = LLM_REGISTRY.get(key, {})
        session["model"] = key
        from state import save_session_config
        save_session_config(client_id, session)
        # Recompute effective window and trim history immediately
        trimmed = _notify_chain_model_switch(session, old_model, key, old_cfg, new_cfg)
        prev_len = len(session.get("history", []))
        session["history"] = trimmed
        dropped = prev_len - len(trimmed)
        await push_model(client_id, key)
        msg = f"Model set to '{key}'."
        if dropped > 0:
            msg += f" History trimmed: {dropped} message(s) removed ({len(trimmed)} kept)."
        await push_tok(client_id, msg)
    else:
        available = ", ".join(LLM_REGISTRY.keys())
        await push_tok(client_id,
            f"ERROR: Unknown model '{key}'\n"
            f"Available models: {available}\n"
            f"Use !model to list all models")
    await conditional_push_done(client_id)


async def cmd_reset(client_id: str, session: dict):
    """Clear conversation history for current session, summarizing to memory first."""
    from state import delete_history
    history = list(session.get("history", []))
    history_len = len(history)

    # Summarize departing history into short-term memory before clearing.
    # Skip entirely when memory.enabled is false — just wipe history.
    from agents import _memory_feature, _memory_cfg
    _sess_mem_enabled = session.get("memory_enabled", None)
    if history_len >= 4 and _memory_feature("enabled") and _memory_feature("reset_summarize") and (_sess_mem_enabled is None or _sess_mem_enabled):
        try:
            from memory import summarize_and_save
            set_model_context(session.get("model", ""))
            from config import get_model_role
            summarizer_model = _memory_cfg().get("summarizer_model") or get_model_role("summarizer")
            _reset_suppress = session.get("tool_suppress", False)
            if not _reset_suppress:
                await push_tok(client_id, "[memory] Summarizing session to memory...\n")
            status = await summarize_and_save(
                session_id=client_id,
                history=history,
                model_key=summarizer_model,
            )
            if not _reset_suppress:
                await push_tok(client_id, f"[memory] {status}\n")
        except Exception as _mem_err:
            log.warning(f"cmd_reset: memory summarize failed: {_mem_err}")

    session["history"] = []
    session["tool_subscriptions"] = {}
    session["tool_list_injected"] = False
    model_cfg = LLM_REGISTRY.get(session.get("model", ""), {})
    import plugin_history_default as _phd
    session["history_max_ctx"] = _phd.compute_effective_max_ctx(model_cfg)
    delete_history(client_id)
    await push_tok(client_id, f"Conversation history cleared ({history_len} messages removed).")
    await conditional_push_done(client_id)

async def cmd_tools(client_id: str, arg: str, session: dict):
    """
    !tools                        - show heat status of all tools for current model
    !tools reset                  - decay all subscriptions to cold, re-baseline
    !tools hot <toolsets>         - force-subscribe toolsets (comma or space separated)
    """
    from config import LLM_REGISTRY, LLM_TOOLSETS, LLM_TOOLSET_META
    from agents import _compute_active_tools, _subscribe_toolset

    model_key = session.get("model", "")
    cfg = LLM_REGISTRY.get(model_key, {})
    authorized_toolsets = cfg.get("llm_tools", [])
    subs = session.setdefault("tool_subscriptions", {})

    subcmd = arg.strip().lower() if arg else ""

    if subcmd == "reset":
        session["tool_subscriptions"] = {}
        session["tool_list_injected"] = False
        await push_tok(client_id, "Tool subscriptions cleared. Tool list will re-inject on next turn.")
        await conditional_push_done(client_id)
        return

    if subcmd.startswith("hot"):
        rest = subcmd[3:].strip().replace(",", " ").split()
        if not rest:
            await push_tok(client_id, "Usage: !tools hot <toolset1> [toolset2 ...]")
            await conditional_push_done(client_id)
            return
        activated = []
        unknown = []
        for ts in rest:
            if ts in authorized_toolsets and ts in LLM_TOOLSETS:
                _subscribe_toolset(session, ts, call_count=1)
                activated.append(ts)
            else:
                unknown.append(ts)
        msg_parts = []
        if activated:
            msg_parts.append(f"Force-subscribed: {', '.join(activated)}")
        if unknown:
            msg_parts.append(f"Unknown/unauthorized: {', '.join(unknown)}")
        await push_tok(client_id, "\n".join(msg_parts))
        await conditional_push_done(client_id)
        return

    # Default: show heat status table.
    # Resolve each llm_tools entry: toolset group names are used directly; literal
    # tool names are shown under their parent toolset label but filtered to only
    # the tools actually authorized for this model.
    seen_toolsets: set[str] = set()
    resolved: list[tuple[str, list[str], bool]] = []  # (ts_name, tools_list, is_group)

    # Set of toolset group names explicitly listed in this model's llm_tools.
    authorized_groups: set[str] = {e for e in authorized_toolsets if e in LLM_TOOLSETS}

    def _best_parent_ts(tool_name: str, all_literals: list[str]) -> str | None:
        """Find the parent toolset for a literal tool name.
        Prefer the toolset with the most authorized literal overlap; break
        ties by preferring smaller (more specific) toolsets."""
        candidates: list[tuple[int, int, str]] = []  # (-overlap, size, ts_name)
        for ts_name, tools in LLM_TOOLSETS.items():
            if tool_name in tools:
                overlap = sum(1 for t in all_literals if t in tools)
                candidates.append((-overlap, len(tools), ts_name))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][2]

    # Pre-group literals by parent toolset (preserving first-seen order per parent)
    literal_by_parent: dict[str, list[str]] = {}  # parent_ts -> [authorized tool names]
    entry_order: list[tuple[str, bool]] = []  # (key, is_group) in original order

    all_literals = [e for e in authorized_toolsets if e not in LLM_TOOLSETS]

    for entry in authorized_toolsets:
        if entry in LLM_TOOLSETS:
            entry_order.append((entry, True))
        else:
            parent_ts = _best_parent_ts(entry, all_literals)
            if parent_ts:
                if parent_ts not in literal_by_parent:
                    literal_by_parent[parent_ts] = []
                literal_by_parent[parent_ts].append(entry)
                entry_order.append((parent_ts, False))
            else:
                # Orphan — no parent toolset
                entry_order.append((entry, None))

    for key, kind in entry_order:
        if kind is True:
            # Toolset group name
            if key not in seen_toolsets:
                seen_toolsets.add(key)
                resolved.append((key, LLM_TOOLSETS[key], True))
        elif kind is False:
            # Literal tool(s) grouped under parent toolset
            if key not in seen_toolsets:
                seen_toolsets.add(key)
                resolved.append((key, literal_by_parent[key], False))
        else:
            # Orphan tool with no toolset
            resolved.append((key, [key], False))

    lines = [f"Tool heat status — model: {model_key}"]
    lines.append(f"{'Toolset':<16} {'Status':<18} {'Tools'}")
    lines.append("-" * 70)
    for ts_name, tools_in_set, is_group in resolved:
        meta = LLM_TOOLSET_META.get(ts_name, {})
        # Literal tool rows (is_group=False) are always-active — _compute_active_tools
        # adds them unconditionally regardless of the parent toolset's always_active flag.
        if not is_group or meta.get("always_active", True):
            status = "always-active"
        else:
            sub = subs.get(ts_name, {})
            heat = sub.get("heat", 0)
            call_count = sub.get("call_count", 0)
            heat_curve = meta.get("heat_curve", [])
            cap = heat_curve[-1] if heat_curve else "?"
            status = f"hot heat={heat}/{cap} calls={call_count}" if heat > 0 else "cold"
        lines.append(f"  {ts_name:<14} {status:<18} {', '.join(tools_in_set)}")
    await push_tok(client_id, "\n".join(lines))
    await conditional_push_done(client_id)


async def cmd_session(client_id: str, arg: str):
    """
    Manage sessions: list, attach, or delete.
    Usage:
      !session                - list all sessions (current marked)
      !session <ID> attach    - switch to different session
      !session <ID> delete    - delete a session
    """
    from state import sessions, get_or_create_shorthand_id, get_session_by_shorthand, remove_shorthand_mapping, estimate_history_size, format_session_token_line
    from memory import _mem_plugin_cfg as _mpcfg

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
                history = data.get("history", [])
                history_len = len(history)
                shorthand_id = get_or_create_shorthand_id(sid)
                peer_ip = data.get("peer_ip")
                ip_str = f", ip={peer_ip}" if peer_ip else ""
                size = estimate_history_size(history)
                char_k = size["char_count"]
                tok_est = size["token_est"]
                size_str = f" (~{char_k:,} chars, ~{tok_est:,} tok est)"
                token_line = format_session_token_line(data)
                _mem_flag = data.get("memory_enabled", None)
                if _mem_flag is not None:
                    mem_str = f", mem={'ON' if _mem_flag else 'OFF'}(session)"
                else:
                    mem_str = f", mem={'ON' if _mpcfg().get('enabled', True) else 'OFF'}(global)"
                lines.append(
                    f"  ID [{shorthand_id}] {sid}: model={model}, "
                    f"history={history_len} msgs{size_str}{ip_str}{mem_str}{marker}"
                )
                lines.append(token_line)
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


async def cmd_sleep(client_id: str, arg: str):
    """
    Sleep for a specified number of seconds.

    !sleep <seconds>  - pause for 1–300 seconds
    """
    import asyncio as _asyncio
    arg = arg.strip()
    if not arg:
        await push_tok(client_id, "Usage: !sleep <seconds>  (1–300)")
        await conditional_push_done(client_id)
        return
    try:
        seconds = int(arg.split()[0])
    except ValueError:
        await push_tok(client_id, f"ERROR: '{arg}' is not a valid integer.\nUsage: !sleep <seconds>")
        await conditional_push_done(client_id)
        return
    if seconds < 1 or seconds > 300:
        await push_tok(client_id, "ERROR: seconds must be between 1 and 300.")
        await conditional_push_done(client_id)
        return
    await push_tok(client_id, f"Sleeping for {seconds} second(s)...")
    await _asyncio.sleep(seconds)
    await push_tok(client_id, f"Done. Slept for {seconds} second(s).")
    await conditional_push_done(client_id)


# ---------------------------------------------------------------------------
# Phase 3 — Unified resource !commands
# Pattern: !<resource> <action> [args...]
# ---------------------------------------------------------------------------

async def cmd_llm_tools(client_id: str, arg: str):
    """!llm_tools <action> [name] [tools]"""
    from tools import _llm_tools_exec
    parts = arg.strip().split(None, 2)
    action = parts[0] if parts else "list"
    name = parts[1] if len(parts) > 1 else ""
    tools = parts[2] if len(parts) > 2 else ""
    result = await _llm_tools_exec(action, name=name, tools=tools)
    await push_tok(client_id, result + "\n")
    await conditional_push_done(client_id)


async def cmd_model_cfg(client_id: str, arg: str):
    """!model_cfg <action> [name] [field] [value]"""
    from tools import _model_cfg_exec
    parts = arg.strip().split(None, 3)
    action = parts[0] if parts else "list"
    name = parts[1] if len(parts) > 1 else ""
    field = parts[2] if len(parts) > 2 else ""
    value = parts[3] if len(parts) > 3 else ""
    result = await _model_cfg_exec(action, name=name, field=field, value=value)
    await push_tok(client_id, result + "\n")
    await conditional_push_done(client_id)


async def cmd_sysprompt_cfg(client_id: str, arg: str):
    """!sysprompt_cfg <action> [model] [file|newdir] [content]"""
    from tools import _sysprompt_cfg_exec
    parts = arg.strip().split(None, 3)
    action = parts[0] if parts else "list_dir"
    model = parts[1] if len(parts) > 1 else ""
    file_or_newdir = parts[2] if len(parts) > 2 else ""
    content = parts[3] if len(parts) > 3 else ""
    # Map file/newdir based on action
    if action in ("copy_dir", "set_dir"):
        result = await _sysprompt_cfg_exec(action, model=model, newdir=file_or_newdir)
    elif action == "write":
        result = await _sysprompt_cfg_exec(action, model=model, file=file_or_newdir, content=content)
    else:
        result = await _sysprompt_cfg_exec(action, model=model, file=file_or_newdir)
    await push_tok(client_id, result + "\n")
    await conditional_push_done(client_id)


async def cmd_config_cfg(client_id: str, arg: str):
    """!config <action> [key] [value]"""
    from tools import _config_cfg_exec
    from state import current_client_id
    parts = arg.strip().split(None, 2)
    action = parts[0] if parts else "list"
    key = parts[1] if len(parts) > 1 else ""
    value = parts[2] if len(parts) > 2 else ""
    token = current_client_id.set(client_id)
    try:
        result = await _config_cfg_exec(action, key=key, value=value)
    finally:
        current_client_id.reset(token)
    await push_tok(client_id, result + "\n")
    await conditional_push_done(client_id)


async def cmd_limits_cfg(client_id: str, arg: str):
    """!limits <action> [key] [value]"""
    from tools import _limits_cfg_exec
    from state import current_client_id
    parts = arg.strip().split(None, 2)
    action = parts[0] if parts else "list"
    key = parts[1] if len(parts) > 1 else ""
    value = parts[2] if len(parts) > 2 else ""
    token = current_client_id.set(client_id)
    try:
        result = await _limits_cfg_exec(action, key=key, value=value)
    finally:
        current_client_id.reset(token)
    await push_tok(client_id, result + "\n")
    await conditional_push_done(client_id)


async def cmd_vscode(client_id: str, arg: str):
    """!vscode list|read — browse and pull local Claude Code sessions into chat context."""
    from plugin_claude_vscode_sessions import _cmd_vscode
    from state import current_client_id
    token = current_client_id.set(client_id)
    try:
        result = await _cmd_vscode(arg)
    finally:
        current_client_id.reset(token)
    await push_tok(client_id, result + "\n")
    await conditional_push_done(client_id)


async def cmd_judge(client_id: str, arg: str, session: dict):
    """!judge — LLM-as-judge configuration and control."""
    from judge import cmd_judge as _judge_cmd
    result = await _judge_cmd(client_id, arg, session)
    await push_tok(client_id, result + "\n")
    await conditional_push_done(client_id)


async def cmd_stream(client_id: str, arg: str, session: dict):
    """!stream [0-3] — show or set response streaming/latency optimization level.

    0: off — ainvoke(), fresh client per request, serial enrich (default, conservative)
    1: LLM client cache per model-key (~100–200ms saved per turn)
    2: + parallel auto-enrich with 300ms timeout (inject if ready, defer if slow)
    3: + astream() sentence-chunking to TTS (first sentence arrives in ~1s)
    """
    LEVELS = {
        "0": "off — ainvoke(), fresh client, serial enrich",
        "1": "LLM client cache per model-key (~100–200ms saved)",
        "2": "+ parallel auto-enrich, 300ms timeout, inject-or-defer",
        "3": "+ astream() sentence-chunking to TTS",
    }
    arg = arg.strip()
    if not arg:
        cur = session.get("stream_level", 0)
        lines = [f"stream_level: {cur}\n"]
        for k, v in LEVELS.items():
            marker = "▶" if str(cur) == k else " "
            lines.append(f" {marker} {k}: {v}")
        await push_tok(client_id, "\n".join(lines) + "\n")
    elif arg in LEVELS:
        session["stream_level"] = int(arg)
        from state import save_session_config
        save_session_config(client_id, session)
        await push_tok(client_id, f"Stream level set to {arg}: {LEVELS[arg]}\n")
    else:
        await push_tok(client_id, "Usage: !stream [0-3]\n" + "\n".join(f"  {k}: {v}" for k, v in LEVELS.items()) + "\n")
    await conditional_push_done(client_id)


async def process_request(client_id: str, text: str, raw_payload: dict, peer_ip: str = None):
    from state import get_or_create_shorthand_id, load_history, load_session_config

    if client_id not in sessions:
        # Enforce max_users limit before creating new session
        max_u = get_max_users()
        if max_u > 0 and len(sessions) >= max_u:
            await push_tok(client_id,
                f"ERROR: Session limit reached ({max_u} active sessions). "
                f"Try again later or ask an administrator to increase !maxusers.")
            await push_done(client_id)
            return
        prior_history = load_history(client_id)
        prior_cfg = load_session_config(client_id)
        # Restore persisted model; payload default_model only applies for brand-new sessions
        _saved_model = prior_cfg.get("model")
        if _saved_model and _saved_model in LLM_REGISTRY:
            model_key = _saved_model
        else:
            model_key = raw_payload.get("default_model", DEFAULT_MODEL)
            if model_key not in LLM_REGISTRY:
                model_key = DEFAULT_MODEL
        model_cfg = LLM_REGISTRY.get(model_key, {})
        import plugin_history_default as _phd
        effective_ctx = _phd.compute_effective_max_ctx(model_cfg)
        import time as _time_init
        _model_tool_suppress = model_cfg.get("tool_suppress", get_default_tool_suppress())
        # Explicit user override (prior_cfg has the key) wins; otherwise model setting wins.
        _effective_tool_suppress = prior_cfg["tool_suppress"] if "tool_suppress" in prior_cfg else _model_tool_suppress
        _effective_mss = model_cfg.get("memory_scan_suppress", False) or prior_cfg.get("memory_scan_suppress", False)
        # agent_call_stream: model False wins (restrictive); model None defers to prior_cfg/default
        _model_stream = model_cfg.get("agent_call_stream", None)
        _effective_stream = (False if _model_stream is False else prior_cfg.get("agent_call_stream", True))
        # stream_level: model sets default if specified; prior_cfg (user !stream command) overrides
        _model_stream_level = model_cfg.get("stream_level", None)
        _effective_stream_level = prior_cfg.get("stream_level", _model_stream_level if _model_stream_level is not None else 0)
        sessions[client_id] = {
            "model": model_key,
            "history": prior_history,
            "history_max_ctx": effective_ctx,
            "tool_preview_length": prior_cfg.get("tool_preview_length", get_default_tool_preview_length()),
            "tool_suppress": _effective_tool_suppress,
            "memory_scan_suppress": _effective_mss,
            "agent_call_stream": _effective_stream,
            "stream_level": _effective_stream_level,
            "auto_enrich": prior_cfg.get("auto_enrich", True),
            "memory_enabled": prior_cfg["memory_enabled"] if "memory_enabled" in prior_cfg else None,
            "_client_id": client_id,
            "created_at": _time_init.time(),
            "tool_subscriptions": {},
            "tool_list_injected": False,
        }
        # Assign shorthand ID when session is created
        get_or_create_shorthand_id(client_id)
        # Age stale short-term memories to long-term on every session start (new or rehydrated)
        import asyncio as _asyncio
        try:
            from memory import age_to_longterm
            _asyncio.create_task(age_to_longterm())
        except Exception:
            pass
    # Store/update peer IP whenever we have it
    if peer_ip:
        sessions[client_id]["peer_ip"] = peer_ip
    session = sessions[client_id]

    # Reject requests from sessions flagged by AIRS async violation
    if session.get("airs_blocked"):
        import time as _time
        session["last_active"] = _time.time()  # keep session alive for admin review
        await push_tok(client_id,
            "[SECURITY] This session has been blocked due to an AI Runtime Security policy "
            "violation detected in a previous response. Contact an administrator to review "
            f"the violation report (report_id: {session.get('airs_block_report_id', 'unknown')}).")
        await push_done(client_id)
        return

    # If session's model was deleted from the registry, fall back to DEFAULT_MODEL
    if session.get("model") not in LLM_REGISTRY:
        old_model = session.get("model", "")
        session["model"] = DEFAULT_MODEL
        await push_tok(client_id,
            f"[WARNING] Model '{old_model}' is no longer available. "
            f"Switched to default model '{DEFAULT_MODEL}'.")

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
                elif cmd == "tools":
                    await cmd_tools(client_id, arg, session)
                elif cmd == "db_query":
                    await cmd_db_query(client_id, arg, session.get("model", ""))
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
                elif cmd == "llm_call_invoke":
                    await cmd_llm_call_invoke(client_id, arg)
                elif cmd == "model":
                    if arg:
                        await cmd_set_model(client_id, arg, session)
                    else:
                        await cmd_list_models(client_id, session["model"])
                elif cmd == "session":
                    await cmd_session(client_id, arg)
                elif cmd == "stop":
                    await cmd_stop(client_id)
                elif cmd == "sleep":
                    await cmd_sleep(client_id, arg)
                elif cmd == "llm_tools":
                    await cmd_llm_tools(client_id, arg)
                elif cmd == "model_cfg":
                    await cmd_model_cfg(client_id, arg)
                elif cmd == "sysprompt_cfg":
                    await cmd_sysprompt_cfg(client_id, arg)
                elif cmd == "config":
                    await cmd_config_cfg(client_id, arg)
                elif cmd == "limits":
                    await cmd_limits_cfg(client_id, arg)
                elif cmd == "vscode":
                    await cmd_vscode(client_id, arg)
                elif cmd == "memory":
                    await cmd_memory(client_id, arg)
                elif cmd == "memstats":
                    await cmd_memstats(client_id, session.get("model", ""))
                elif cmd == "membackfill":
                    await cmd_membackfill(client_id, session.get("model", ""))
                elif cmd == "memreconcile":
                    await cmd_memreconcile(client_id, session.get("model", ""))
                elif cmd == "memreview":
                    await cmd_memreview(client_id, arg, session.get("model", ""))
                elif cmd == "memclassify":
                    await cmd_memclassify(client_id, arg, session.get("model", ""))
                elif cmd == "memage":
                    await cmd_memage(client_id)
                elif cmd == "memtrim":
                    await cmd_memtrim(client_id, arg, session.get("model", ""))
                elif cmd == "cogn":
                    await cmd_cogn(client_id, arg, session.get("model", ""))
                elif cmd == "drives":
                    await cmd_drives(client_id, arg, session.get("model", ""))
                elif cmd == "stream":
                    await cmd_stream(client_id, arg, session)
                elif get_plugin_command(cmd) is not None:
                    await cmd_plugin_command(client_id, cmd, arg)
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
        if cmd == "tools":
            await cmd_tools(client_id, arg, session)
            return
        if cmd == "db_query":
            await cmd_db_query(client_id, arg, session.get("model", ""))
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
        if cmd == "llm_call_invoke":
            await cmd_llm_call_invoke(client_id, arg)
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
        if cmd == "sleep":
            await cmd_sleep(client_id, arg)
            return
        if cmd == "llm_tools":
            await cmd_llm_tools(client_id, arg)
            return
        if cmd == "model_cfg":
            await cmd_model_cfg(client_id, arg)
            return
        if cmd == "sysprompt_cfg":
            await cmd_sysprompt_cfg(client_id, arg)
            return
        if cmd == "config":
            await cmd_config_cfg(client_id, arg)
            return
        if cmd == "limits":
            await cmd_limits_cfg(client_id, arg)
            return
        if cmd == "vscode":
            await cmd_vscode(client_id, arg)
            return
        if cmd == "memory":
            await cmd_memory(client_id, arg)
            return
        if cmd == "memstats":
            await cmd_memstats(client_id, session.get("model", ""))
            return
        if cmd == "membackfill":
            await cmd_membackfill(client_id, session.get("model", ""))
            return
        if cmd == "memreconcile":
            await cmd_memreconcile(client_id, session.get("model", ""))
            return
        if cmd == "memreview":
            await cmd_memreview(client_id, arg, session.get("model", ""))
            return
        if cmd == "memclassify":
            await cmd_memclassify(client_id, arg, session.get("model", ""))
            return
        if cmd == "memage":
            await cmd_memage(client_id)
            return
        if cmd == "memtrim":
            await cmd_memtrim(client_id, arg, session.get("model", ""))
            return
        if cmd == "cogn":
            await cmd_cogn(client_id, arg, session.get("model", ""))
            return
        if cmd == "drives":
            await cmd_drives(client_id, arg, session.get("model", ""))
            return
        if cmd == "stream":
            await cmd_stream(client_id, arg, session)
            return
        if cmd == "judge":
            await cmd_judge(client_id, arg, session)
            return
        if cmd == "stop":
            await cmd_stop(client_id)
            return

        # Plugin-registered commands (e.g. !tmux, !tmux_call_limit from plugin_tmux)
        if get_plugin_command(cmd) is not None:
            await cmd_plugin_command(client_id, cmd, arg)
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
                # Same model — strip prefix, keep temp_model flag for this turn
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

    import time as _time
    session["last_active"] = _time.time()
    model_cfg = LLM_REGISTRY.get(session["model"], {})
    session["history"].append({"role": "user", "content": stripped})
    session["history"] = _run_history_chain(session["history"], session, model_cfg)

    try:
        final = await dispatch_llm(session["model"], session["history"], client_id)
        if final:
            # Strip memory_save() calls from final when memory_scan is disabled —
            # keeps history and conv_log clean of inline save noise.
            if not model_cfg.get("memory_scan", False):
                from agents import _strip_memory_calls
                final = _strip_memory_calls(final)
            session["history"].append({"role": "assistant", "content": final})
            # Post-response chain pass: plugins that inspect history[-1]["role"] == "assistant"
            # can filter or replace the response (e.g. security scanning).
            session["history"] = _run_history_chain(session["history"], session, model_cfg)
            # Verbatim conversation logging — save user prompt + assistant response as paired
            # memory rows when the model has conv_log enabled.
            # memory_enabled is checked at three levels (all must allow):
            #   model cfg (memory_enabled: false → always off for this model)
            #   session   (memory_enabled: false → off for this session)
            #   global    (plugins-enabled.json memory.enabled)
            _sess_mem = session.get("memory_enabled", None)
            _model_mem = model_cfg.get("memory_enabled", None)
            log.debug(f"conv_log gate: conv_log={model_cfg.get('conv_log')} sess_mem={_sess_mem} model_mem={_model_mem} model={session['model']}")
            if model_cfg.get("conv_log") and (_model_mem is None or _model_mem) and (_sess_mem is None or _sess_mem):
                try:
                    from memory import save_conversation_turn
                    set_model_context(session.get("model", ""))
                    _, _, _topic = await save_conversation_turn(
                        user_text=stripped,
                        assistant_text=final,
                        session_id=client_id,
                        memory_types_enabled=model_cfg.get("memory_types_enabled", False),
                    )
                    if _topic:
                        session["current_topic"] = _topic
                except Exception as _cl_err:
                    import logging as _log
                    _log.getLogger("routes").warning(f"conv_log save failed: {_cl_err}")
        else:
            # Remove dangling user message if LLM returned empty — prevents consecutive
            # user turns in history, which causes Gemini to return empty on next request
            if session["history"] and session["history"][-1]["role"] == "user":
                session["history"].pop()
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
    from state import get_or_create_shorthand_id, load_history, load_session_config
    if client_id not in sessions:
        import plugin_history_default as _phd
        prior_history = load_history(client_id)
        prior_cfg = load_session_config(client_id)
        # Restore persisted model; fall back to DEFAULT_MODEL for new sessions
        _saved_model = prior_cfg.get("model")
        _init_model = _saved_model if (_saved_model and _saved_model in LLM_REGISTRY) else DEFAULT_MODEL
        _mcfg = LLM_REGISTRY.get(_init_model, {})
        _model_tool_suppress = _mcfg.get("tool_suppress", get_default_tool_suppress())
        # Explicit user override (prior_cfg has the key) wins; otherwise model setting wins.
        _effective_tool_suppress = prior_cfg["tool_suppress"] if "tool_suppress" in prior_cfg else _model_tool_suppress
        _effective_mss = _mcfg.get("memory_scan_suppress", False) or prior_cfg.get("memory_scan_suppress", False)
        _model_stream = _mcfg.get("agent_call_stream", None)
        _effective_stream = (False if _model_stream is False else prior_cfg.get("agent_call_stream", True))
        _model_stream_level = _mcfg.get("stream_level", None)
        _effective_stream_level = prior_cfg.get("stream_level", _model_stream_level if _model_stream_level is not None else 0)
        sessions[client_id] = {
            "model": _init_model,
            "history": prior_history,
            "history_max_ctx": _phd.compute_effective_max_ctx(_mcfg),
            "tool_preview_length": prior_cfg.get("tool_preview_length", get_default_tool_preview_length()),
            "tool_suppress": _effective_tool_suppress,
            "memory_scan_suppress": _effective_mss,
            "agent_call_stream": _effective_stream,
            "stream_level": _effective_stream_level,
            "auto_enrich": prior_cfg.get("auto_enrich", True),
            "memory_enabled": prior_cfg["memory_enabled"] if "memory_enabled" in prior_cfg else None,
            "_client_id": client_id,
            "tool_subscriptions": {},
            "tool_list_injected": False,
        }
        get_or_create_shorthand_id(client_id)
        # Age stale short-term memories to long-term on every session start (new or rehydrated)
        import asyncio as _asyncio
        try:
            from memory import age_to_longterm
            _asyncio.create_task(age_to_longterm())
        except Exception:
            pass
    peer_ip = request.client.host if request.client else None
    if peer_ip:
        sessions[client_id]["peer_ip"] = peer_ip

    # Fix stale model in session (e.g. model was deleted while server was running)
    if sessions[client_id].get("model") not in LLM_REGISTRY:
        sessions[client_id]["model"] = DEFAULT_MODEL

    q = await get_queue(client_id)
    # Push current model so client knows what model is active immediately
    q.put_nowait({"t": "model", "d": sessions[client_id]["model"]})

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
            elif t == "model": yield {"event": "model", "data": item["d"]}
            elif t == "gate": yield {"event": "gate", "data": item["d"]}

    return EventSourceResponse(generator())

async def endpoint_stop(request: Request) -> JSONResponse:
    """Cancel the active LLM job for a client without starting a new one."""
    try: payload = await request.json()
    except: return JSONResponse({"error": "json"}, 400)
    client_id = payload.get("client_id")
    if not client_id: return JSONResponse({"error": "Missing client_id"}, 400)
    cancelled = await cancel_active_task(client_id)
    if cancelled:
        await push_done(client_id)
    return JSONResponse({"status": "OK", "cancelled": cancelled})


async def endpoint_gate_respond(request: Request) -> JSONResponse:
    """
    Resolve a pending gate request for a client.

    POST /gate_respond
    Body: {"client_id": "...", "approved": true|false}

    Returns {"status": "ok", "resolved": true} if a gate was pending,
    or {"status": "ok", "resolved": false} if no gate was waiting.
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    client_id = payload.get("client_id")
    approved = payload.get("approved")
    if not client_id:
        return JSONResponse({"error": "Missing client_id"}, status_code=400)
    if not isinstance(approved, bool):
        return JSONResponse({"error": "approved must be boolean"}, status_code=400)

    from state import resolve_gate
    resolved = resolve_gate(client_id, approved)
    return JSONResponse({"status": "ok", "resolved": resolved})


async def endpoint_health(request: Request) -> JSONResponse:
    return JSONResponse({
        "status": "healthy",
        "models": list(LLM_REGISTRY.keys()),
    })

async def endpoint_list_sessions(request: Request) -> JSONResponse:
    """List all active sessions with metadata."""
    from state import sessions, sse_queues, get_or_create_shorthand_id, estimate_history_size

    client_id_filter = request.query_params.get("client_id")

    session_list = []
    for cid, data in sessions.items():
        if client_id_filter and cid != client_id_filter:
            continue
        history = data.get("history", [])
        size = estimate_history_size(history)
        session_list.append({
            "client_id": cid,
            "shorthand_id": get_or_create_shorthand_id(cid),
            "model": data.get("model", "unknown"),
            "history_length": len(history),
            "history_chars": size["char_count"],
            "history_token_est": size["token_est"],
            "tokens_in_total": data.get("tokens_in_total", 0),
            "tokens_out_total": data.get("tokens_out_total", 0),
            "tokens_in_last": data.get("tokens_in_last"),
            "tokens_out_last": data.get("tokens_out_last"),
            "peer_ip": data.get("peer_ip"),
            "memory_enabled": data.get("memory_enabled", None),
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