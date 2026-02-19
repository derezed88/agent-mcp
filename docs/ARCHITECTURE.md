# Architecture

## Overview

The MCP Agent is a multi-client AI agent server. It maintains persistent sessions with conversation history, routes all client requests through a central LLM dispatch loop, enforces human-approval gates on tool calls, and exposes a modular plugin system for data tools and client interfaces.

```
Clients                 Server                          Backends
───────                 ──────                          ────────
shell.py     ──SSE──►  agent-mcp.py                    OpenAI API (grok, gpt, local)
open-webui    ─HTTP─►  ┌──────────────────────┐        Gemini API
LM Studio app ─HTTP─►  │  routes.py           │        Local llama.cpp / Ollama
Slack  ─Socket Mode──►  │  (process_request)   │
       ◄─Web API(bot)─  │                      │
api_client.py ─HTTP─►  │    │                 │
Agent B       ─HTTP─►  │    ▼                 │
                       │  agents.py           │
                       │  (dispatch_llm)      │
                       │  LangChain agentic   │
                       │  loop               │
                       │    │                 │
                       │    ▼                 │
                       │  execute_tool()      │──► gate.py ──► shell.py
                       │    │  agent_call()   │                (approval)
                       │    ▼                 │
                       │  Plugin executors    │──► MySQL
                       │  (db, drive, search) │──► Google Drive
                       └──────────────────────┘──► Web search APIs
                                │ agent_call
                                ▼
                         Agent B / Agent C  (other agent-mcp instances)
```

## Source Files

| File | Responsibility |
|---|---|
| `agent-mcp.py` | Entry point. Initialises plugins, builds Starlette app, starts uvicorn |
| `config.py` | LLM registry (loaded from `llm-models.json`), environment loading, rate limit config |
| `state.py` | In-memory session store, SSE queues, gate state, context vars |
| `routes.py` | HTTP endpoints, `!command` routing, `@model` switching, `process_request()` |
| `agents.py` | LangChain-based LLM dispatch loop (`agentic_lc`), `execute_tool()`, rate limiter |
| `gate.py` | Human approval gate logic per tool type |
| `tools.py` | Core tool definitions as LangChain `StructuredTool` objects; plugin tool registry |
| `prompt.py` | Recursive system prompt loader, section tree, `apply_prompt_operation()` |
| `plugin_loader.py` | `BasePlugin` ABC, dynamic plugin loading from manifest |
| `database.py` | MySQL connection and SQL execution helpers |

## Request Flow

```
Client sends message
       │
       ▼
process_request()  ──── stripped.startswith("!") ──► cmd_* handler ──► push_tok ──► done
       │
       ├── multi-command batch? ──► process each !cmd sequentially
       │
       ├── stripped.startswith("@") ──► @model temp switch
       │       sets session["model"] + session["_temp_model_active"]
       │
       ▼
session["history"].append(user message)
       │
       ▼
dispatch_llm(model, history, client_id)
       │
       └── agentic_lc()  ← single loop for all model types (OpenAI + Gemini)
               │
               │  _build_lc_llm(model_key)  ← ChatOpenAI or ChatGoogleGenerativeAI
               │  llm.bind_tools(_CURRENT_LC_TOOLS)
               │
               ▼ (loop, max MAX_TOOL_ITERATIONS)
        LLM API call via LangChain ainvoke()
               │
        tool_calls present?  ── No ──► try_force_tool_calls() (bare text fallback)
               │                             │
               │                      forced calls? ──► execute_tool() ──► inject as HumanMessage
               │
               ▼ (Yes — native tool calls)
        execute_tool(tool_name, tool_args, client_id)
               │
               ├── check_rate_limit()
               │
               ├── check_human_gate()
               │       ├── _temp_model_active? ──► auto-allow
               │       ├── llama/slack client? ──► auto-reject
               │       └── shell.py client ──► push gate to SSE ──► await approval
               │
               └── executor(**tool_args) ──► result ──► ToolMessage ──► back to LLM context
               │
        LLM produces final text response
               │
       ▼
session["history"].append(assistant message)
push_tok(response text) ──► SSE queue ──► client
```

## LangChain Integration

The agent uses LangChain as its LLM abstraction and tool-calling layer.

### LLM abstraction (`agents.py`)

`_build_lc_llm(model_key)` creates a LangChain chat model from the registry:

```python
# OPENAI type → ChatOpenAI (covers OpenAI, xAI, local llama.cpp, Ollama)
ChatOpenAI(model=..., base_url=..., api_key=..., streaming=True, timeout=...)

# GEMINI type → ChatGoogleGenerativeAI
ChatGoogleGenerativeAI(model=..., google_api_key=...)
```

Both return the same `ainvoke()` / `bind_tools()` interface — the rest of `agentic_lc()` is model-agnostic.

### Tool schema format (`tools.py`)

All tools — core and plugin — are defined as LangChain `StructuredTool` objects with Pydantic `BaseModel` argument schemas. This is the **single source of truth**:

```python
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

class _MyToolArgs(BaseModel):
    query: str = Field(description="Search query")

tool = StructuredTool.from_function(
    coroutine=my_executor,      # async executor function
    name="my_tool",
    description="Description shown to the LLM.",
    args_schema=_MyToolArgs,
)
```

`_lc_tool_to_openai_dict()` converts StructuredTool to OpenAI dict format on the fly for:
- `try_force_tool_calls()` — bare/XML tool call fallback for local models
- `get_openai_tool_schema()` — used by `llm_clean_tool`

No separate OpenAI or Gemini tool definitions are maintained.

### Content normalisation

`_content_to_str(content)` normalises `AIMessage.content` to a plain string.
Gemini returns content as `list[dict]` (content blocks); OpenAI returns `str`.
All `.content` accesses in `agents.py` go through this helper.

### Bare tool call fallback

Local models (Qwen, Hermes) sometimes output tool calls as raw text or XML rather than using the native tool-calling API. `try_force_tool_calls()` parses these and injects results as `HumanMessage` objects back into the context. Tool names are validated against the live `StructuredTool` registry.

## Slack Client Transport

The Slack plugin uses **asymmetric** transports — inbound and outbound are different mechanisms:

| Direction | Transport | Credential |
|---|---|---|
| Inbound (Slack → agent) | Socket Mode WebSocket (persistent connection) | `SLACK_APP_TOKEN` (starts with `xapp-`) |
| Outbound (agent → Slack) | Slack Web API `chat.postMessage` | `SLACK_BOT_TOKEN` (starts with `xoxb-`) |

**Required `.env` variables:**

```
SLACK_BOT_TOKEN=xoxb-...    # Web API calls (chat.postMessage for replies)
SLACK_APP_TOKEN=xapp-...    # Socket Mode connection (inbound events)
```

`SLACK_WEBHOOK_URL` is **not used** — do not add it. Webhook URLs cannot target specific threads, so `chat.postMessage` is used instead to maintain Slack thread context.

**Flow:**
1. Slack sends event over Socket Mode WebSocket → `_handle_socket_mode_request()`
2. Socket Mode acknowledgement sent immediately (required by Slack within 3 seconds)
3. Message dispatched to `process_request()` → LLM → response accumulated in queue
4. Accumulated response sent via `chat.postMessage` in the originating thread

## Session Model

Sessions are keyed by `client_id`:

| Client type | client_id format | Gate support |
|---|---|---|
| shell.py | read from `.aiops_session_id` | Full interactive approval |
| llama proxy | `llama-<client-ip>` | Auto-reject (non-interactive) |
| Slack | `slack-<channel_id>-<thread_ts>` | Auto-reject (non-interactive) |
| API plugin (direct) | `api-<8 hex chars>` (auto-generated) | 2-second window then auto-reject |
| Swarm (agent_call) | `api-swarm-<8 hex chars>` (derived from caller + URL) | Auto-reject (non-interactive) |

Each session stores: `model`, `history`, `tool_preview_length`, `_temp_model_active`.

Sessions persist in memory until explicitly deleted (`!session <ID> delete`) or server restart.

Shorthand IDs (101, 102, ...) are assigned sequentially and map to full client IDs for convenience.

## Plugin System

Plugins are declared in `plugin-manifest.json` and enabled/disabled in `plugins-enabled.json`.

### Plugin types

**`client_interface`** — adds HTTP/WebSocket endpoints to the server:
- `plugin_client_shellpy` — SSE streaming endpoint for shell.py (port 8765)
- `plugin_proxy_llama` — OpenAI/Ollama-compatible proxy (configurable port, default 11434)
- `plugin_client_slack` — Slack client: inbound via Socket Mode WebSocket, outbound via Web API (`chat.postMessage`)
- `plugin_client_api` — JSON/SSE HTTP API for programmatic access and swarm coordination (port 8767)

**`data_tool`** — registers tools callable by the LLM:
- `plugin_database_mysql` — `db_query` tool
- `plugin_storage_googledrive` — `google_drive` tool
- `plugin_search_ddgs` — `ddgs_search` tool
- `plugin_search_tavily` — `tavily_search` tool
- `plugin_search_xai` — `xai_search` tool
- `plugin_search_google` — `google_search` tool
- `plugin_urlextract_tavily` — `url_extract` tool

### Plugin loading sequence (agent-mcp.py startup)

1. Read `plugin-manifest.json` — all known plugins and their metadata
2. Read `plugins-enabled.json` — which plugins are active
3. Import each enabled plugin module dynamically
4. Call `plugin.init(config)` — connects to DB, authenticates, opens ports
5. Call `plugin.get_tools()` — returns `{'lc': [StructuredTool, ...]}` list
6. Call `plugin.get_gate_tools()` — registers gate types per tool
7. Register plugin routes with Starlette
8. Call `agents_module.update_tool_definitions()` — rebuilds `_CURRENT_LC_TOOLS` from core + all plugins

### BasePlugin contract

```python
class BasePlugin(ABC):
    PLUGIN_NAME: str       # e.g. "plugin_database_mysql"
    PLUGIN_TYPE: str       # "client_interface" | "data_tool"

    def init(self, config: dict) -> bool: ...    # return False to abort load
    def shutdown(self) -> None: ...
    def get_tools(self) -> dict: ...             # {"lc": [StructuredTool, ...]}
    def get_gate_tools(self) -> dict: ...        # {"tool_name": {"type": "search"|"drive"|..., ...}}
    def get_routes(self) -> list[Route]: ...     # client_interface only
```

## Gate System

Gates are per-tool human approval checkpoints. `check_human_gate()` in `gate.py` runs before every tool execution.

**Gate types and state:**

| Gate type | State dict | Controls |
|---|---|---|
| `db`      | `auto_aidb_state[table][read\|write]` | Per-table SQL approval |
| `search`  | `tool_gate_state[tool_name][read]` | Search tool approval |
| `extract` | `tool_gate_state[tool_name][read]` | URL extraction approval |
| `drive`   | `tool_gate_state["google_drive"][read\|write]` | Drive operation approval |
| `system`  | `tool_gate_state["update_system_prompt"][write]` | System prompt write approval |

**Wildcard defaults:** `auto_aidb_state["*"]` and `tool_gate_state["*"]` set defaults for all tables/tools without specific entries.

**Auto-bypass conditions:**
- `session["_temp_model_active"] = True` — set by `@model` prefix, bypasses all gates for that turn
- `client_id.startswith("llama-")` or `client_id.startswith("slack-")` — auto-rejects (non-interactive)
- `client_id.startswith("api-")` — pushes gate to SSE queue, waits 2 seconds (`API_GATE_TIMEOUT`) for client to respond via `POST /api/v1/gate/{gate_id}`, then auto-rejects if no response. `AgentClient` with `auto_approve_gates` policy responds within milliseconds when configured.

## System Prompt Structure

The system prompt is a recursive tree of section files:

```
.system_prompt                    ← root: main paragraph + [SECTIONS] list
  .system_prompt_memory-hierarchy
  .system_prompt_tool-guardrails
  .system_prompt_tools            ← container: [SECTIONS] list only
    .system_prompt_tool-db-query
    .system_prompt_tool-url-extract
    ...
  .system_prompt_behavior
```

Loop detection prevents circular references at load time. Duplicate section names across branches are also caught.

All sections at all depths are addressable by name or index:
- `read_system_prompt("tool-url-extract")` — returns just that tool's definition
- `update_system_prompt("behavior", "append", "...")` — edits any leaf section

Container sections (those with `[SECTIONS]`) cannot be directly edited — edit their children instead.

## LLM Model Registry

Models are registered in `llm-models.json` and loaded at startup by `config.py`. Two types are supported:

| Type | LangChain class | Examples |
|---|---|---|
| `OPENAI` | `ChatOpenAI` | grok-4, gpt-5.2, local llama.cpp, Ollama |
| `GEMINI` | `ChatGoogleGenerativeAI` | gemini-2.5-flash |

Each model entry:

| Field | Description |
|---|---|
| `model_id` | Model name passed to the API |
| `type` | `OPENAI` or `GEMINI` |
| `host` | API base URL (`null` for official Gemini endpoint) |
| `env_key` | `.env` key holding the API key (`null` for keyless local models) |
| `max_context` | Max messages retained in session history |
| `enabled` | `true`/`false` — disabled models are excluded from the registry entirely |
| `tool_call_available` | Whether `llm_clean_text`/`llm_clean_tool` may delegate to this model |
| `llm_call_timeout` | Timeout in seconds for delegation calls |
| `description` | Human-readable label shown in `!model` and `!llm_call` |

### Adding a new model

1. Add an entry to `llm-models.json`
2. Add the API key to `.env` if required
3. Restart the server — `config.py` loads `llm-models.json` at import time

No code changes are needed for models that use the standard `OPENAI` or `GEMINI` type.

## LLM Delegation Tools

Four mechanisms for the session LLM to delegate work:

| Tool | Target | Context sent | Use case |
|---|---|---|---|
| `llm_clean_text(model, prompt)` | Local model | Prompt only — no context, no tools | Summarization, classification, analysis |
| `llm_clean_tool(model, tool, arguments)` | Local model | Tool definition only | Isolated tool call via a second model |
| `agent_call(agent_url, message)` | Remote agent-mcp instance | None (fresh remote session or persisted swarm session) | Swarm / multi-agent coordination |
| `@model <prompt>` (user-initiated) | Local model | Full session | Full turn delegation to free/local model |

`llm_clean_text` and `llm_clean_tool` require `tool_call_available: true` on the target model. Enable with `!llm_call <model> true`.

`agent_call` routes through `plugin_client_api` and requires no additional configuration beyond the API plugin being enabled on the target instance.

## Swarm Architecture

Agent-to-agent communication is built on the API client plugin (`plugin_client_api`) as the transport layer.

```
Agent A (calling instance)           Agent B (target instance)
─────────────────────────            ──────────────────────────
LLM emits agent_call tool call
        │
        ▼
agent_call(agent_url, message)
        │
        │  POST /api/v1/submit        ┌─────────────────────────┐
        │─────────────────────────►   │  plugin_client_api      │
        │                             │  process_request()      │
        │  GET  /api/v1/stream/{id}   │  agents.py              │
        │◄────── SSE events ──────────│  (full LLM + tools)     │
        │        tok / gate / done    └─────────────────────────┘
        │
        ▼
result returned to Agent A's LLM as tool result
```

**Session persistence:** The swarm `client_id` is derived as `api-swarm-{md5(calling_client_id + ":" + agent_url)[:8]}`. This means repeated `agent_call` invocations from the same human session to the same remote agent reuse the same remote session, preserving conversation history across calls. Pass `target_client_id` explicitly to override.

**Depth guard:** If a session's `client_id` starts with `api-swarm-`, further `agent_call` invocations from that session are rejected immediately. This prevents unbounded recursion at 1 hop.

**Discovery:** There is currently no automatic discovery mechanism — target agents must be specified by URL. A discovery scheme is left to the operator or future development.

## Rate Limiting

Universal rate limiter in `agents.py` runs before the gate check. Configured per tool type in `plugins-enabled.json`:

```json
"rate_limits": {
  "llm_call": {"calls": 3, "window_seconds": 20, "auto_disable": true},
  "search":   {"calls": 5, "window_seconds": 10, "auto_disable": false},
  "extract":  {"calls": 5, "window_seconds": 30, "auto_disable": false},
  "drive":    {"calls": 10, "window_seconds": 60, "auto_disable": false},
  "db":       {"calls": 20, "window_seconds": 60, "auto_disable": false}
}
```

`auto_disable: true` disables all tools of that type for the session when the limit is exceeded.

## Client Protocol Detection

The llama proxy (`plugin_proxy_llama`) auto-detects the client format:

| Signal | Format |
|---|---|
| User-Agent contains "ollama" OR path starts with `/api/` | Ollama (NDJSON) |
| User-Agent contains "open-webui" | OpenAI (SSE) |
| Path starts with `/v1/api/` | Enchanted hybrid → Ollama response |
| Default | OpenAI (SSE) |

All formats route to the same `process_request()` — protocol differences are only in request parsing and response serialization.
