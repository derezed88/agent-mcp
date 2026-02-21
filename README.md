# agent-mcp

A multi-client AI agent server with a plugin architecture. Maintains persistent conversation sessions, routes all requests through a unified LangChain-based LLM dispatch loop, enforces human-approval gates on tool calls, and exposes a modular plugin system for data tools and client interfaces.

---

## Why Start Here Instead of From Scratch

Building a multi-LLM agent from scratch means solving: async session management, streaming SSE, multi-interface adapters (OpenAI vs. Ollama wire format alone is a week of work), per-tool gate/permission systems, rate limiting with auto-disable, tiered system prompt assembly, swarm coordination with depth guards, and a plugin discovery mechanism. This codebase has all of that working and tested across production use.

You inherit ~16,000 lines solving infrastructure so you can write the 100 lines that make your agent unique.

This system serves two distinct developer profiles. Both get a foundation rather than a blank page.

---

### Audience 1: Agent System Designers

You want to deploy a capable multi-LLM agent and shape its behavior — without writing Python.

**You don't write code. You design behavior.**

#### 5-Minute Start

```bash
git clone https://github.com/derezed88/agent-mcp.git
cd agent-mcp
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add at least one API key (e.g. GEMINI_API_KEY)
python agent-mcp.py           # server is running
python shell.py               # connect and start chatting
```

That's a working multi-tool LLM agent with MySQL, Google Drive, web search, and Slack support available as plugins.

#### Pick Your LLM — Or Use Several

`llm-models.json` is the model registry. Add an entry, drop the API key in `.env`, restart. No code. Out of the box:

- **Local**: Qwen2.5-7B via llama.cpp (no API cost, 32k context)
- **OpenAI**: gpt-4o-mini, gpt-5-mini, gpt-5-nano
- **Google**: Gemini 2.5 Flash / Flash-Lite
- **xAI**: Grok 4 Fast (reasoning and non-reasoning)

Switch the active model at runtime: `!model gemini25f` — persisted to disk immediately, no restart.

#### Control What the LLM Can Touch

Gates are per-tool, per-table read/write permissions. Set defaults in `gate-defaults.json` or change live in chat:

```
!autogate drive read true          # allow Drive reads this session
!autoAIdb write true               # allow DB writes for all tables
!autoAISysPrompt read true         # allow LLM to inspect its own prompt
```

Wildcard `"*"` key sets the default for all tables or all tools at once. Non-interactive clients (open-webui, Slack) auto-reject gated calls rather than hanging — the LLM is told why and asks for alternatives.

#### Tune Agent Behavior via Text Files

The system prompt is 22 modular section files — not a monolithic blob. Edit any section, add new ones, or assign a different folder to each model:

```
system_prompt/
├── 000_default/         ← 22 section files (behavior, memory chain, tool rules...)
└── 001_blank/           ← minimal alternative for specialized deployments
```

Give a local low-power model a stripped prompt; give frontier models the full PDDS memory chain. Each model's folder is set in `llm-models.json` — one field, no code.

#### Let the LLM Evolve Its Own Behavior

Say "remember permanently: always respond in bullet points" — the LLM calls `update_system_prompt`, writes a new section file, and that rule persists across restarts. Say "show me your system prompt" — it calls `read_system_prompt` and shows you. This is a live, self-evolving configuration controlled by natural language.

#### Runtime Admin Without Code

All configuration is JSON + commands. Nothing requires a restart:

```
!model gemini25f               # switch LLM
!autogate search read true     # allow web searches
!maxctx 50                     # set history window
!session                       # list active sessions with shorthand IDs
!session 102 delete            # drop a session
!limit_set max_agent_call_depth 2   # raise swarm recursion limit
```

Rate limits, session timeouts, tool permissions, model timeouts — all configurable live via `!commands` or `plugin-manager.py` CLI.

---

### Audience 2: Code-Level Developers

You're building custom integrations, new tools, or specialized agent behaviors on top of a working foundation.

**You write the 100 lines that make your agent unique. The other 15,900 are already here.**

#### 5-Minute Start for a New Plugin

The smallest working example in the codebase is [`plugin_search_ddgs.py`](plugin_search_ddgs.py) — 111 lines, fully functional:

```python
class SearchDdgsPlugin(BasePlugin):
    PLUGIN_NAME    = "plugin_search_ddgs"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE    = "data_tool"
    DESCRIPTION    = "Web search via DuckDuckGo (no API key required)"
    DEPENDENCIES   = ["ddgs"]

    def get_tools(self) -> dict:
        return {"lc": [StructuredTool.from_function(
            coroutine=search_ddgs_executor,
            name="search_ddgs",
            args_schema=_DdgsSearchArgs,
        )]}
```

That's it. [`plugin_loader.py`](plugin_loader.py) discovers it, wires it into the gate system, rate limiter, and LLM tool dispatch automatically. Declare the schema with Pydantic, return the tool, done.

#### The Tool Ecosystem Is Already Populated

[`tools.py`](tools.py) (1,249 lines) defines the core tool layer. Plugins extend it. What's already working:

| Category | Tools |
|---|---|
| Memory / storage | `db_query` (MySQL), `google_drive` (list/read/write/search) |
| Search | `search_ddgs` (no API key), `search_google`, `search_tavily`, `search_xai` |
| Web | `url_extract` (Tavily content extraction) |
| System | `get_system_info`, `read_system_prompt`, `update_system_prompt` |
| LLM delegation | `llm_clean_text`, `llm_clean_tool`, `at_llm` (full context), `agent_call` (swarm) |
| Inspection | `limit_list`, `gate_list`, `session`, `llm_list` |
| Terminal | `tmux_new`, `tmux_exec`, `tmux_history` (7 sub-tools via [`plugin_tmux.py`](plugin_tmux.py)) |

New tools you add are immediately available to all connected LLMs — no restart, because registration happens at load time.

#### History Is a Swappable Chain

[`plugin_history_default.py`](plugin_history_default.py) provides a sliding-window implementation. The contract is two methods:

```python
def process(history, session, model_cfg) -> list[dict]:
    """Called per-request. Returns trimmed history."""

def on_model_switch(session, old_model, new_model, ...) -> list[dict]:
    """Called immediately on !model switch."""
```

Swap it for Redis, SQLite, or a vector store by implementing a new `plugin_history_*.py`. The chain is configured in `plugins-enabled.json`:

```json
"chain": ["plugin_history_default", "plugin_history_custom"]
```

Each plugin in the chain receives the output of the previous one — composable history processing.

#### Swarm Architecture Is Already Wired

[`plugin_client_api.py`](plugin_client_api.py) exposes:

```
POST /api/v1/submit              ← submit message or command
GET  /api/v1/stream/{id}         ← SSE stream (tok, gate, done, error events)
POST /api/v1/gate/{gate_id}      ← respond to a gate programmatically
GET  /api/v1/sessions            ← list active sessions
```

`agent_call(agent_url, message)` lets any LLM call any other agent-mcp instance. Depth guards (`max_at_llm_depth`, `max_agent_call_depth`) prevent recursive runaway. Swarm session IDs are deterministic — repeated calls from the same session to the same remote agent reuse the same remote session, preserving history across calls.

Drive a session programmatically in ~10 lines via [`api_client.py`](api_client.py):

```python
from api_client import AgentClient
client = AgentClient("http://localhost:8767")
response = await client.send("!model")
async for token in client.stream("summarise my drive files"):
    print(token, end="", flush=True)
```

#### What You'd Have to Build Without This

The infrastructure already solved here, that you would otherwise spend weeks on:

- Async session management with per-client queues (`state.py`)
- SSE streaming with keepalives across all four client types
- OpenAI vs. Ollama wire format detection and translation (`plugin_proxy_llama.py`)
- Per-tool gate UI with interactive shell approval and auto-reject for non-interactive clients
- Rate limiting by tool type with auto-disable on breach (`agents.py:check_rate_limit`)
- Bare-JSON tool call extraction for local models that don't use the native function-calling API
- Deterministic swarm session ID derivation and one-hop depth guards
- Modular system prompt assembly with per-model folder assignment
- Plugin dependency validation, env-var checking, and priority-ordered load

#### The Plugin Contract Reference

```python
class BasePlugin(ABC):
    PLUGIN_NAME: str       # unique, matches filename
    PLUGIN_VERSION: str
    PLUGIN_TYPE: str       # "data_tool" or "client_interface"
    DESCRIPTION: str
    DEPENDENCIES: List[str]  # pip package names — validated at load time

    def init(self, config: dict) -> bool: ...      # connect, validate env
    def shutdown(self) -> None: ...
    def get_tools(self) -> dict: ...               # {"lc": [StructuredTool, ...]}
    def get_gate_tools(self) -> dict: ...          # declares gate types per tool
    def get_routes(self) -> List[Route]: ...       # client_interface only
    def get_commands(self) -> dict: ...            # optional !command handlers
    def get_help(self) -> str: ...                 # optional help text
```

See [`plugin_search_ddgs.py`](plugin_search_ddgs.py) (111 lines) for the minimal working pattern. See [`plugin_client_api.py`](plugin_client_api.py) (~300 lines) for a full client interface with SSE, gates, and session management.

---

## Technical Overview

### LangChain LLM Abstraction

All LLM backends are unified under a single `agentic_lc()` loop using LangChain:

- **`ChatOpenAI`** — covers OpenAI, xAI/Grok, and any local llama.cpp or Ollama server (all speak the OpenAI chat completions API)
- **`ChatGoogleGenerativeAI`** — covers Gemini models via the Google GenAI SDK

`_build_lc_llm(model_key)` constructs the correct LangChain chat model from `llm-models.json` at call time. Adding a new LLM backend requires only a JSON entry — no code changes.

The loop calls `llm.bind_tools(tools)` once and `llm.ainvoke(messages)` each iteration. Tool calls, results, and conversation history are exchanged as typed LangChain message objects (`AIMessage`, `ToolMessage`, `HumanMessage`). A `_content_to_str()` helper normalises model responses — Gemini returns content as a list of typed blocks; OpenAI returns a plain string.

### Tool Definition — LangChain StructuredTool

All tools — core and plugin — are defined as LangChain `StructuredTool` objects with Pydantic `BaseModel` argument schemas. This is the single source of truth for every backend:

```python
class _SearchArgs(BaseModel):
    query: str = Field(description="Search query")
    max_results: Optional[int] = Field(default=10, description="Max results")

StructuredTool.from_function(
    coroutine=search_executor,
    name="search_ddgs",
    description="Search the web via DuckDuckGo.",
    args_schema=_SearchArgs,
)
```

`_lc_tool_to_openai_dict()` converts StructuredTool to OpenAI dict format on the fly for the bare-text tool call fallback used by local models. No separate OpenAI or Gemini tool declarations are maintained anywhere.

### Bare Tool Call Fallback

Local models (Qwen, Hermes) sometimes output tool calls as raw JSON or XML rather than using the native function-calling API. `try_force_tool_calls()` extracts these from the model's text output and re-injects them as proper tool calls, with results fed back as `HumanMessage` objects. Tool names are validated against the live StructuredTool registry.

### Plugin Architecture

Plugins are loaded dynamically from `plugin-manifest.json` at startup. Each plugin is a Python file with a single `BasePlugin` subclass:

```python
class MyPlugin(BasePlugin):
    PLUGIN_TYPE = "data_tool"          # or "client_interface"

    def init(self, config) -> bool: ...     # connect, validate env
    def shutdown(self) -> None: ...
    def get_tools(self) -> dict: ...        # returns {"lc": [StructuredTool, ...]}
    def get_gate_tools(self) -> dict: ...   # declares gate types per tool
```

Executors are auto-extracted from `StructuredTool.coroutine` — no separate executor registry needed.

### Client Interfaces

| Client | Transport | Protocol |
|---|---|---|
| `shell.py` terminal | SSE (port 8765) | Custom SSE streaming + gate approval UI |
| OpenAI-compatible chat apps (open-webui, LM Studio) | HTTP (`llama_port`, default 11434) | OpenAI chat completions (streaming + non-streaming) |
| Ollama-compatible apps | HTTP (`llama_port`, default 11434) | Ollama NDJSON |
| Slack | Socket Mode WebSocket (inbound) + Web API (outbound) | Slack Events API / `chat.postMessage` |
| Programmatic / swarm | SSE (port 8767) | JSON/SSE via `api_client.py` or HTTP directly |

The llama proxy auto-detects client format from User-Agent and path prefix, then routes all formats to the same `process_request()` pipeline.

### Human Approval Gate System

Every tool call passes through `check_human_gate()` before execution. Gates are registered per tool type by plugins at startup — no hardcoded tool names in gate logic.

| Gate type | Command | Granularity |
|---|---|---|
| `search` tools | `!search_ddgs_gate_read true/false` | Per search engine |
| `url_extract` | `!url_extract_gate_read true/false` | Read gate |
| `google_drive` | `!google_drive_gate_read/write true/false` | Separate read and write |
| `db_query` | `!db_query_gate_read/write [table\|*] true/false` | Per-table, per-operation |
| `sysprompt_write` | `!sysprompt_gate_write true/false` | System prompt writes |
| `session` / `model` / `reset` | `!session_gate_read/write true/false` etc. | Per operation |

Gate defaults persist across restarts via `gate-defaults.json` (managed with `plugin-manager.py gate-set`).

Non-interactive clients (llama proxy, Slack) auto-reject gated calls immediately with an instructive message to the LLM. API clients get a 2-second window for programmatic approval.

### LLM Delegation

The session LLM can delegate sub-tasks to other registered models:

| Tool | What is sent | Use case |
|---|---|---|
| `llm_clean_text(model, prompt)` | Prompt only — no context, no tools | Summarisation, analysis |
| `llm_clean_tool(model, tool, args)` | Tool definition only | Isolated tool call via a second model |

Enable delegation per model with `!llm_call <model> true`. Rate-limited by default (3 calls / 20 s, auto-disables on breach).

### Swarm / Multi-Agent Communication

The `plugin_client_api` plugin exposes a JSON/SSE HTTP API (port 8767 by default) for programmatic and agent-to-agent access. Combined with the `agent_call` tool, any LLM on any instance can reach any other instance that has the API plugin enabled.

**`agent_call(agent_url, message)`** — core swarm tool:
- Sends `message` to a remote agent-mcp instance at `agent_url`
- The remote agent processes the message through its full stack (LLM, tools, gates)
- Returns the complete text response to the calling LLM
- Session persistence: the remote session is derived deterministically from the calling session + target URL, so repeated calls from the same session to the same agent reuse the same remote session (history preserved across calls)
- Depth guard: calls from a swarm-originated session are rejected with an error to prevent unbounded recursion (max 1 hop)

**Programmatic access via `api_client.py`:**

```python
from api_client import AgentClient

client = AgentClient("http://localhost:8767")
response = await client.send("!model")          # sync — returns full text
async for token in client.stream("hello"):      # streaming — yields tokens
    print(token, end="", flush=True)
```

The API plugin and `api_client.py` are also the transport layer used internally by `agent_call`.

> **Note:** A scheme for agents to discover each other automatically is not yet implemented. Swarm targets must be specified explicitly by URL.

### Modular System Prompt

The system prompt is assembled at runtime from section files stored in `system_prompt/<folder>/`. Each model can have its own folder, configured via `system_prompt_folder` in `llm-models.json`. The default folder is `system_prompt/000_default/`.

Sections are editable by the LLM via `sysprompt_write` / `sysprompt_delete` tools or by the operator via `!sysprompt_*` commands. The full prompt or any section can be read with `!sysprompt_read <model> [section]`.

---

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/derezed88/agent-mcp.git
cd agent-mcp
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add at least one LLM API key (e.g. GEMINI_API_KEY)
```

Edit `llm-models.json` to match your models and endpoints. At minimum, make sure one model is `"enabled": true` with a valid `env_key` pointing to your `.env` variable.

### 3. Start the server

```bash
source venv/bin/activate
python agent-mcp.py
```

### 4. Connect with shell.py (in a second terminal)

```bash
source venv/bin/activate
python shell.py
```

Type `!help` to see all commands. Some useful ones to start:

```
!model                          list available LLMs (* = current)
!model <name>                   switch active model
!search_ddgs_gate_read true     auto-allow DuckDuckGo searches (no gate pop-ups)
!db_query_gate_read * true      auto-allow all DB reads
!reset                          clear conversation history
!session                        list all active sessions
```

---

## Architecture

```
Clients                 Server                          Backends
───────                 ──────                          ────────
shell.py    ──SSE──►   agent-mcp.py                    OpenAI API
open-webui  ─HTTP──►   ┌──────────────────────────┐    Gemini API
LM Studio   ─HTTP──►   │ routes.py                │    xAI API
Slack ─Socket Mode──►  │ agents.py (agentic_lc)   │    llama.cpp / Ollama
api_client  ─HTTP──►   │   LangChain bind_tools()  │
Agent B ────HTTP──►    │   ChatOpenAI              │──► MySQL
                       │   ChatGoogleGenerativeAI  │──► Google Drive
                       │ plugin_*.py               │──► Web search APIs
                       └──────────────────────────┘
                                    │ agent_call tool
                                    ▼
                             Agent B / Agent C  (other agent-mcp instances)
```

- **`agent-mcp.py`** — entry point; loads plugins, builds Starlette app, starts uvicorn servers
- **`agents.py`** — LangChain dispatch loop, tool execution, rate limiting, LLM delegation
- **`tools.py`** — StructuredTool registry; single source of truth for all tool schemas
- **`shell.py`** — interactive terminal client with gate approval UI
- **`plugin_*.py`** — pluggable data tools and client interfaces
- **`plugin-manager.py`** — CLI tool for managing plugins and models

---

## Plugins

| Plugin | Type | What it adds |
|---|---|---|
| `plugin_client_shellpy` | client_interface | shell.py terminal client (SSE, port 8765) |
| `plugin_proxy_llama` | client_interface | OpenAI/Ollama API (configurable port, default 11434) |
| `plugin_client_slack` | client_interface | Slack bidirectional interface |
| `plugin_client_api` | client_interface | JSON/SSE HTTP API for programmatic access and swarm (port 8767) |
| `plugin_database_mysql` | data_tool | `db_query` — SQL against MySQL |
| `plugin_storage_googledrive` | data_tool | `google_drive` — CRUD within authorised folder |
| `plugin_search_ddgs` | data_tool | `search_ddgs` — DuckDuckGo (no API key) |
| `plugin_search_tavily` | data_tool | `search_tavily` — AI-curated results |
| `plugin_search_xai` | data_tool | `search_xai` — Grok x_search (web + X/Twitter) |
| `plugin_search_google` | data_tool | `search_google` — Gemini grounding |
| `plugin_urlextract_tavily` | data_tool | `url_extract` — web page content extraction |

Manage plugins:

```bash
python plugin-manager.py list
python plugin-manager.py enable <plugin_name>
python plugin-manager.py disable <plugin_name>
```

---

## Documentation

| Doc | Contents |
|---|---|
| [docs/QUICK_START.md](docs/QUICK_START.md) | Essential commands and first steps |
| [docs/ADMINISTRATION.md](docs/ADMINISTRATION.md) | Full plugin/model/gate/session management reference |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System internals, LangChain integration, request flow |
| [docs/PLUGIN_DEVELOPMENT.md](docs/PLUGIN_DEVELOPMENT.md) | How to write new plugins and add new models |
| [docs/setup_services.md](docs/setup_services.md) | systemd, tmux, screen, and tunnel deployment |
| [docs/plugin-client-api.md](docs/plugin-client-api.md) | API plugin — programmatic access and swarm setup |
| [docs/SWARMDESIGN.md](docs/SWARMDESIGN.md) | Swarm foundation and discovery design options |
| [docs/plugin-*.md](docs/) | Per-plugin setup and configuration |

---

## Configuration Files

| File | Purpose |
|---|---|
| `.env` | API keys and credentials (never commit) |
| `llm-models.json` | Model registry — `type`, `host`, `env_key`, `enabled`, `tool_call_available`, `system_prompt_folder` |
| `plugins-enabled.json` | Active plugins, rate limits, per-plugin config |
| `gate-defaults.json` | Gate auto-allow defaults loaded at startup (managed via `plugin-manager.py gate-set`) |
| `system_prompt/<folder>/` | Modular system prompt sections; `000_default/` ships with the repo |
