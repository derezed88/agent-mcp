# agent-mcp

A multi-client AI agent server with a plugin architecture. Maintains persistent conversation sessions, routes all requests through a unified LangChain-based LLM dispatch loop, enforces human-approval gates on tool calls, and exposes a modular plugin system for data tools and client interfaces.

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
