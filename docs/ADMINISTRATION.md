# Administration Guide

## Installation

### Prerequisites

- Python 3.11+
- A Python virtual environment
- At minimum: no external services required for shell.py-only mode

### Setup

```bash
git clone <repo>
cd agent-mcp
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # or install per-plugin deps below
cp .env.example .env              # fill in API keys
```

### Environment Variables (`.env`)

| Variable | Required by | Description |
|---|---|---|
| `GEMINI_API_KEY` | gemini models, google_search | Google AI Studio key |
| `XAI_API_KEY` | grok models, xai_search | xAI key |
| `OPENAI_API_KEY` | openai model | OpenAI key |
| `MYSQL_USER` | plugin_database_mysql | Database username |
| `MYSQL_PASS` | plugin_database_mysql | Database password |
| `TAVILY_API_KEY` | tavily_search, url_extract | Tavily API key |
| `FOLDER_ID` | plugin_storage_googledrive | Google Drive folder ID |
| `SLACK_BOT_TOKEN` | plugin_client_slack | Slack bot token |
| `SLACK_APP_TOKEN` | plugin_client_slack | Slack app token (Socket Mode) |

---

## Starting the Server

```bash
source venv/bin/activate
python agent-mcp.py
```

The server starts on port **8765** (MCP/shell.py) by default. Optional flags:

```bash
python agent-mcp.py --help
```

---

## Plugin Management (`plugin-manager.py`)

All plugin and model configuration is done through `plugin-manager.py`. Run interactively or with CLI arguments.

```bash
python plugin-manager.py           # interactive menu
python plugin-manager.py <cmd>     # direct CLI
```

### Plugin Commands

```bash
python plugin-manager.py list                    # list all plugins with status
python plugin-manager.py info <plugin_name>      # detailed info + setup instructions
python plugin-manager.py enable <plugin_name>    # enable a plugin
python plugin-manager.py disable <plugin_name>   # disable a plugin
```

**Plugin status indicators:**
- `✓` Enabled — active, all dependencies met, all credentials present
- `–` Disabled — in `enabled_plugins` but turned off via `enabled: false` in `plugin_config`
- `○` Configured — available (file + deps present) but not in `enabled_plugins`
- `✗` Has Issues — enabled but missing dependencies, env vars, or config files
- `⊗` Unavailable — not enabled and has unresolved issues

**Plugin names** (use with enable/disable):

| Plugin | Type | What it enables |
|---|---|---|
| `plugin_client_shellpy` | client_interface | shell.py terminal client (always keep enabled) |
| `plugin_proxy_llama` | client_interface | OpenAI/Ollama API (port set via `llama_port` in `plugins-enabled.json`) |
| `plugin_client_slack` | client_interface | Slack bidirectional client (see tuning below) |
| `plugin_database_mysql` | data_tool | `db_query` tool |
| `plugin_storage_googledrive` | data_tool | `google_drive` tool |
| `plugin_search_ddgs` | data_tool | `ddgs_search` tool (no key required) |
| `plugin_search_tavily` | data_tool | `tavily_search` tool |
| `plugin_search_xai` | data_tool | `xai_search` tool |
| `plugin_search_google` | data_tool | `google_search` tool |
| `plugin_urlextract_tavily` | data_tool | `url_extract` tool |

### Slack Plugin Tuning

The Slack plugin posts each agent turn to Slack immediately as it completes, rather than
waiting for the full conversation to finish. After each turn it waits up to
`inter_turn_timeout` seconds for the next turn to begin before declaring the conversation done.

**Why you may need to tune this:**

- **Too low (< 10s):** The final summary from the orchestrating LLM (e.g. grok4) is cut off.
  After the last `agent_call` returns, the LLM still needs one more inference pass to generate
  its synthesis — frontier models typically need 3–10s for this. If the timeout expires first,
  the Slack conversation ends without the closing summary.
- **Too high (> 60s):** No functional harm, but the Slack thread stays "open" for longer after
  the last message appears, which may look like the agent is still working.

**Default:** 30 seconds. Configure in `plugins-enabled.json`:

```json
"plugin_client_slack": {
  "enabled": true,
  "slack_port": 8766,
  "slack_host": "0.0.0.0",
  "inter_turn_timeout": 30
}
```

Override without editing JSON using `.env`:
```
SLACK_INTER_TURN_TIMEOUT=45
```
The JSON value takes precedence over `.env` if both are set.

---

### Model Commands

```bash
python plugin-manager.py models                              # list all models
python plugin-manager.py model-info <model_name>            # detailed model info
python plugin-manager.py model-add                          # interactive wizard
python plugin-manager.py model-remove <model_name>          # remove a model
python plugin-manager.py model-enable <model_name>          # enable a model
python plugin-manager.py model-disable <model_name>         # disable a model
python plugin-manager.py model <model_name>                 # set as default model
python plugin-manager.py model-llmcall <model_name> <t|f>  # set tool_call_available
python plugin-manager.py model-llmcall-all <t|f>           # set for all models
python plugin-manager.py model-timeout <model_name> <secs> # set llm_call_timeout
```

**Safety rules:** The default model cannot be disabled or removed. Change the default first with `model <name>`.

### Rate Limit Commands

```bash
python plugin-manager.py ratelimit-list                     # show current limits
python plugin-manager.py ratelimit-set <type> <n> <secs>   # set limit
python plugin-manager.py ratelimit-autodisable <type> <t|f> # set auto-disable
```

Tool types: `llm_call`, `search`, `extract`, `drive`, `db`, `system`

---

## Runtime Administration (shell.py commands)

Once connected via shell.py, all administration is done via `!commands`.

### Model Management

```
!model                     list available models (current marked)
!model <name>              switch active LLM for this session
```

### Gate Management

Gates are per-session controls that require human approval before tool calls execute.

**Database gates:**
```
!autoAIdb status                        show current DB gate settings
!autoAIdb <table> read <true|false>     toggle read gate for a specific table
!autoAIdb <table> write <true|false>    toggle write gate for a specific table
!autoAIdb <table> <true|false>          toggle both read+write for a table
!autoAIdb read <true|false>             set DEFAULT for ALL tables
!autoAIdb write <true|false>            set DEFAULT for ALL tables
!autoAIdb __meta__ read <true|false>    toggle gate for SHOW/DESCRIBE queries
```

**Tool gates (search, extract, drive):**
```
!autogate status                        show all tool gate settings
!autogate search <true|false>           toggle gate for ALL search tools
!autogate <tool_name> <true|false>      toggle gate for one specific tool
!autogate extract <true|false>          toggle gate for ALL extract tools
!autogate drive read <true|false>       toggle Drive read gate
!autogate drive write <true|false>      toggle Drive write gate
!autogate read <true|false>             set DEFAULT for all tools
!autogate write <true|false>            set DEFAULT for all tools
```

**System prompt gates:**
```
!autoAISysPrompt                        show current settings
!autoAISysPrompt read <true|false>      toggle read_system_prompt gate
!autoAISysPrompt write <true|false>     toggle update_system_prompt gate
```

`true` = gate OFF (auto-allow, no approval needed)
`false` = gate ON (requires human approval each time)

**Wildcard defaults:** Setting a gate without a specific table/tool name sets the default for all unspecified entries.

### @model — Per-Turn Model Switch

Prefix any prompt with `@ModelName` to temporarily use a different model for that one turn:

```
@localmodel extract https://www.example.com and summarize it
@gemini25 what is the weather like today?    ← same model, just bypasses gates
```

- Result lands in shared session history
- Original model restored after the turn
- All gates are bypassed (admin-initiated delegation)
- Works with same model too (gate bypass without model switch)

### Session Management

```
!session                        list all active sessions with shorthand IDs
!session <ID> delete            delete a session (ID = shorthand integer or full ID)
!reset                          clear conversation history for current session
```

### Tool Preview Control

Tool results are always sent in full to the LLM. This controls what is displayed in the chat:

```
!tool_preview_length            show current setting (default: 500 chars)
!tool_preview_length <n>        set to n characters
!tool_preview_length 0          unlimited (no truncation)
```

Gate pop-up preview length (shell.py only):
```
!gate_preview_length [n]        get/set gate approval preview char limit
```

### Agent Streaming Control

```
!stream                 show current agent_call streaming setting (default: enabled)
!stream <true|false>    enable/disable real-time token relay from remote agent
```

> **Note:** The primary node is always the orchestrator. The remote agent responds
> to single messages — it does not itself call agent_call back. Multi-turn conversations
> are conducted by the primary node making repeated agent_call invocations, one per turn.

When enabled (default), remote agent tokens are relayed via push_tok in real-time —
Slack sees each remote turn as it completes rather than as a batch at the end.
Set to `false` to suppress streaming and return only the final result.

---

### LLM Tool Calls

Control which models the session LLM can delegate to via `llm_call_clean`:

```
!llm_call                       list models with tool_call_available status
!llm_call <model> <true|false>  enable/disable llm_call_clean for a model
!llm_call <true|false>          set for ALL models
```

### System Prompt

```
!sysprompt                      show current assembled system prompt
!sysprompt reload               reload all .system_prompt_* files from disk
!read_system_prompt             show full cached prompt
!read_system_prompt <index>     show section by index (0, 1, 2...)
!read_system_prompt <name>      show section by name (e.g. "tool-url-extract")
```

### Direct SQL

```
!db <SQL>                       run SQL directly (no LLM, no gate)
```

---

## System Prompt Administration

The system prompt is split into section files under the project directory. Edit them directly or via the `update_system_prompt` tool (if the write gate is open).

### File structure

```
.system_prompt                      root file (main paragraph + [SECTIONS] list)
.system_prompt_<section-name>       each section file
```

A section file that contains `[SECTIONS]` is a container — it declares child sections. A section file with body text is a leaf.

### Adding a new section

1. Add the entry to the `[SECTIONS]` list in the appropriate parent file:
   ```
   my-new-rule: Description of the rule
   ```
2. Create `.system_prompt_my-new-rule` with the content.
3. Run `!sysprompt reload` in shell.py.

**Loop detection:** If a section name appears in its own ancestor chain, the loader substitutes an error placeholder and logs the error. Duplicate section names across branches are also caught.

### LLM-editable sections

The `update_system_prompt` tool (when gate is open) can perform surgical edits:
- `append` — add to end of a section
- `prepend` — add to beginning
- `replace` — find exact string and replace
- `delete` — remove lines containing a target string
- `overwrite` — replace entire section (requires `confirm_overwrite=true`)

Container sections cannot be directly edited — edit their child sections instead.

---

## Deployment

### Development (foreground)

```bash
source venv/bin/activate
python agent-mcp.py
```

### Production (systemd)

Create `/etc/systemd/system/agent-mcp.service`:

```ini
[Unit]
Description=Agent MCP
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/projects/agent-mcp
ExecStart=/home/YOUR_USER/projects/agent-mcp/venv/bin/python agent-mcp.py
Restart=on-failure
RestartSec=5
EnvironmentFile=/home/YOUR_USER/projects/agent-mcp/.env
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable agent-mcp
sudo systemctl start agent-mcp
sudo journalctl -u agent-mcp -f
```

### Development (tmux)

```bash
tmux new-session -d -s mcp 'cd /home/YOUR_USER/projects/agent-mcp && source venv/bin/activate && python agent-mcp.py'
tmux attach -t mcp
```

### Remote access via SSH tunnel

The agent can be exposed remotely via an SSH tunnel service (e.g. Pinggy, ngrok, Cloudflare Tunnel).
Configure the tunnel to forward your local port to the remote endpoint, then add the remote host
to `llm-models.json` for any models served via that tunnel.

---

## Configuration Files

| File | Purpose | Edited by |
|---|---|---|
| `.env` | API keys and credentials | Admin manually |
| `plugins-enabled.json` | Which plugins are active + per-plugin config + rate limits | `plugin-manager.py` or direct edit |
| `plugin-manifest.json` | Plugin registry — metadata, deps, env vars (read-only) | Plugin authors only |
| `llm-models.json` | Model registry (enabled, model_id, type, etc.) | `plugin-manager.py` |
| `.system_prompt` | Root system prompt file | Admin manually or LLM via tool |
| `.system_prompt_*` | Individual section files | Admin manually or LLM via tool |
| `.aiops_session_id` | shell.py session persistence | shell.py automatically |

### `plugin-manifest.json` vs `plugins-enabled.json`

These two files have distinct, non-overlapping roles:

**`plugin-manifest.json` — the plugin catalog (read-only)**

Declares that a plugin *exists* and what it needs to run: its Python file, type,
pip dependencies, required `.env` variables, and load priority.  This file is
maintained by plugin authors and committed to the repo.  The agent and
`plugin-manager.py` read it purely for validation — to check whether a plugin's
dependencies and credentials are present before attempting to load it.  You never
edit this file to enable or disable plugins.

**`plugins-enabled.json` — the operator control panel (read/write)**

Determines what actually runs.  It has three jobs:

1. **`enabled_plugins` list** — the ordered list of plugin names the agent will
   attempt to load at startup.  Add a plugin here to activate it; remove it to
   deactivate it entirely.  Managed by `plugin-manager.py enable/disable` or by
   direct edit.

2. **`plugin_config` blocks** — per-plugin runtime settings such as port, host,
   and the `enabled` flag.  The `enabled: false` pattern lets you keep a plugin
   in `enabled_plugins` (preserving its config) without starting it.  This is how
   `plugin_proxy_llama` and `plugin_client_slack` ship: configured but off until
   you flip `"enabled": true` or run `plugin-manager.py enable <plugin>`.

3. **`rate_limits`** and **`default_model`** — server-wide settings also stored here.

**Practical rule:** to enable or disable a plugin, always use `plugin-manager.py`
or edit `plugins-enabled.json`.  Never add enable/disable logic to `plugin-manifest.json`.

**Fresh installs:** `setup-agent-mcp.sh` clones the repo and copies credentials
(`.env`, `credentials.json`, `llm-models.json`) from a reference installation.
It intentionally does *not* copy `plugins-enabled.json` — the repo's version is
the authoritative default for new installs, and port assignments are adjusted
per-instance afterward with `plugin-manager.py port-set`.

---

## Diagnostics

### Test scripts

All `test_*.sh` scripts verify client protocol compatibility. Run against a live server:

```bash
./test_ollama_client.sh          # Ollama API endpoints
./test_openai_models.sh          # OpenAI /v1/models endpoint
./test_enchanted_app.sh          # Enchanted iOS hybrid endpoints
./test_ios_compatibility.sh      # OpenAI API for iOS apps
./test_openwebui.sh              # open-webui bare paths
./test_llama_proxy.sh            # llama proxy !commands
./test_llama_proxy_gate.sh       # gate auto-rejection for proxy clients
./test_streaming_role_fix.sh     # first-chunk role field (OpenAI spec)
./test_history_ignore.sh         # server ignores client-sent history
./test_model_ignore.sh           # server ignores client-sent model
./test_model_disable.sh          # disabled models excluded from registry
./test_unknown_commands.sh       # unknown !commands caught
./test_immediate_disconnect.sh   # disconnect handling
./test_win11_response.sh         # local model streaming
```

### Health endpoint

```bash
curl http://localhost:8765/health
```

Returns: server status, loaded models, current gate state.
