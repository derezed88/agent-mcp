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

## System Administration (`agentctl.py`)

All plugin and model configuration is done through `agentctl.py`. Run interactively or with CLI arguments.

```bash
python agentctl.py           # interactive menu
python agentctl.py <cmd>     # direct CLI
```

### Plugin Commands

```bash
python agentctl.py list                    # list all plugins with status
python agentctl.py info <plugin_name>      # detailed info + setup instructions
python agentctl.py enable <plugin_name>    # enable a plugin
python agentctl.py disable <plugin_name>   # disable a plugin
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
| `plugin_tmux` | data_tool | PTY shell sessions (`tmux_new`, `tmux_exec`, etc.) |

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
python agentctl.py models                              # list all models
python agentctl.py model-info <model_name>            # detailed model info
python agentctl.py model-add                          # interactive wizard
python agentctl.py model-remove <model_name>          # remove a model
python agentctl.py model-enable <model_name>          # enable a model
python agentctl.py model-disable <model_name>         # disable a model
python agentctl.py model <model_name>                 # set as default model
python agentctl.py model-llmcall <model_name> <t|f>  # set tool_call_available
python agentctl.py model-llmcall-all <t|f>           # set for all models
python agentctl.py model-timeout <model_name> <secs> # set llm_call_timeout
```

**Safety rules:** The default model cannot be disabled or removed. Change the default first with `model <name>`.

### Rate Limit Commands

```bash
python agentctl.py ratelimit-list                     # show current limits
python agentctl.py ratelimit-set <type> <n> <secs>   # set limit
python agentctl.py ratelimit-autodisable <type> <t|f> # set auto-disable
```

Tool types: `llm_call`, `search`, `extract`, `drive`, `db`, `system`, `tmux`

---

## API Client Trust Model

**All clients connecting to the API port (`plugin_client_api`, default 8767) are treated as trusted administrators.**

There is no inbound command ACL — any client that can reach the port can send any message, including `!commands`. The gate system is the enforcement layer: LLMs must pass gate checks for sensitive tool calls, but the human-facing command interface has no restrictions for API clients.

This is intentional. The API port is for trusted orchestrators: other agent-mcp instances, automation scripts, and inter-agent swarms. Inbound restriction is out of scope; network-level controls (firewall, SSH tunnel, VPN) are expected to restrict who can reach the port at all.

### Outbound Agent Message Filters (`OUTBOUND_AGENT_*`)

When an LLM makes an `agent_call` to a remote agent, the outbound message text can be filtered
before it is sent. This is a secondary safeguard for operators who want to restrict what
instructions their agent forwards to other agents.

Configure in `plugins-enabled.json` under `plugin_config.plugin_client_api`:

```json
"plugin_client_api": {
  "OUTBOUND_AGENT_ALLOWED_COMMANDS": [],
  "OUTBOUND_AGENT_BLOCKED_COMMANDS": [
    "rm -rf",
    "shutdown",
    "reboot"
  ]
}
```

**Semantics:**
- `OUTBOUND_AGENT_ALLOWED_COMMANDS`: empty `[]` = all outbound messages permitted (no check).
  Non-empty = message must start with one of the listed prefixes, otherwise blocked.
- `OUTBOUND_AGENT_BLOCKED_COMMANDS`: always checked when non-empty; empty `[]` = nothing blocked.
  Message must not start with any listed prefix.

**Default:** both lists are empty — all agent-to-agent messages are permitted. These filters exist
for operators who want an extra layer of control over what one agent forwards to another.

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

### PTY Shell Sessions (`plugin_tmux`)

The tmux plugin provides persistent PTY (pseudo-terminal) shell sessions. LLMs interact via
tool calls; humans manage sessions via `!tmux` commands.

> **Advanced users only.** PTY sessions give an LLM direct shell access. Output is captured
> after a silence timeout — long-running commands should be backgrounded with `&` and polled.
> See the "Advanced: PTY Session Semantics" section below.

#### Gate Control

All tmux tool calls are **write-gated by default** — the human must approve each call. No
read gate exists: `tmux_ls` and `tmux_history` are also write-gated because they expose
session topology (PIDs, names) and command history including credentials.

```
!tmux_gate_write <true|false>              — group gate covering all tmux tools
!tmux_new_gate_write <true|false>          — per-tool override
!tmux_exec_gate_write <true|false>         — per-tool override
!tmux_ls_gate_write <true|false>           — per-tool override
!tmux_history_gate_write <true|false>      — per-tool override
!tmux_kill_session_gate_write <true|false> — per-tool override
!tmux_kill_server_gate_write <true|false>  — per-tool override
!tmux_history_limit_gate_write <true|false>— per-tool override
!tmux_call_limit_gate_read <true|false>    — rate limit read gate
!tmux_call_limit_gate_write <true|false>   — rate limit write gate
```

`true` = gate OFF (auto-allow). `false` = gate ON (requires human approval).

#### Session Commands

```
!tmux new <name>              — create a new PTY session
!tmux ls                      — list active sessions
!tmux kill-session <name>     — terminate one session
!tmux kill-server             — terminate all sessions
!tmux a <name>                — show session history (attach view)
!tmux history-limit [n]       — show or set rolling history line limit
!tmux filters                 — show current command filter configuration
```

#### Rate Limiting

```
!tmux_call_limit                        — show current rate limit
!tmux_call_limit <calls> <window_secs>  — set rate limit
```

Default: 30 calls per 60 seconds. `auto_disable=true` — on breach, all tmux tools are
disabled until agent restart. Configure base values in `plugins-enabled.json`:

```json
"rate_limits": {
  "tmux": { "calls": 30, "window_seconds": 60, "auto_disable": true }
}
```

#### Command Filtering

Two filter lists in `plugin_config.plugin_tmux` control which commands can be sent to PTY sessions:

```json
"plugin_tmux": {
  "TMUX_ALLOWED_COMMANDS": [],
  "TMUX_BLOCKED_COMMANDS": ["rm -rf", "dd if=", "mkfs", "shutdown", "reboot"]
}
```

**Semantics:**
- `TMUX_ALLOWED_COMMANDS`: empty `[]` = all commands permitted. Non-empty = command must
  match a listed prefix, otherwise blocked.
- `TMUX_BLOCKED_COMMANDS`: always checked; empty `[]` = nothing blocked.

Additionally, `OUTBOUND_AGENT_BLOCKED_COMMANDS` (from `plugin_client_api`) is also applied
inside `tmux_exec` — so the same patterns that block outbound agent messages also block PTY
commands when configured.

#### Advanced: PTY Session Semantics

PTY sessions are true pseudo-terminals with persistent state (cwd, environment, background
jobs). Key behaviors to understand:

- **Output capture:** output is drained after a configurable silence timeout (default 10s).
  If a command produces no output for 10 seconds, the call returns with what was captured so far.
- **Long-running commands:** background with `&` and tee to a log file. Poll with
  `tail logfile` + `jobs` in subsequent `tmux_exec` calls.
- **Credential exposure:** any secrets printed to the terminal (API keys, passwords, tokens)
  will appear in the captured output and be visible to the LLM. Use `.env` files and avoid
  echoing secrets.
- **Prompt injection risk:** malicious content in command output (e.g. from a web response
  stored in a file and cat'd) could attempt to manipulate the LLM. Review outputs before
  passing them back to untrusted LLMs.
- **No interactive prompts:** PTY commands that pause for interactive input (sudo password,
  confirmation prompts) will hang until the timeout expires. Use `-y` flags, `yes |`, or
  pre-configure passwordless sudo for commands the LLM needs to run.

---

### LLM Delegation

The agent supports four modes of LLM delegation — ways for either the user or the LLM itself to
invoke another model. Each mode provides a different level of context isolation and is subject to
different gates and rate limits.

> **Security note:** Every `!command` available to the human user is also available as a tool call
> to the LLM. A model with write gates open can switch models, reset history, edit system prompts,
> and change gate and limit settings on its own. Enable only the gates you intend to. When an
> `at_llm` or `agent_call` is delegating to another model, that remote model inherits the same
> tool access — read the depth limit section below before opening write gates broadly.

---

#### Delegation Method Comparison

| Method | User command | Tool call | System prompt | Chat history | Result in history | Gate |
|---|---|---|---|---|---|---|
| `@model` prefix | `@gpt5m <prompt>` | — | ✓ current session | ✓ full | ✓ yes | none (admin-initiated) |
| `at_llm` | — | `at_llm(model, prompt)` | ✓ current session | ✓ full | ✗ no | write gate |
| `llm_clean_text` | `!llm_clean_text <model> <prompt>` | `llm_clean_text(model, prompt)` | ✗ none | ✗ none | ✗ no | rate limit only |
| `llm_clean_tool` | `!llm_clean_tool <model> <tool> <args>` | `llm_clean_tool(model, tool, arguments)` | ✗ none (tool schema only) | ✗ none | ✗ no | rate limit only |
| `agent_call` | — | `agent_call(agent_url, message)` | ✓ remote instance's own | ✓ remote session's own | ✗ no | rate limited, depth guarded |

---

#### `@model` — Per-Turn Model Switch (user only)

Prefix any message with `@ModelName` to use a different model for that one turn. All gates are
bypassed — this is an explicit admin action. The result is added to shared session history and the
original model is restored afterward.

```
@gpt5m summarize what we discussed so far
@gemini25 what is 2+2       ← same model just bypasses gates for this turn
```

---

#### `at_llm` — Full-Context LLM Delegation (tool only)

Passes the **full current session context** (assembled system prompt + complete chat history +
the given prompt as a new user turn) to the named model and returns its response. The result is
**not** added to session history, matching `@model` semantics but invocable as a tool call.

All tool gates are bypassed for the called model (same as `@model`). Subject to the `llm_call`
rate limit bucket.

```
# LLM tool call:
at_llm(model="gpt5m", prompt="Review the last tool result and suggest improvements.")
```

**Gate:** write-gated — controlled by `!at_llm_gate_write <true|false>` (default: gated).
Enable with `!at_llm_gate_write false` to allow without approval.

---

#### `llm_clean_text` — Stateless LLM Call

Calls a target model with **no context at all** — no system prompt, no history, no tool
definitions. The target model sees only the literal prompt string. The response is returned
directly to the calling LLM or printed in the shell; it is never added to history.

Use for: offloading summarization, translation, reformatting, or analysis of data you embed in
the prompt itself. Only models with `tool_call_available=true` may be called.

```
# User command:
!llm_clean_text nuc11Local Summarize: "The quick brown fox..."

# LLM tool call:
llm_clean_text(model="nuc11Local", prompt="Summarize the following text: ...")
```

**Rate limited:** default 3 calls per 20 seconds (shared with `llm_clean_tool` and `at_llm`).
No gate — always allowed when within rate limits.

---

#### `llm_clean_tool` — Stateless Tool Delegation

Delegates a **single named tool call** to a target model with minimal context. The target model
receives only that tool's schema as its system prompt and the `arguments` string as the user
message. The server executes the resulting tool call (normal gates apply), and the tool result is
synthesized by the target model before returning the final text.

Use for: offloading tool execution to a local/free model to save frontier API quota.

```
# User command:
!llm_clean_tool nuc11Local url_extract https://example.com summarize the main points

# LLM tool call:
llm_clean_tool(model="nuc11Local", tool="url_extract", arguments="https://example.com summarize main points")
```

**Rate limited:** same bucket as `llm_clean_text`. No gate — always allowed when within limits.

---

#### `agent_call` — Remote Agent Delegation (tool only)

Sends a single message to another running agent-mcp instance and returns its response. The remote
agent runs with its **own session context** (its own system prompt, its own history) — it receives
only the message string. By default, remote tokens are streamed in real-time via `push_tok`.

Use for: multi-agent swarms, parallel task execution, cross-instance verification.

```
# LLM tool call:
agent_call(agent_url="http://192.168.1.50:8765", message="Search for recent Python 3.13 release notes.")
```

**Rate limited:** default 5 calls per 60 seconds. Subject to depth guard (see below).

---

#### LLM Tool Call Enablement

Not all models are permitted to use tool calls by default. Control which models can use
`llm_clean_text`, `llm_clean_tool`, and `at_llm` via:

```
!llm_call                       list models with tool_call_available status
!llm_call <model> <true|false>  enable/disable for a specific model
!llm_call <true|false>          set for ALL models
```

---

### Delegation Depth Limits

Unconstrained delegation chains can grow multiplicatively. Each `at_llm` call gets a **fresh**
tool loop counter (up to `MAX_TOOL_ITERATIONS=10`), so an unbounded chain of depth N could
execute up to 10^N tool calls. `agent_call` chains have the same issue across instances.

Depth limits are enforced via session-stored counters. When the limit is reached, the delegation
is rejected with an instructive message telling the LLM not to retry.

#### `max_at_llm_depth`

Controls how many nested `at_llm` calls are allowed within a single session turn.

- **1 (default):** The LLM can call `at_llm` once, but the called model cannot itself call
  `at_llm` again. No recursion.
- **2+:** Allows chaining — model A calls model B which calls model C. Each hop multiplies the
  maximum tool iterations.
- **0:** Disables `at_llm` entirely (every call is rejected immediately).

> **Warning:** Setting this above 1 allows recursive model chains. With `max_at_llm_depth=3`
> and `MAX_TOOL_ITERATIONS=10`, a worst-case chain could issue 10³ = 1,000 tool calls in a
> single session turn. Keep this at 1 unless you have a specific controlled use case.

#### `max_agent_call_depth`

Controls how many nested `agent_call` hops are allowed from a given session.

- **1 (default):** The orchestrator can call a remote agent, but that remote agent cannot itself
  call `agent_call` back. The remote agent responds directly.
- **2+:** Allows multi-hop swarms (A → B → C). Each hop is a separate instance with its own
  tool loop.
- **0:** Disables `agent_call` entirely.

> **Warning:** Multi-hop swarms are difficult to observe and kill. Remote instances do not share
> your gate state, so a delegated model at hop 2 may operate with fewer constraints than expected.
> Keep this at 1 for controlled swarms where you are the explicit orchestrator.

#### Kill Switch

If a runaway delegation chain occurs, switch models in shell.py:

```
!model <any_model>
```

This calls `cancel_active_task()` which propagates `CancelledError` through all nested `at_llm`
and `agent_call` awaits in the current coroutine chain, terminating the entire tree.

#### Managing Limits via `agentctl.py`

View and update limits from the command line (requires agent restart to take effect):

```bash
python agentctl.py limit-list                         # show current values
python agentctl.py limit-set max_at_llm_depth 1       # set at_llm depth
python agentctl.py limit-set max_agent_call_depth 1   # set agent_call depth
```

Limits are stored in the `"limits"` section of `llm-models.json`:

```json
"limits": {
  "max_at_llm_depth": 1,
  "max_agent_call_depth": 1
}
```

#### Managing Limits at Runtime (shell.py / tool calls)

View and update limits without restarting the agent. Changes persist to `llm-models.json` but
only affect new sessions after restart.

```
!limit_list                              show all limits with current values
!limit_set max_at_llm_depth 2            set at_llm depth limit
!limit_set max_agent_call_depth 1        set agent_call depth limit
```

These commands are also available as gated LLM tool calls:
- `limit_list()` — read gate (controlled by `!limit_list_gate_read`, default: gated)
- `limit_set(key, value)` — write gate (controlled by `!limit_set_gate_write`, default: gated)

Set gate defaults for startup via `agentctl.py gate-set`:

```bash
python agentctl.py gate-set limit_list read true     # auto-allow reads
python agentctl.py gate-set limit_set write false    # keep writes gated
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
| `plugins-enabled.json` | Which plugins are active + per-plugin config + rate limits | `agentctl.py` or direct edit |
| `plugin-manifest.json` | Plugin registry — metadata, deps, env vars (read-only) | Plugin authors only |
| `llm-models.json` | Model registry (enabled, model_id, type, etc.) | `agentctl.py` |
| `.system_prompt` | Root system prompt file | Admin manually or LLM via tool |
| `.system_prompt_*` | Individual section files | Admin manually or LLM via tool |
| `.aiops_session_id` | shell.py session persistence | shell.py automatically |

### `plugin-manifest.json` vs `plugins-enabled.json`

These two files have distinct, non-overlapping roles:

**`plugin-manifest.json` — the plugin catalog (read-only)**

Declares that a plugin *exists* and what it needs to run: its Python file, type,
pip dependencies, required `.env` variables, and load priority.  This file is
maintained by plugin authors and committed to the repo.  The agent and
`agentctl.py` read it purely for validation — to check whether a plugin's
dependencies and credentials are present before attempting to load it.  You never
edit this file to enable or disable plugins.

**`plugins-enabled.json` — the operator control panel (read/write)**

Determines what actually runs.  It has three jobs:

1. **`enabled_plugins` list** — the ordered list of plugin names the agent will
   attempt to load at startup.  Add a plugin here to activate it; remove it to
   deactivate it entirely.  Managed by `agentctl.py enable/disable` or by
   direct edit.

2. **`plugin_config` blocks** — per-plugin runtime settings such as port, host,
   and the `enabled` flag.  The `enabled: false` pattern lets you keep a plugin
   in `enabled_plugins` (preserving its config) without starting it.  This is how
   `plugin_proxy_llama` and `plugin_client_slack` ship: configured but off until
   you flip `"enabled": true` or run `agentctl.py enable <plugin>`.

3. **`rate_limits`** and **`default_model`** — server-wide settings also stored here.

**Practical rule:** to enable or disable a plugin, always use `agentctl.py`
or edit `plugins-enabled.json`.  Never add enable/disable logic to `plugin-manifest.json`.

**Fresh installs:** `setup-agent-mcp.sh` clones the repo and copies credentials
(`.env`, `credentials.json`, `llm-models.json`) from a reference installation.
It intentionally does *not* copy `plugins-enabled.json` — the repo's version is
the authoritative default for new installs, and port assignments are adjusted
per-instance afterward with `agentctl.py port-set`.

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
