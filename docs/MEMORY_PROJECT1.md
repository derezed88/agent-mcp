# Tiered Memory System — Project 1

Automatic, topic-aware, tiered memory with cross-session recall for the Samaritan agent persona.

---

## Overview

By default, LLM sessions are stateless — each `!reset` or reconnect starts from a blank slate. This feature adds a persistent memory layer beneath all sessions: facts distilled from conversations survive resets, accumulate across weeks of use, and are automatically injected back into future requests with zero manual overhead.

```
Session ends (!reset)
        │
        ▼
[summarizer-anthropic] ──extracts──► JSON facts (topic, content, importance)
        │
        ▼
samaritan_memory_shortterm  ◄── auto-injected into every request (≤400 tokens)
        │
        │  after 48h (low-importance rows)
        ▼
samaritan_memory_longterm   ◄── on-demand recall via tool call
        │
        │  future
        ▼
Google Drive archive        ◄── bulk cold storage (not yet built)
```

---

## Dependencies

### Software

| Dependency | Purpose | Install |
|---|---|---|
| Python 3.11+ | Runtime | System |
| agent-mcp | Agent server framework | `git clone` + `pip install -r requirements.txt` |
| MySQL / MariaDB | Short-term and long-term memory storage | System |
| `aiomysql` | Async MySQL driver | In `requirements.txt` |
| `langchain-openai` | LLM dispatch (OpenAI-compatible) | In `requirements.txt` |
| `langchain-google-genai` | LLM dispatch (Gemini) | In `requirements.txt` |

### API Keys Required

| Key | Used For | Env Var |
|---|---|---|
| Anthropic | `summarizer-anthropic` (Claude Haiku) | `ANTHROPIC_API_KEY` |
| xAI | `samaritan-reasoning` (Grok) | `XAI_API_KEY` |
| OpenAI | `samaritan-execution` (gpt-4o-mini) | `OPENAI_API_KEY` |

Gemini (`GEMINI_API_KEY`) is optional — `summarizer-gemini` is a fallback summarizer.

### Files Added / Modified

| File | Role |
|---|---|
| `memory.py` | Core memory module — all read/write/age/summarize logic; `_parse_table()` fixed for pipe-separated output |
| `database.py` | Added `execute_insert()` — returns `cursor.lastrowid` in same connection (fixes LAST_INSERT_ID race) |
| `agents.py` | `auto_enrich_context()` injects short-term block; loop guard fixed (resolves tool_calls before HumanMessage injection); threshold 2→3 |
| `routes.py` | `cmd_reset()` triggers summarize-before-clear |
| `tools.py` | Memory tools registered in both `CORE_LC_TOOLS` and `core_executors` (previously only in CORE_LC_TOOLS); `memory` toolset added |
| `llm-models.json` | `samaritan-reasoning`: `memory` toolset added, temp 0.7→0.2, top_p 0.9→0.7; `samaritan-execution` unchanged |
| `llm-tools.json` | `"memory"` toolset added |
| `db-config.json` | Instance-specific (gitignored): database name + table names; loaded by `memory.py` and `database.py` at startup |
| `system_prompt/004_reasoning/.system_prompt_memory` | Rewritten: direct tool call instructions + CRITICAL hallucination warning |
| `system_prompt/004_execution/.system_prompt_memory` | Removed hardcoded table names |

---

## Database Setup

Run once against your `agent_mcp` database:

```sql
CREATE TABLE samaritan_memory_shortterm (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  topic         VARCHAR(255) NOT NULL,
  content       TEXT NOT NULL,
  importance    TINYINT DEFAULT 5 COMMENT '1=low 5=med 10=critical',
  source        ENUM('session','user','directive') DEFAULT 'session',
  session_id    VARCHAR(255),
  created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_topic (topic),
  INDEX idx_importance (importance DESC),
  INDEX idx_created (created_at)
);

CREATE TABLE samaritan_memory_longterm (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  topic         VARCHAR(255) NOT NULL,
  content       TEXT NOT NULL,
  importance    TINYINT DEFAULT 5,
  source        ENUM('session','user','directive') DEFAULT 'session',
  session_id    VARCHAR(255),
  shortterm_id  INT COMMENT 'original shortterm row id',
  created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  aged_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_topic (topic),
  INDEX idx_importance (importance DESC)
);

CREATE TABLE samaritan_chat_summaries (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  session_id    VARCHAR(255) NOT NULL,
  summary       TEXT NOT NULL,
  message_count INT DEFAULT 0,
  model_used    VARCHAR(100),
  created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_session (session_id),
  INDEX idx_created (created_at)
);
```

---

## Configuration

### `llm-models.json` — Model Roles and Parameters

The memory system depends on three models working together. Here is the relevant configuration and the significance of each parameter.

#### `samaritan-reasoning` — The Reasoning Brain (Grok)

```json
"samaritan-reasoning": {
  "model_id": "grok-4-1-fast-non-reasoning",
  "type": "OPENAI",
  "host": "https://api.x.ai/v1",
  "env_key": "XAI_API_KEY",
  "max_context": 5000,
  "llm_tools": ["get_system_info", "llm_call", "llm_list", "search_tavily", "search_xai", "memory"],
  "system_prompt_folder": "system_prompt/004_reasoning",
  "temperature": 0.2,
  "top_p": 0.7,
  "token_selection_setting": "custom"
}
```

| Parameter | Value | Why |
|---|---|---|
| `model_id` | `grok-4-1-fast-non-reasoning` | Grok without chain-of-thought overhead; better for chat turns |
| `llm_tools` | minimal set + `memory` | Memory tools added so Grok can call them directly if it issues a proper tool call |
| `temperature` | `0.2` | Lowered from 0.7 to encourage tool call generation over narration |
| `top_p` | `0.7` | Tightened from 0.9; reduces divergent token selection during tool decisions |
| `token_selection_setting` | `"custom"` | Applies the temperature/top_p above (vs. `"default"` which ignores them) |
| `max_context` | `5000` | Short-term memory injection + conversation fits comfortably |
| `system_prompt_folder` | `004_reasoning` | Loads the memory-aware Samaritan prompt tree |

> **Grok tool-calling caveat:** `grok-4-1-fast-non-reasoning` was observed to narrate tool execution
> (write text claiming memory was saved) instead of issuing an actual tool call, even at temperature=0.2.
> The `memory` toolset is bound so that *if* Grok does issue a tool call it will succeed. For reliability,
> explicit memory saves should be delegated to `samaritan-execution` via `llm_call`. The system prompt
> includes a CRITICAL warning: never claim a save without the tool call appearing in the response.

#### `samaritan-execution` — The Obedient Executor (gpt-4o-mini)

```json
"samaritan-execution": {
  "model_id": "gpt-4o-mini",
  "type": "OPENAI",
  "host": "https://api.openai.com/v1",
  "env_key": "OPENAI_API_KEY",
  "max_context": 100000,
  "llm_tools": ["core", "db", "drive", "search", "memory"],
  "system_prompt_folder": "system_prompt/004_execution",
  "temperature": 0.1,
  "top_p": 0.5,
  "token_selection_setting": "custom"
}
```

| Parameter | Value | Why |
|---|---|---|
| `model_id` | `gpt-4o-mini` | Most reliable tool caller tested; follows instructions precisely |
| `llm_tools` | `["core","db","drive","search","memory"]` | Full tool access including the memory toolset |
| `temperature` | `0.1` | Near-deterministic; no creative variation in tool execution |
| `top_p` | `0.5` | Aggressive nucleus cutoff — only the most likely tokens |
| `token_selection_setting` | `"custom"` | Enforces the low temperature/top_p |
| `max_context` | `100000` | Can handle large delegation prompts from Grok |

#### `summarizer-anthropic` — The Memory Distiller (Claude Haiku)

```json
"summarizer-anthropic": {
  "model_id": "claude-haiku-4-5-20251001",
  "type": "OPENAI",
  "host": "https://api.anthropic.com/v1",
  "env_key": "ANTHROPIC_API_KEY",
  "max_context": 100000,
  "llm_tools": ["get_system_info", "drive", "vscode"],
  "system_prompt_folder": "system_prompt/003_claudeVSCode",
  "token_selection_setting": "default"
}
```

| Parameter | Value | Why |
|---|---|---|
| `model_id` | `claude-haiku-4-5-20251001` | Fast, cheap, accurate at structured JSON extraction |
| `max_context` | `100000` | Can handle long conversation histories without truncation |
| `llm_tools` | minimal | Summarizer only reads and writes; no tool execution needed |
| `token_selection_setting` | `"default"` | Haiku's defaults are well-tuned for extraction tasks |

### `llm-tools.json` — The Memory Toolset

```json
"memory": [
  "memory_save",
  "memory_recall",
  "memory_age"
]
```

This toolset is assigned to both `samaritan-execution` and `samaritan-reasoning`. `samaritan-execution` reliably calls these tools. `samaritan-reasoning` (Grok) has them bound so a direct call succeeds if Grok chooses to issue one; in practice Grok often narrates rather than calls, so delegation via `llm_call` remains the reliable path.

---

## System Prompt Tree (`system_prompt/004_reasoning/`)

The prompt is a tree of files. The root `.system_prompt` assembles the tree via `[SECTIONS]` markers. Each section entry **must** use the format `name: description` — bare names are silently skipped by the parser.

```
.system_prompt                          ← root: identity + prime directives
  ├── tools: Available tool definitions
  │     ├── tool_get_system_info
  │     ├── tool_db_query
  │     ├── tool_google_drive
  │     ├── tool_search_tavily
  │     ├── tool_search_xai
  │     ├── tool_llm_list
  │     └── tool_llm_delegation         ← describes llm_call delegation pattern
  ├── behavior: Behaviour and delegation rules
  └── continuity: Cross-session memory and continuity
        ├── continuity_tiered           ← Tier 1/2 startup SQL + decision logging
        └── memory                      ← memory save/recall/age procedures
```

### Memory-Relevant Sections

**`.system_prompt_behavior`** — tells Grok how to behave:
- Rule 6: Delegate DB writes and tool chains to `samaritan-execution` via `llm_call`
- Rule 7: Treat injected `## Active Memory` as ground truth; do not ask about known facts

**`.system_prompt_continuity`** — describes the three-tier model and what triggers each tier

**`.system_prompt_memory`** — detailed procedures:
- When to save explicitly (user preference, key decision, "remember this")
- How to trigger long-term recall via delegation
- How to trigger memory aging

**`.system_prompt_continuity_tiered`** — startup SQL for loading tasks/initiatives/assets from other `samaritan_*` tables (separate from the memory tables; used for operational continuity)

### What Triggers Memory Features

| Trigger | What Happens |
|---|---|
| Any request (≥1 memory row, importance≥3) | `auto_enrich_context()` prepends `## Active Memory` block to system message |
| `!reset` with ≥4 messages in history | `cmd_reset()` calls `summarize_and_save()` before clearing |
| Grok sees user preference or key decision | Behavior rule 6: delegates `memory_save` call to `samaritan-execution` |
| User says "remember this" | Grok delegates explicit save with importance=8-10 |
| Topic present but not in active memory | Grok delegates `memory_recall(topic, tier="long")` to `samaritan-execution` |
| Session created or rehydrated from disk | `age_to_longterm()` runs as background task — rows >48h moved to long-term |

---

## Runtime: How Memory Flows

### Request Flow (Every Conversation Turn)

```
User message arrives
        │
        ▼
dispatch_llm() called
        │
        ▼
auto_enrich_context()
  ├── load_context_block(limit=15, min_importance=3)
  ├── queries samaritan_memory_shortterm
  └── if rows exist → injects "## Active Memory" system message before user message
        │
        ▼
agentic_lc() — LLM called with enriched context
  └── Grok sees memory block; responds with continuity
```

Token cost of injection: **~200–400 tokens** for 15 rows, regardless of how many sessions
have accumulated in storage.

### Reset Flow (Summarize → Save → Clear)

```
User sends !reset
        │
        ▼
cmd_reset() in routes.py
  ├── if history ≥ 4 messages:
  │     ├── push "[memory] Summarizing session to memory..."
  │     ├── summarize_and_save(session_id, history, "summarizer-anthropic")
  │     │     ├── builds condensed history text (last 60 turns, skip tool messages)
  │     │     ├── calls _call_llm_text("summarizer-anthropic", extraction_prompt)
  │     │     ├── parses JSON → [{topic, content, importance}, ...]
  │     │     ├── calls save_memory() for each fact → samaritan_memory_shortterm
  │     │     └── INSERT INTO samaritan_chat_summaries (full raw summary)
  │     └── push "[memory] Summarized N messages → M memories saved."
  ├── session["history"] = []
  └── push "Conversation history cleared."
```

### Delegation Flow (Grok → gpt-4o-mini)

The preferred path for memory writes: Grok delegates to `samaritan-execution` via `llm_call`.

```
Grok reasoning turn
  └── llm_call("samaritan-execution", "save memory: topic=X content=Y importance=8")
            │
            ▼
      samaritan-execution (gpt-4o-mini) receives prompt
        └── calls memory_save(topic="X", content="Y", importance=8)
                │
                ▼
          INSERT INTO samaritan_memory_shortterm
```

gpt-4o-mini at temp=0.1 reliably translates natural-language delegation prompts into precise tool calls.

### Direct Memory Call (Grok → tool)

Grok also has the `memory` toolset bound directly. If it issues a proper tool call (not just narrates), it succeeds:

```
Grok reasoning turn
  └── memory_save(topic="X", content="Y", importance=8)   ← direct tool call
            │
            ▼
      _memory_save_exec() in tools.py
        └── save_memory() in memory.py
                │
                ▼
          INSERT INTO samaritan_memory_shortterm
```

**Reliability note:** `grok-4-1-fast-non-reasoning` frequently generates text narrating a tool call instead of issuing one. Direct calls succeed when they happen; delegation is the more reliable path.

### Aging Flow

Rows older than 48 hours with low importance are moved to `samaritan_memory_longterm`.

Aging fires automatically as a background task whenever a session is created or rehydrated from
disk (including after an idle reap/reconnect). No manual trigger or cron job is needed.

```
Client connects (new session or rehydrated after idle reap)
        │
        ▼
routes.py session init block
  └── asyncio.create_task(age_to_longterm())   ← non-blocking background task
            │
            ▼
      memory_age(older_than_hours=48, max_rows=100)
        ├── SELECT from shortterm WHERE created_at < NOW() - INTERVAL 48 HOUR
        │   ORDER BY importance ASC (lowest first)
        ├── for each row:
        │     ├── INSERT INTO longterm (copying all fields + shortterm_id)
        │     └── DELETE FROM shortterm WHERE id = row.id
        └── returns count of rows moved (logged only)
```

Manual override: ask Grok to delegate `memory_age(older_than_hours=24)` to age more aggressively.

---

## Use Cases

### 1. Preference Persistence

**Without memory:** Every new session needs re-establishing: "I use Python 3.11, always use f-strings, prefer concise responses."

**With memory:**
```
Session 3: "Always use type hints in code examples."
!reset

Session 4: [Active Memory already contains: "Prefers type hints in code examples (imp=7)"]
Grok uses type hints without being asked.
```

---

### 2. Project Continuity

```
Session 7: Long debug session on agent-mcp — found that Grok loops on tool calls,
           fixed with fingerprint detection.
!reset

→ [memory] 5 memories saved:
  technical-decisions: "Grok tool-call loop fixed via fingerprint detection in agents.py"
  project-status: "Memory system build complete, 18/18 tests passing"

Session 8: "Where did we leave off?"
Grok: "Last session: completed the tiered memory system. Key decision:
       Grok reasons, gpt-4o-mini executes. Both tested and working."
```

---

### 3. Explicit Important Fact

```
User: "Remember this: the production API rotates keys every 90 days, next rotation April 15."

Grok → delegates:
  llm_call("samaritan-execution",
    "save memory: topic=security, content=prod API key rotates every 90 days next April 15, importance=9")

Confirmed: memory saved (imp=9). Will appear in every future session.
```

---

### 4. Long-Term Recall

```
User: "What was the decision about the Hermes model?"
[Not in active short-term memory — too old, aged out]

Grok → delegates:
  llm_call("samaritan-execution", "recall memories about: Hermes tier=long")

← "Hermes-3-Llama-3.1-8B tested on nuc11, rejected — stalls after listing files,
   cannot self-chain tool calls. Decision: use Qwen2.5-7B instead."

Grok answers with full context from 3 weeks ago.
```

---

### 5. Keeping Short-Term Lean

After many sessions, short-term fills up. Trigger aging to keep injection fast:

```
!db_query SELECT COUNT(*) FROM samaritan_memory_shortterm
→ 87 rows

Ask Grok: "Age memories older than 24 hours."
Grok → delegates: memory_age(older_than_hours=24, max_rows=200)
→ "Aged 62 memories from short-term to long-term (threshold: 24h)."

!db_query SELECT COUNT(*) FROM samaritan_memory_shortterm
→ 25 rows  ← injection stays fast
```

---

### 6. Viewing What's Remembered

```
!db_query SELECT topic, content, importance FROM samaritan_memory_shortterm ORDER BY importance DESC LIMIT 10
!db_query SELECT session_id, message_count, created_at FROM samaritan_chat_summaries ORDER BY created_at DESC LIMIT 5
```

---

## `memory.py` Public API

```python
# Save a fact to short-term memory
await save_memory(topic, content, importance=5, source="session", session_id="")

# Load hot memories as list of dicts (importance DESC)
rows = await load_short_term(limit=20, min_importance=1)

# Move old rows from short-term to long-term; returns count moved
moved = await age_to_longterm(older_than_hours=48, max_rows=100)

# Get formatted string ready to inject into a system message
block = await load_context_block(limit=15, min_importance=3)

# Summarize a conversation history and save extracted facts to short-term
status = await summarize_and_save(session_id, history, model_key="summarizer-anthropic")
```

---

## Tool Reference

### `memory_save`
Save an explicit fact to short-term memory.

```
memory_save(
  topic      = "user-preferences",     # short label; groups display
  content    = "Prefers dark mode.",    # one concise sentence
  importance = 7,                       # 1=low, 5=medium, 10=critical
  source     = "user"                   # "user" | "session" | "directive"
)
```

### `memory_recall`
Query stored memories by topic and tier.

```
memory_recall(
  topic = "security",   # keyword filter on topic column (LIKE %topic%)
  tier  = "short",      # "short" (default) or "long"
  limit = 20
)
```

### `memory_age`
Move old short-term rows to long-term.

```
memory_age(
  older_than_hours = 48,    # move rows older than this
  max_rows         = 100    # cap per invocation
)
```

---

## Bugs Discovered and Fixed (2026-03-01)

All three bugs were independently silent — no exceptions were raised; the system appeared functional while saving nothing.

### Bug 1: Memory tools missing from `get_tool_executor()` — main culprit

**File:** `tools.py` — `get_tool_executor()`

`memory_save`, `memory_recall`, and `memory_age` were registered in `CORE_LC_TOOLS` (so the LLM could see them and issue calls) but were absent from the `core_executors` dict that `get_tool_executor()` uses to dispatch calls.

Every tool call returned `"Unknown tool: memory_save"` as a ToolMessage result. The model (gpt-4o-mini) retried with identical args → loop guard fired → zero rows saved, while the model responded "Memory saved successfully."

**Fix:** Added the three tools to `core_executors` in `get_tool_executor()`.

```python
# Before (missing):
core_executors = { 'get_system_info': ..., 'llm_call': ..., ... }

# After:
core_executors = {
    ...
    'memory_save':   _memory_save_exec,
    'memory_recall': _memory_recall_exec,
    'memory_age':    _memory_age_exec,
}
```

### Bug 2: `_parse_table()` split on wrong delimiter

**File:** `memory.py` — `_parse_table()`

`_parse_table()` split header and data lines on `\t` (tab character), but `execute_sql()` returns pipe-separated output:

```
id | topic         | content          | importance
---+---------------+------------------+-----------
16 | schedule      | Lee has Mon off  | 7
```

Every `load_short_term()` and `load_long_term()` call returned rows where all fields were `None`. Memory recall always said "No memories found" even when rows existed in the DB.

**Fix:** Changed delimiters from `\t` to `|` and added a separator-line filter:

```python
# Before:
headers = [h.strip() for h in lines[0].split("\t")]
vals = line.split("\t")

# After:
headers = [h.strip() for h in lines[0].split("|")]
if set(line.strip()) <= set("-+"):  # skip ---+--- separator lines
    continue
vals = line.split("|")
```

### Bug 3: `LAST_INSERT_ID()` race condition in `save_memory()`

**File:** `database.py` + `memory.py`

`save_memory()` ran the INSERT and `SELECT LAST_INSERT_ID()` as two separate `execute_sql()` calls. Each call opens and closes a fresh DB connection. `LAST_INSERT_ID()` is connection-scoped — calling it on a new connection always returns 0.

Effect: the second `memory_save` call in any session always returned `row_id=0`, which the dedup check treated as "already exists". The first call saved correctly (row was actually inserted) but reported id=0, then the retry on the second call hit the dedup check and silently skipped.

**Fix:** New `execute_insert()` in `database.py` that returns `cursor.lastrowid` before closing the connection:

```python
def _run_insert(sql: str) -> int:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(sql)
    conn.commit()
    return cursor.lastrowid or 0   # same connection — correct value

async def execute_insert(sql: str) -> int:
    return await asyncio.to_thread(_run_insert, sql)
```

### Loop guard: OpenAI 400 on HumanMessage injection

**File:** `agents.py` — `agentic_lc()`

When the loop guard fired (same tool+args repeated `_TOOL_LOOP_THRESHOLD` times), it injected a `HumanMessage("stop, answer now")` into the context before resolving the current turn's `ai_msg.tool_calls`. OpenAI requires every `tool_calls` in an AIMessage to be followed by corresponding `ToolMessage` entries; the bare HumanMessage caused a 400 error.

**Fix:** Execute all pending tool calls (add ToolMessages to ctx) before injecting the HumanMessage break. Threshold also raised from 2→3 to allow one retry after a legitimate dedup/no-op result.

### Hallucination guard: system prompt CRITICAL warning

**File:** `system_prompt/004_reasoning/.system_prompt_memory`

Grok was narrating saves without calling the tool. Added an explicit CRITICAL instruction:

```
**CRITICAL**: You MUST actually invoke the `memory_save` tool. Never claim memory was saved
without having called the tool. If the tool call does not appear in your response, memory
was NOT saved.
```

This is an instruction, not a technical fix — Grok may still narrate. The reliable path remains delegation to `samaritan-execution`.

---

## What's Not Yet Built

Priority = functionality gain ÷ implementation complexity. P1 = high value, low effort. P4 = low value or high complexity.

| Priority | Feature | Status | Effort | Notes |
|---|---|---|---|---|
| P1 | Memory confirmation UX | **Built** | Low | `[memory] Summarized N messages → M memories saved, K duplicate(s) skipped.` pushed after every reset. |
| P1 | Deduplication | **Built** | Low | `save_memory()` checks both shortterm and longterm for identical topic+content before inserting. Duplicate returns 0 without touching the DB. |
| P2 | Importance decay | Not built | Low-Med | Scheduled SQL: `UPDATE shortterm SET importance = importance - 1 WHERE importance > 3 AND created_at < NOW() - INTERVAL 7 DAY`. Could run at session-start alongside aging. Prevents stale high-imp facts from crowding injection. |
| P2 | Per-topic retention policies | Not built | Med | JSON config mapping topic patterns → `age_after_hours` overrides. Requires modifying `age_to_longterm()` to apply per-row policy instead of a single threshold. High value for operational use but needs schema for the config. |
| P3 | Google Drive archival | Not built | Med | `memory_age` extended to optionally export aged rows to a Drive file before deletion. Google Drive plugin already exists — mainly plumbing. Low urgency since longterm table handles this well enough. |
| P3 | Drive → short-term reload | Not built | Med | Inverse of archival: parse Drive export back into shortterm. Depends on archival being built first; blocked on P3 above. |
| P4 | Semantic/vector search | Not built | High | Requires embedding model, vector store (pgvector or Chroma), and rewrite of `memory_recall`. Significant infra lift. Only matters once memory grows large enough that `LIKE '%topic%'` misses things. |
