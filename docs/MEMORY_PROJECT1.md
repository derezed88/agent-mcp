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
| `memory.py` | Core memory module — all read/write/age/summarize logic |
| `agents.py` | Modified: `auto_enrich_context()` injects short-term block; `_call_llm_text()` added |
| `routes.py` | Modified: `cmd_reset()` triggers summarize-before-clear |
| `tools.py` | Modified: three new tools registered (`memory_save`, `memory_recall`, `memory_age`) |
| `llm-models.json` | Modified: `samaritan-reasoning` and `samaritan-execution` rewired |
| `llm-tools.json` | Modified: `"memory"` toolset added |
| `system_prompt/004_reasoning/` | Modified: behavior, continuity, memory sections updated |

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
  "llm_tools": ["get_system_info", "llm_call", "llm_list", "search_tavily", "search_xai"],
  "system_prompt_folder": "system_prompt/004_reasoning",
  "temperature": 0.7,
  "top_p": 0.9,
  "token_selection_setting": "custom"
}
```

| Parameter | Value | Why |
|---|---|---|
| `model_id` | `grok-4-1-fast-non-reasoning` | Grok without chain-of-thought overhead; better for chat turns |
| `llm_tools` | minimal set | Grok is inconsistent at tool calling; it reasons and delegates, not executes |
| `temperature` | `0.7` | Creative and expressive for natural conversation |
| `top_p` | `0.9` | Slight nucleus sampling to prevent repetition |
| `token_selection_setting` | `"custom"` | Applies the temperature/top_p above (vs. `"default"` which ignores them) |
| `max_context` | `5000` | Short-term memory injection + conversation fits comfortably |
| `system_prompt_folder` | `004_reasoning` | Loads the memory-aware Samaritan prompt tree |

**Grok does not write to the database.** It reads injected memory and delegates tool execution to `samaritan-execution` via `llm_call`.

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

This toolset is assigned to `samaritan-execution` in its `llm_tools` array. It is **not** assigned to `samaritan-reasoning` — Grok does not call memory tools directly; it delegates to the execution model.

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

When Grok needs to write a memory or recall long-term facts:

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

gpt-4o-mini at temp=0.1 reliably translates natural-language delegation prompts into precise tool calls. Grok never touches the DB directly.

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

## What's Not Yet Built

Priority = functionality gain ÷ implementation complexity. P1 = high value, low effort. P4 = low value or high complexity.

| Priority | Feature | Status | Effort | Notes |
|---|---|---|---|---|
| P1 | Memory confirmation UX | Not built | Low | Push "[memory] saved: topic=X (imp=8)" after each reset — one line in `summarize_and_save()`. Immediately visible value with near-zero complexity. |
| P1 | Deduplication | Not built | Low | `INSERT … ON DUPLICATE KEY UPDATE` or a pre-save `SELECT` by topic+content hash. Repeated resets accumulate noise fast; easy SQL fix. |
| P2 | Importance decay | Not built | Low-Med | Scheduled SQL: `UPDATE shortterm SET importance = importance - 1 WHERE importance > 3 AND created_at < NOW() - INTERVAL 7 DAY`. Could run at session-start alongside aging. Prevents stale high-imp facts from crowding injection. |
| P2 | Per-topic retention policies | Not built | Med | JSON config mapping topic patterns → `age_after_hours` overrides. Requires modifying `age_to_longterm()` to apply per-row policy instead of a single threshold. High value for operational use but needs schema for the config. |
| P3 | Google Drive archival | Not built | Med | `memory_age` extended to optionally export aged rows to a Drive file before deletion. Google Drive plugin already exists — mainly plumbing. Low urgency since longterm table handles this well enough. |
| P3 | Drive → short-term reload | Not built | Med | Inverse of archival: parse Drive export back into shortterm. Depends on archival being built first; blocked on P3 above. |
| P4 | Semantic/vector search | Not built | High | Requires embedding model, vector store (pgvector or Chroma), and rewrite of `memory_recall`. Significant infra lift. Only matters once memory grows large enough that `LIKE '%topic%'` misses things. |
