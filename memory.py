"""
memory.py — Tiered memory system

Tiers:
  Short-term  : MySQL (hot, injected every request)
  Long-term   : MySQL (aged-out, on-demand recall)
  Archive     : Google Drive (bulk export, future)

Public API:
  save_memory(topic, content, importance, source, session_id)
  load_short_term(limit, min_importance) -> list[dict]
  age_to_longterm(older_than_hours, max_rows) -> int
  load_context_block(limit, min_importance) -> str   # ready to prepend to prompt
  summarize_and_save(session_id, history, model_key) -> str
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

from database import execute_sql

log = logging.getLogger("memory")

# ---------------------------------------------------------------------------
# Load table names from db-config.json (instance-specific, gitignored)
# ---------------------------------------------------------------------------

def _load_db_config() -> dict:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db-config.json")
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        log.warning("db-config.json not found — using default table names")
        return {}
    except Exception as e:
        log.warning(f"db-config.json load failed: {e} — using default table names")
        return {}

_db_cfg = _load_db_config()
_tables = _db_cfg.get("tables", {})
_ST  = _tables.get("memory_shortterm", "memory_shortterm")
_LT  = _tables.get("memory_longterm",  "memory_longterm")
_SUM = _tables.get("chat_summaries",   "chat_summaries")

# ---------------------------------------------------------------------------
# Short-term: save
# ---------------------------------------------------------------------------

async def save_memory(
    topic: str,
    content: str,
    importance: int = 5,
    source: str = "session",
    session_id: str = "",
) -> int:
    """Insert a new short-term memory row. Returns new row id, or 0 if duplicate/error."""
    topic = topic.replace("'", "''")[:255]
    content = content.replace("'", "''")
    session_id = (session_id or "").replace("'", "''")[:255]
    source = source if source in ("session", "user", "directive") else "session"
    importance = max(1, min(10, int(importance)))

    # Dedup: skip insert if identical topic+content exists in shortterm or longterm
    try:
        dup_check = (
            f"SELECT 1 FROM {_ST} "
            f"WHERE topic = '{topic}' AND content = '{content}' LIMIT 1"
        )
        dup_result = await execute_sql(dup_check)
        if dup_result.strip() and "1" in dup_result:
            return 0
        dup_check_lt = (
            f"SELECT 1 FROM {_LT} "
            f"WHERE topic = '{topic}' AND content = '{content}' LIMIT 1"
        )
        dup_result_lt = await execute_sql(dup_check_lt)
        if dup_result_lt.strip() and "1" in dup_result_lt:
            return 0
    except Exception as e:
        log.warning(f"save_memory dedup check failed: {e}")

    sql = (
        f"INSERT INTO {_ST} "
        f"(topic, content, importance, source, session_id) "
        f"VALUES ('{topic}', '{content}', {importance}, '{source}', '{session_id}')"
    )
    try:
        await execute_sql(sql)
        id_result = await execute_sql("SELECT LAST_INSERT_ID() AS new_id")
        for line in id_result.splitlines():
            if line.strip().isdigit():
                return int(line.strip())
        return 0
    except Exception as e:
        log.error(f"save_memory failed: {e}")
        return 0


# ---------------------------------------------------------------------------
# Short-term: load (returns list of dicts, most important first)
# ---------------------------------------------------------------------------

async def load_short_term(limit: int = 20, min_importance: int = 1) -> list[dict]:
    """Load recent short-term memories, highest importance first."""
    sql = (
        f"SELECT id, topic, content, importance, source, session_id, "
        f"created_at, last_accessed "
        f"FROM {_ST} "
        f"WHERE importance >= {min_importance} "
        f"ORDER BY importance DESC, created_at DESC "
        f"LIMIT {limit}"
    )
    try:
        raw = await execute_sql(sql)
        return _parse_table(raw)
    except Exception as e:
        log.error(f"load_short_term failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Long-term: load (returns list of dicts, most important first)
# ---------------------------------------------------------------------------

async def load_long_term(limit: int = 20, topic: str = "") -> list[dict]:
    """Load long-term memories, optionally filtered by topic substring."""
    where = f"WHERE topic LIKE '%{topic}%'" if topic else ""
    sql = (
        f"SELECT id, topic, content, importance, created_at "
        f"FROM {_LT} {where} "
        f"ORDER BY importance DESC, created_at DESC "
        f"LIMIT {limit}"
    )
    try:
        raw = await execute_sql(sql)
        return _parse_table(raw)
    except Exception as e:
        log.error(f"load_long_term failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Aging: move old low-importance rows to long-term
# ---------------------------------------------------------------------------

async def age_to_longterm(older_than_hours: int = 48, max_rows: int = 100) -> int:
    """
    Move short-term rows older than N hours to long-term.
    Returns number of rows moved.
    """
    select_sql = (
        f"SELECT * FROM {_ST} "
        f"WHERE created_at < NOW() - INTERVAL {older_than_hours} HOUR "
        f"ORDER BY importance ASC, created_at ASC "
        f"LIMIT {max_rows}"
    )
    try:
        raw = await execute_sql(select_sql)
        rows = _parse_table(raw)
        if not rows:
            return 0

        moved = 0
        for row in rows:
            rid = row.get("id", "")
            topic = row.get("topic", "").replace("'", "''")
            content = row.get("content", "").replace("'", "''")
            imp = row.get("importance", 5)
            src = row.get("source", "session")
            sid = row.get("session_id", "").replace("'", "''")

            insert = (
                f"INSERT INTO {_LT} "
                f"(topic, content, importance, source, session_id, shortterm_id) "
                f"VALUES ('{topic}', '{content}', {imp}, '{src}', '{sid}', {rid})"
            )
            await execute_sql(insert)
            await execute_sql(
                f"DELETE FROM {_ST} WHERE id = {rid}"
            )
            moved += 1

        return moved
    except Exception as e:
        log.error(f"age_to_longterm failed: {e}")
        return 0


# ---------------------------------------------------------------------------
# Context block: formatted string ready to inject into system prompt
# ---------------------------------------------------------------------------

async def load_context_block(limit: int = 15, min_importance: int = 3) -> str:
    """
    Return a formatted string of short-term memories for prompt injection.
    Returns empty string if no memories.
    """
    rows = await load_short_term(limit=limit, min_importance=min_importance)
    if not rows:
        return ""

    lines = ["## Active Memory (short-term recall)\n"]
    # Group by topic
    by_topic: dict[str, list[dict]] = {}
    for row in rows:
        t = row.get("topic", "general")
        by_topic.setdefault(t, []).append(row)

    for topic, items in by_topic.items():
        lines.append(f"**{topic}**")
        for item in items:
            imp = item.get("importance", 5)
            lines.append(f"  [imp={imp}] {item.get('content', '')}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summarize and save: call summarizer LLM on a history, store as memory
# ---------------------------------------------------------------------------

async def summarize_and_save(
    session_id: str,
    history: list[dict],
    model_key: str = "summarizer-anthropic",
) -> str:
    """
    Call a summarizer LLM on conversation history.
    Extracts topic-tagged memories and saves them to short-term.
    Also saves a chat summary row.
    Returns a status string.
    """
    if not history:
        return "No history to summarize."

    # Build condensed history text (skip tool calls to save tokens)
    lines = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content or role == "tool":
            continue
        if isinstance(content, list):
            # Extract text parts from structured content
            parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
            content = " ".join(parts)
        if content:
            lines.append(f"{role.upper()}: {content[:500]}")

    history_text = "\n".join(lines[-60:])  # last ~60 turns max
    if not history_text.strip():
        return "History had no text content to summarize."

    prompt = (
        "You are a memory distillation engine. Given this conversation, extract the most important facts, "
        "decisions, preferences, and context. Output ONLY valid JSON — a list of objects with keys: "
        "topic (short string), content (one concise sentence), importance (1-10 int). "
        "Topic examples: user-preferences, project-status, technical-decisions, security, tasks. "
        "Output 3-8 items maximum. No markdown, no explanation, just the JSON array.\n\n"
        f"CONVERSATION:\n{history_text}"
    )

    try:
        # Use llm_call infrastructure directly
        from agents import _call_llm_text
        result_text = await _call_llm_text(model_key, prompt)
    except Exception as e:
        log.error(f"summarize_and_save LLM call failed: {e}")
        result_text = None

    memories_saved = 0
    memories_skipped = 0
    if result_text:
        # Strip any markdown fences
        cleaned = result_text.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])

        try:
            items = json.loads(cleaned)
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    topic = str(item.get("topic", "general"))[:255]
                    content = str(item.get("content", ""))[:2000]
                    importance = int(item.get("importance", 5))
                    if topic and content:
                        new_id = await save_memory(
                            topic=topic,
                            content=content,
                            importance=importance,
                            source="session",
                            session_id=session_id,
                        )
                        if new_id:
                            memories_saved += 1
                        else:
                            memories_skipped += 1
        except (json.JSONDecodeError, ValueError) as e:
            log.warning(f"summarize_and_save JSON parse failed: {e}. Raw: {result_text[:200]}")

    # Save chat summary row regardless
    summary_text = result_text or "(summarization failed)"
    summary_text = summary_text.replace("'", "''")
    msg_count = len(history)
    model_key_safe = model_key.replace("'", "''")
    sid_safe = session_id.replace("'", "''")
    await execute_sql(
        f"INSERT INTO {_SUM} "
        f"(session_id, summary, message_count, model_used) "
        f"VALUES ('{sid_safe}', '{summary_text[:4000]}', {msg_count}, '{model_key_safe}')"
    )

    skip_note = f", {memories_skipped} duplicate(s) skipped" if memories_skipped else ""
    return f"Summarized {msg_count} messages → {memories_saved} memories saved{skip_note}."


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_table(raw: str) -> list[dict]:
    """Parse tab-separated execute_sql output into list of dicts."""
    lines = raw.strip().splitlines()
    if len(lines) < 2:
        return []
    headers = [h.strip() for h in lines[0].split("\t")]
    rows = []
    for line in lines[1:]:
        if not line.strip() or line.startswith("---"):
            continue
        vals = line.split("\t")
        row = {}
        for i, h in enumerate(headers):
            v = vals[i].strip() if i < len(vals) else ""
            row[h] = v
        rows.append(row)
    return rows
