"""
reflection.py — Periodic reflection loop.

Runs as a background asyncio task. On each cycle it:
  1. Pulls recent conversation turns from samaritan_memory_shortterm
     (source IN ('user','assistant'), ordered by created_at DESC, up to
     reflection_turn_limit rows).
  2. Calls reflection_model with a structured prompt:
       "What did I learn? What should I follow up on? What patterns do I notice?"
  3. Parses the JSON response — an array of {topic, content, importance, type}
     objects — and saves each as a new short-term memory row via save_memory().
     source='assistant', type from response (defaults to 'context').
  4. Saves a reflection summary row to samaritan_memory_shortterm with
     topic='reflection-summary' so it appears in context.

Reflection rows are tagged source='assistant' and carry a 'reflection' flag in
topic ('reflection-summary') so the contradiction scanner, aging logic, and
context injection all treat them like any other assistant memory. The fuzzy-dedup
gate prevents redundant saves between cycles.

Config (plugins-enabled.json → plugin_config.proactive_cognition):
    enabled:                    bool   — master switch
    reflection_enabled:         bool   — this loop (default true when master on)
    reflection_interval_h:      float  — hours between runs (default 6)
    reflection_model:           str    — model key (default "summarizer-gemini")
    reflection_turn_limit:      int    — max ST rows to pull per cycle (default 40)
    reflection_min_turns:       int    — skip if fewer recent turns than this (default 5)
    reflection_max_memories:    int    — max memory rows to save per cycle (default 6)

Runtime control:
    get_reflection_stats()      → dict of counters + last-run info
    trigger_now()               → wake sleeping loop immediately
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone

log = logging.getLogger("reflection")

# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

_stats: dict = {
    "runs":              0,
    "turns_processed":   0,
    "memories_saved":    0,
    "memories_skipped":  0,
    "last_run_at":       None,
    "last_run_duration_s": None,
    "last_run_saved":    0,
    "last_error":        None,
    "last_feedback":     None,   # last feedback verdict dict
}

# Self-summary refresh every N reflection cycles
_SELF_SUMMARY_EVERY_N = 5

_wake_event: asyncio.Event | None = None

_PLUGINS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins-enabled.json")


def get_reflection_stats() -> dict:
    return dict(_stats)


def trigger_now() -> None:
    if _wake_event:
        _wake_event.set()


# ---------------------------------------------------------------------------
# Config helper — shares runtime overrides from contradiction.py
# ---------------------------------------------------------------------------

def _rcogn_cfg() -> dict:
    try:
        with open(_PLUGINS_PATH) as f:
            raw = json.load(f).get("plugin_config", {}).get("proactive_cognition", {})
    except Exception:
        raw = {}

    try:
        from contradiction import get_runtime_overrides
        ovr = get_runtime_overrides()
    except ImportError:
        ovr = {}

    base = {
        "enabled":               raw.get("enabled",               False),
        "reflection_enabled":    raw.get("reflection_enabled",    True),
        "reflection_interval_h": float(raw.get("reflection_interval_h", 6.0)),
        "reflection_model":      raw.get("reflection_model",      "summarizer-gemini"),
        "reflection_turn_limit": int(raw.get("reflection_turn_limit",  40)),
        "reflection_min_turns":  int(raw.get("reflection_min_turns",   5)),
        "reflection_max_memories": int(raw.get("reflection_max_memories", 6)),
    }
    base.update(ovr)
    return base


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a reflective memory assistant. You analyse recent conversation turns "
    "and extract durable insights: things learned, patterns noticed, follow-up items. "
    "Be concise. Do not repeat what was said verbatim — synthesise it.\n\n"
    "You also detect behavioral patterns about the AI agent itself. When you notice "
    "repeated successes, failures, or preferences in how the agent operates, emit "
    "rows with prefixed topics:\n"
    "  - self-capability-<slug>: something the agent does well\n"
    "  - self-failure-<slug>: something the agent struggles with or gets wrong repeatedly\n"
    "  - self-preference-<slug>: a stylistic or behavioral tendency the agent exhibits\n"
    "Keep these terse — 1-2 sentences each. Use type='self_model' for these rows.\n\n"
    "You also detect goal completions. If active goals are provided and the conversation "
    "clearly shows one has been achieved, emit a goal_done entry."
)

_USER_PROMPT_TMPL = (
    "Below are recent conversation turns (newest last). "
    "Identify up to {max_memories} distinct insights worth remembering.\n\n"
    "Return a JSON object with two keys:\n"
    '  "memories": array of memory rows\n'
    '  "goals_done": array of goal IDs (integers) clearly completed in these turns\n\n'
    "Each memory row:\n"
    '  {{"topic": "slug-label", "content": "one sentence", '
    '"importance": 1-10, "type": "context|belief|semantic|episodic|procedural|self_model"}}\n\n'
    "Memory rules:\n"
    "- topic: a short dash-separated slug (reuse existing topics when possible)\n"
    "- importance: 7-9 for concrete follow-up items; 5-6 for general observations\n"
    "- type: use 'belief' for inferred facts about user/world; 'semantic' for general "
    "knowledge; 'procedural' for how-to; 'episodic' for specific events; 'context' otherwise\n"
    "- For self-model patterns: use topic prefix 'self-capability-', 'self-failure-', or "
    "'self-preference-' and type='self_model'. Only emit these when a clear pattern is evident.\n"
    "- Skip anything already obvious from the raw text or too ephemeral to be useful\n\n"
    "Goal completion rules:\n"
    "- Only mark a goal done if the conversation clearly shows the objective was achieved\n"
    "- Do NOT mark done based on plans or intent — only on evidence of completion\n"
    "- Return empty array [] for goals_done if no goals were clearly completed\n\n"
    "{active_goals_block}"
    "TURNS:\n{turns_text}"
)


# ---------------------------------------------------------------------------
# Fetch recent turns from ST
# ---------------------------------------------------------------------------

async def _fetch_recent_turns(limit: int) -> list[dict]:
    from database import fetch_dicts
    from memory import _ST
    try:
        return await fetch_dicts(
            f"SELECT id, topic, content, source, created_at "
            f"FROM {_ST()} "
            f"WHERE source IN ('user', 'assistant') "
            f"ORDER BY created_at DESC LIMIT {limit}"
        ) or []
    except Exception as e:
        log.warning(f"reflection: fetch_recent_turns failed: {e}")
        return []


def _format_turns(rows: list[dict]) -> str:
    # Rows come back newest-first; reverse to chronological for the prompt
    lines = []
    for r in reversed(rows):
        src = r.get("source", "?").upper()
        content = (r.get("content") or "").strip()[:400]
        topic = r.get("topic", "")
        lines.append(f"[{src}|{topic}] {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

async def _call_llm(model_key: str, turns_text: str, max_memories: int,
                   active_goals: list[dict] | None = None) -> tuple[list[dict], list[int]]:
    """
    Returns (memory_items, goal_ids_done).
    memory_items: list of memory row dicts to save.
    goal_ids_done: list of goal IDs the LLM identified as completed.
    """
    from config import LLM_REGISTRY
    from agents import _build_lc_llm, _content_to_str
    from langchain_core.messages import SystemMessage, HumanMessage

    if active_goals:
        goals_lines = "\n".join(
            f"  [id={g['id']}] {g.get('title','')} — {g.get('description','')}"
            for g in active_goals
        )
        goals_block = f"ACTIVE GOALS (check for completion):\n{goals_lines}\n\n"
    else:
        goals_block = ""

    prompt = _USER_PROMPT_TMPL.format(
        turns_text=turns_text,
        max_memories=max_memories,
        active_goals_block=goals_block,
    )
    try:
        if model_key not in LLM_REGISTRY:
            log.warning(f"reflection: unknown model {model_key!r}")
            return [], []
        cfg = LLM_REGISTRY[model_key]
        timeout = cfg.get("llm_call_timeout", 90)
        llm = _build_lc_llm(model_key)
        msgs = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        response = await asyncio.wait_for(llm.ainvoke(msgs), timeout=timeout)
        raw = _content_to_str(response.content)
    except Exception as e:
        log.warning(f"reflection: LLM call failed: {e}")
        return [], []

    if not raw:
        return [], []

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])

    try:
        parsed = json.loads(cleaned)
        # New format: {"memories": [...], "goals_done": [...]}
        if isinstance(parsed, dict):
            memories = [x for x in parsed.get("memories", []) if isinstance(x, dict)]
            goals_done = [int(x) for x in parsed.get("goals_done", []) if str(x).isdigit()]
            return memories, goals_done
        # Legacy format: bare array (backward compat)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)], []
    except (json.JSONDecodeError, ValueError) as e:
        log.warning(f"reflection: JSON parse failed: {e}. raw={raw[:200]}")
    return [], []


# ---------------------------------------------------------------------------
# Core run logic
# ---------------------------------------------------------------------------

async def run_reflection() -> dict:
    """
    Run one reflection pass. Safe to call manually (e.g. from !cogn reflection run).
    Returns summary dict.
    """
    cfg = _rcogn_cfg()
    model_key    = cfg["reflection_model"]
    turn_limit   = cfg["reflection_turn_limit"]
    min_turns    = cfg["reflection_min_turns"]
    max_memories = cfg["reflection_max_memories"]

    from database import set_model_context
    from config import DEFAULT_MODEL
    set_model_context(DEFAULT_MODEL)

    t_start = time.monotonic()
    summary = {"turns": 0, "saved": 0, "skipped": 0, "error": None}

    try:
        rows = await _fetch_recent_turns(turn_limit)
        summary["turns"] = len(rows)
        _stats["turns_processed"] += len(rows)

        if len(rows) < min_turns:
            log.debug(f"reflection: only {len(rows)} turns < min={min_turns}, skipping")
            summary["skipped_reason"] = f"only {len(rows)} recent turns (min={min_turns})"
            return summary

        turns_text = _format_turns(rows)

        # Fetch active goals to pass for completion detection
        active_goals: list[dict] = []
        try:
            from memory import _GOALS
            from database import fetch_dicts as _fd
            active_goals = await _fd(
                f"SELECT id, title, description FROM {_GOALS()} WHERE status = 'active'"
            ) or []
        except Exception as _ge:
            log.debug(f"reflection: goals fetch failed: {_ge}")

        items, goals_done = await _call_llm(model_key, turns_text, max_memories, active_goals)

        from memory import save_memory
        for item in items:
            if not isinstance(item, dict):
                continue
            topic   = str(item.get("topic", "reflection"))[:255]
            content = str(item.get("content", ""))[:2000]
            imp     = max(1, min(10, int(item.get("importance", 6))))
            mtype   = str(item.get("type", "context"))
            # self_model is not an ST enum value — store as semantic with topic prefix intact
            if mtype == "self_model":
                mtype = "semantic"

            if not topic or not content:
                continue

            new_id = await save_memory(
                topic=topic,
                content=content,
                importance=imp,
                source="assistant",
                type=mtype,
            )
            if new_id:
                summary["saved"] += 1
                _stats["memories_saved"] += 1
            else:
                summary["skipped"] += 1
                _stats["memories_skipped"] += 1

        # Process goal completions detected by LLM
        if goals_done:
            from memory import _GOALS
            from database import execute_sql as _exec_sql
            valid_ids = {g["id"] for g in active_goals}
            marked = []
            for gid in goals_done:
                if gid not in valid_ids:
                    log.debug(f"reflection: goal_done id={gid} not in active goals — skipped")
                    continue
                try:
                    await _exec_sql(
                        f"UPDATE {_GOALS()} SET status='done' WHERE id={gid} AND status='active'"
                    )
                    marked.append(gid)
                    log.info(f"reflection: marked goal id={gid} as done (detected in conversation)")
                except Exception as _ge2:
                    log.warning(f"reflection: goal mark-done failed id={gid}: {_ge2}")
            if marked:
                summary["goals_marked_done"] = marked

    except Exception as e:
        log.error(f"reflection: run error: {e}")
        summary["error"] = str(e)
        _stats["last_error"] = str(e)

    duration = time.monotonic() - t_start
    _stats["runs"]             += 1
    _stats["last_run_at"]       = datetime.now(timezone.utc).isoformat()
    _stats["last_run_duration_s"] = round(duration, 2)
    _stats["last_run_saved"]    = summary["saved"]

    log.info(
        f"reflection: run done — turns={summary['turns']} saved={summary['saved']} "
        f"skipped={summary['skipped']} dur={duration:.1f}s"
    )

    # Drive decay and goal-based nudge
    try:
        from memory import update_drives_from_goals
        drive_summary = await update_drives_from_goals()
        summary["drives_updated"] = drive_summary.get("drives_updated", 0)
        log.info(
            f"reflection: drives — updated={drive_summary.get('drives_updated', 0)} "
            f"goals_done={drive_summary.get('goals_done', 0)} "
            f"goals_blocked={drive_summary.get('goals_blocked', 0)}"
        )
    except Exception as e:
        log.warning(f"reflection: drive update failed: {e}")

    # Feedback evaluation — update watermark first so evaluator sees rows written this cycle
    try:
        from cogn_feedback import evaluate, LOOP_REFLECTION
        fb = await evaluate(LOOP_REFLECTION, summary)
        _stats["last_feedback"] = fb
        if fb.get("verdict") not in (None, "insufficient_data", "neutral", "useful"):
            log.info(f"reflection: feedback verdict={fb.get('verdict')} strength={fb.get('strength')} streak={fb.get('streak')}")
    except Exception as e:
        log.warning(f"reflection: feedback evaluation failed: {e}")

    # Refresh self-summary every N cycles
    if _stats["runs"] % _SELF_SUMMARY_EVERY_N == 0:
        try:
            from memory import refresh_self_summary
            from agents import _call_llm_text
            from config import LLM_REGISTRY
            _summarizer_key = cfg.get("reflection_model", "summarizer-gemini")
            # Fall back to any available summarizer
            if _summarizer_key not in LLM_REGISTRY:
                _summarizer_key = next(
                    (k for k in LLM_REGISTRY if "summarizer" in k), _summarizer_key
                )

            async def _self_llm_fn(prompt: str) -> str:
                return await _call_llm_text(_summarizer_key, prompt)

            _self_summary = await refresh_self_summary(llm_call_fn=_self_llm_fn)
            if _self_summary:
                log.info(f"reflection: self-summary refreshed ({len(_self_summary)} chars)")
                summary["self_summary_refreshed"] = True
        except Exception as e:
            log.warning(f"reflection: self-summary refresh failed: {e}")

    return summary


# ---------------------------------------------------------------------------
# Background task entry point
# ---------------------------------------------------------------------------

async def reflection_task() -> None:
    """
    Long-running asyncio task. Loops every reflection_interval_h hours.
    Wakes early if trigger_now() is called.
    """
    global _wake_event
    _wake_event = asyncio.Event()

    while True:
        cfg = _rcogn_cfg()

        if not cfg["enabled"] or not cfg["reflection_enabled"]:
            _wake_event.clear()
            try:
                await asyncio.wait_for(_wake_event.wait(), timeout=300)
                _wake_event.clear()
            except asyncio.TimeoutError:
                pass
            continue

        interval_h = cfg["reflection_interval_h"]
        if interval_h <= 0:
            await asyncio.sleep(3600)
            continue

        try:
            await run_reflection()
        except Exception as e:
            log.warning(f"reflection_task: unhandled error: {e}")
            _stats["last_error"] = str(e)

        sleep_sec = interval_h * 3600
        _wake_event.clear()
        try:
            await asyncio.wait_for(_wake_event.wait(), timeout=sleep_sec)
            log.info("reflection_task: woken early by trigger")
            _wake_event.clear()
        except asyncio.TimeoutError:
            pass
