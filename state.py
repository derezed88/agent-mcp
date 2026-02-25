import asyncio
import json
import logging
import os
import re
from contextvars import ContextVar

_log = logging.getLogger("agent")

# Directory for persisted session histories (relative to this file)
SESSION_HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session-history")


def _safe_filename(session_id: str) -> str:
    """Convert a session_id to a safe filename component."""
    # Replace any character that isn't alphanumeric, hyphen, underscore, or dot
    return re.sub(r"[^A-Za-z0-9._-]", "_", session_id)


def save_history(session_id: str, history: list) -> None:
    """Persist a session's history to disk before reaping."""
    if not history:
        return
    try:
        os.makedirs(SESSION_HISTORY_DIR, exist_ok=True)
        path = os.path.join(SESSION_HISTORY_DIR, f"{_safe_filename(session_id)}.history")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False)
        _log.info(f"Session history saved ({len(history)} messages): {path}")
    except Exception as e:
        _log.warning(f"Failed to save session history for {session_id}: {e}")


def load_history(session_id: str) -> list:
    """Load a session's persisted history from disk, if it exists."""
    path = os.path.join(SESSION_HISTORY_DIR, f"{_safe_filename(session_id)}.history")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)
        if isinstance(history, list):
            _log.info(f"Session history loaded ({len(history)} messages): {path}")
            return history
    except Exception as e:
        _log.warning(f"Failed to load session history for {session_id}: {e}")
    return []


# Current client ID context variable — set in execute_tool() so executors can
# read it without needing it as an explicit parameter.
current_client_id: ContextVar[str] = ContextVar("current_client_id", default="")

# Per-client SSE queues
sse_queues: dict[str, asyncio.Queue] = {}
queue_lock = asyncio.Lock()

# Session store
sessions: dict[str, dict] = {}

# Session ID management - shorthand IDs for user convenience
_session_id_counter = 100  # Start at 100 for readability
session_id_to_shorthand: dict[str, int] = {}  # full_session_id -> shorthand_id
shorthand_to_session_id: dict[int, str] = {}  # shorthand_id -> full_session_id

def get_or_create_shorthand_id(session_id: str) -> int:
    """Get existing shorthand ID or create a new one."""
    global _session_id_counter
    if session_id not in session_id_to_shorthand:
        _session_id_counter += 1
        shorthand_id = _session_id_counter
        session_id_to_shorthand[session_id] = shorthand_id
        shorthand_to_session_id[shorthand_id] = session_id
    return session_id_to_shorthand[session_id]

def get_session_by_shorthand(shorthand_id: int) -> str | None:
    """Look up full session ID by shorthand ID."""
    return shorthand_to_session_id.get(shorthand_id)

def remove_shorthand_mapping(session_id: str):
    """Remove shorthand mappings when session is deleted."""
    if session_id in session_id_to_shorthand:
        shorthand_id = session_id_to_shorthand[session_id]
        del session_id_to_shorthand[session_id]
        del shorthand_to_session_id[shorthand_id]

# Active request tasks — one per client_id. Cancelled when a new request arrives
# or when the model is switched while a request is in flight.
active_tasks: dict[str, asyncio.Task] = {}

# Human gate
pending_gates: dict[str, dict] = {}
# Maps client_id -> gate_id while a gate is awaiting human response.
# Used to detect when a new submission should be rejected rather than
# cancelling a task that is legitimately blocked waiting for gate approval.
client_active_gates: dict[str, str] = {}

# autoAIdb state { table_name -> {"read": bool, "write": bool} }
# True = gate OFF (auto-allow), False = gate ON (requires approval)
auto_aidb_state: dict[str, dict[str, bool]] = {}

# Tool gate state { tool_name -> {"read": bool, "write": bool} }
# True = gate OFF (auto-allow), False = gate ON (requires approval)
tool_gate_state: dict[str, dict[str, bool]] = {}

_GATE_DEFAULTS_FILE = os.path.join(os.path.dirname(__file__), "gate-defaults.json")

def load_gate_defaults():
    """Load gate defaults from gate-defaults.json into auto_aidb_state and tool_gate_state."""
    try:
        with open(_GATE_DEFAULTS_FILE, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return  # No defaults file — start with all gates ON (secure by default)
    except (json.JSONDecodeError, OSError):
        return

    db_defaults = data.get("db", {})
    for table, perms in db_defaults.items():
        if isinstance(perms, dict):
            auto_aidb_state[table] = {
                "read":  bool(perms.get("read",  False)),
                "write": bool(perms.get("write", False)),
            }

    tool_defaults = data.get("tools", {})
    for tool, perms in tool_defaults.items():
        if isinstance(perms, dict):
            tool_gate_state[tool] = {
                "read":  bool(perms.get("read",  False)),
                "write": bool(perms.get("write", False)),
            }

load_gate_defaults()

async def get_queue(client_id: str) -> asyncio.Queue:
    async with queue_lock:
        if client_id not in sse_queues:
            sse_queues[client_id] = asyncio.Queue()
        return sse_queues[client_id]

async def drain_queue(client_id: str) -> int:
    """Discard all pending items in the client's SSE queue. Returns count drained."""
    if client_id not in sse_queues:
        return 0
    q = sse_queues[client_id]
    count = 0
    while not q.empty():
        try:
            q.get_nowait()
            count += 1
        except asyncio.QueueEmpty:
            break
    return count

async def push_tok(client_id: str, text: str):
    (await get_queue(client_id)).put_nowait({"t": "tok", "d": text.replace("\n", "\\n")})

async def push_done(client_id: str):
    (await get_queue(client_id)).put_nowait({"t": "done"})

async def push_err(client_id: str, msg: str):
    (await get_queue(client_id)).put_nowait({"t": "err", "d": msg})
    (await get_queue(client_id)).put_nowait({"t": "done"})

async def push_gate(client_id: str, gate_data: dict):
    client_active_gates[client_id] = gate_data["gate_id"]
    (await get_queue(client_id)).put_nowait({"t": "gate", "d": gate_data})

def clear_client_gate(client_id: str):
    """Remove the active gate record for a client once it is resolved or cancelled."""
    client_active_gates.pop(client_id, None)

async def push_model(client_id: str, model_key: str):
    (await get_queue(client_id)).put_nowait({"t": "model", "d": model_key})

# ---------------------------------------------------------------------------
# Legacy gate API (asyncio.Future-based) — used by agents.py for inline gate checks.
# The newer gate system uses push_gate()/clear_client_gate() above.
# ---------------------------------------------------------------------------
_gate_futures: dict[str, asyncio.Future] = {}


def has_pending_gate(client_id: str) -> bool:
    """True if there is an unanswered gate request for this client."""
    return client_id in _gate_futures


async def wait_for_gate(client_id: str, timeout: float = 120.0) -> bool:
    """
    Create a Future for the given client and wait up to `timeout` seconds
    for the user to answer Y/N via resolve_gate().

    Returns True (allow) or False (deny). Auto-denies on timeout.
    """
    loop = asyncio.get_event_loop()
    fut: asyncio.Future = loop.create_future()
    _gate_futures[client_id] = fut
    try:
        return await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
    except asyncio.TimeoutError:
        return False
    finally:
        _gate_futures.pop(client_id, None)


def resolve_gate(client_id: str, approved: bool) -> bool:
    """
    Answer the pending gate Future for the given client.

    Returns True if a pending gate was resolved, False if there was none.
    """
    fut = _gate_futures.get(client_id)
    if fut is None or fut.done():
        return False
    fut.set_result(approved)
    return True


async def cancel_active_task(client_id: str) -> bool:
    """
    Cancel any in-flight request task for this client and drain its output queue.

    Returns True if a task was cancelled, False if there was nothing running.
    Called before spawning a new request task, and on !model switch.
    """
    task = active_tasks.get(client_id)
    # Don't cancel ourselves — the current task is the one processing !model
    if task and not task.done() and task is not asyncio.current_task():
        task.cancel()
        try:
            # Wait up to 5s for the task to acknowledge cancellation.
            # If it's stuck in a non-cancellable I/O call we don't block forever.
            done, _ = await asyncio.wait({task}, timeout=5)
            if not done:
                # Task didn't finish in time — log and move on
                import logging
                logging.getLogger("agent").warning(
                    f"cancel_active_task: task for '{client_id}' didn't finish in 5s after cancel"
                )
        except Exception:
            pass
        active_tasks.pop(client_id, None)
        # Drain any stale tokens the cancelled task already queued
        if client_id in sse_queues:
            q = sse_queues[client_id]
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break
        return True
    active_tasks.pop(client_id, None)
    return False