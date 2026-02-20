import asyncio
from contextvars import ContextVar

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

# autoAIdb state { table_name -> {"read": bool, "write": bool} }
# True = gate OFF (auto-allow), False = gate ON (requires approval)
auto_aidb_state: dict[str, dict[str, bool]] = {}

# Tool gate state { tool_name -> {"read": bool, "write": bool} }
# True = gate OFF (auto-allow), False = gate ON (requires approval)
tool_gate_state: dict[str, dict[str, bool]] = {}

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
    (await get_queue(client_id)).put_nowait({"t": "gate", "d": gate_data})

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