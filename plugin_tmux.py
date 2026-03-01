"""
Tmux-like Shell Session Plugin for MCP Agent

Provides persistent PTY (pseudo-terminal) shell sessions.
LLMs interact via tool calls; humans manage sessions via !tmux commands.

Config keys in plugins-enabled.json → plugin_config.plugin_tmux:
    TMUX_HISTORY_LIMIT        int   Rolling history lines per session (default 200)
    TMUX_EXEC_TIMEOUT         float Seconds to wait for command output in tmux_exec (default 10.0)
    TMUX_ALLOWED_COMMANDS     list  Whitelist of allowed command prefixes (empty = all allowed)
    TMUX_BLOCKED_COMMANDS     list  Blacklist of blocked command prefixes (checked after allow)

Tools (all write-gated via !tmux_gate_write or per-tool !tmux_<name>_gate_write):
    tmux_new(name)                         — create a new shell session
    tmux_exec(session, command, timeout)   — run a command, return output
    tmux_ls()                              — list active sessions
    tmux_kill_session(name)                — terminate one session
    tmux_kill_server()                     — terminate all sessions
    tmux_history(session, lines)           — show rolling history
    tmux_history_limit(n)                  — change history line limit
    tmux_call_limit()                      — show current rate limit config  [read]
    tmux_call_limit(calls, window)         — set rate limit                  [write]

User commands (dispatched from routes.py cmd_tmux):
    !tmux new <name>
    !tmux ls
    !tmux kill-session <name>
    !tmux kill-server
    !tmux buf <name>
    !tmux history-limit [n]
    !tmux send-keys <name> <key>  — send raw keypress (C-c, C-d, C-z, enter)
    !tmux_call_limit              - show current limit
    !tmux_call_limit <calls> <window_seconds>  - set limit

Gates (all default to gated/false):
    Group:    !tmux_gate_write <t|f>              — covers all tmux tools
    Per-tool: !tmux_<toolname>_gate_write <t|f>   — overrides group for that tool
    Rate limit: !tmux_call_limit_gate_read  <t|f>
                !tmux_call_limit_gate_write <t|f>
"""

import asyncio
import fcntl
import os
import pty
import re
import select
import struct
import termios
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from plugin_loader import BasePlugin
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Module-level state — shared by plugin instance and routes.py cmd_tmux()
# ---------------------------------------------------------------------------

# name -> {proc, master, history, name, started, exec_active, reader_thread, reader_stop, ...}
_sessions: Dict[str, Dict[str, Any]] = {}

# name -> asyncio.Event — registered synchronously at start of _do_create,
# set when session is fully ready. Allows exec to wait for in-flight creates.
_pending_creates: Dict[str, asyncio.Event] = {}

# Rolling history limit (lines). Updated by !tmux history-limit / tmux_history_limit.
_history_limit: int = 200

# Command filter lists. Empty allowed = all allowed. Checked in tmux_exec_executor.
# Format: list of command prefixes (lowercased for matching).
_allowed_commands: List[str] = []   # empty = allow all
_blocked_commands: List[str] = []   # empty = block none

# Outbound agent blocked commands — loaded from plugin_client_api.OUTBOUND_AGENT_BLOCKED_COMMANDS.
# Applied in tmux_exec_executor in addition to TMUX_BLOCKED_COMMANDS.
# Empty [] = nothing extra blocked. Always checked when non-empty.
_outbound_blocked_commands: List[str] = []

# Rate limit config (mirrors plugins-enabled.json rate_limits.tmux section at runtime)
# Updated by tmux_call_limit_executor / tmux_command("call-limit", ...).
_rate_limit_calls: int = 30
_rate_limit_window: int = 60

# --- Tuning constants -------------------------------------------------------
_READ_TIMEOUT_DEFAULT: float = 10.0
_DRAIN_PAUSE: float = 0.25
_STARTUP_DRAIN: float = 0.5
_CHUNK: int = 4096
_PTY_COLS: int = 220


# ---------------------------------------------------------------------------
# Command filtering
# ---------------------------------------------------------------------------

def _check_command(command: str) -> Optional[str]:
    """
    Apply TMUX_ALLOWED_COMMANDS and TMUX_BLOCKED_COMMANDS filters.
    Returns None if the command is permitted, or an error string if blocked.

    Matching is against the first word of the command (the executable name),
    case-insensitive.  Patterns may include a second word for subcommand
    matching (e.g. "git push").
    """
    cmd_lower = command.strip().lower()

    # Allowed list: if non-empty, command must match at least one entry
    if _allowed_commands:
        matched = any(cmd_lower.startswith(p) for p in _allowed_commands)
        if not matched:
            allowed_str = ", ".join(_allowed_commands)
            return (
                f"BLOCKED: command not in TMUX_ALLOWED_COMMANDS.\n"
                f"Allowed prefixes: {allowed_str}"
            )

    # Blocked list: command must not match any entry
    for pattern in _blocked_commands:
        if cmd_lower.startswith(pattern):
            return f"BLOCKED: command matches TMUX_BLOCKED_COMMANDS pattern '{pattern}'."

    return None


def _format_filter_status() -> str:
    """Return a human-readable summary of current filter configuration."""
    if _allowed_commands:
        allow_str = f"ALLOW-LIST ({len(_allowed_commands)} entries): " + ", ".join(_allowed_commands)
    else:
        allow_str = "ALLOW-LIST: (empty — all commands permitted)"
    if _blocked_commands:
        block_str = f"BLOCK-LIST ({len(_blocked_commands)} entries): " + ", ".join(_blocked_commands)
    else:
        block_str = "BLOCK-LIST: (empty — nothing explicitly blocked)"
    return f"{allow_str}\n{block_str}"


# ---------------------------------------------------------------------------
# Internal PTY helpers
# ---------------------------------------------------------------------------

def _valid_name(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]+", name))


def _resize_pty(fd: int, rows: int = 24, cols: int = _PTY_COLS) -> None:
    winsize = struct.pack("HHHH", rows, cols, 0, 0)
    try:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)
    except OSError:
        pass


def _make_nonblocking(fd: int) -> None:
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)


def _drain_sync(master_fd: int) -> bytes:
    buf = b""
    try:
        while True:
            chunk = os.read(master_fd, _CHUNK)
            if not chunk:
                break
            buf += chunk
    except (BlockingIOError, OSError):
        pass
    return buf


def _decode_pty(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    # Strip ANSI/VT escape sequences:
    #   CSI sequences:  \x1b[ ... final-byte  (colors, cursor movement, etc.)
    #   OSC sequences:  \x1b] ... \x07|\x1b\\ (shell integration, title, etc.)
    #   Other Fe seqs:  \x1b[@-Z\-_]          (SS2, SS3, DCS, etc.)
    ansi = re.compile(
        r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"   # OSC: ESC ] ... BEL or ESC \
        r"|\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"  # CSI + other Fe
    )
    text = ansi.sub("", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


async def _read_output(master_fd: int,
                       timeout: float = _READ_TIMEOUT_DEFAULT,
                       drain_pause: float = _DRAIN_PAUSE) -> str:
    """
    Read PTY output using loop.add_reader (event-driven, no thread pool).

    Waits up to `timeout` seconds for the first byte, then resets the deadline
    to `drain_pause` seconds after each chunk to catch burst output.
    Returns when `drain_pause` seconds of silence follows the last chunk,
    or when the absolute deadline is exceeded.
    """
    loop = asyncio.get_event_loop()
    buf = b""
    readable = asyncio.Event()

    def _on_readable():
        readable.set()

    loop.add_reader(master_fd, _on_readable)
    try:
        got_data = False
        deadline = loop.time() + timeout

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            wait = drain_pause if got_data else remaining
            readable.clear()
            try:
                await asyncio.wait_for(readable.wait(), timeout=wait)
            except asyncio.TimeoutError:
                break

            # Drain all available bytes without blocking
            while True:
                try:
                    chunk = os.read(master_fd, _CHUNK)
                except (BlockingIOError, InterruptedError):
                    break
                except OSError:
                    return _decode_pty(buf)
                if not chunk:
                    break
                buf += chunk
                got_data = True
                deadline = loop.time() + drain_pause
    finally:
        loop.remove_reader(master_fd)

    return _decode_pty(buf)


def _append_history(name: str, text: str) -> None:
    sess = _sessions.get(name)
    if not sess:
        return
    for line in text.splitlines():
        sess["history"].append(line)


def _ensure_reader_alive(name: str) -> None:
    """Auto-restart the bg reader thread if it has died or was never started."""
    sess = _sessions.get(name)
    if not sess:
        return
    # Only restart if the process is still running
    if sess["proc"].returncode is not None:
        return
    t = sess.get("reader_thread")
    if t is not None and t.is_alive():
        return
    # Thread is dead or missing — restart
    import logging
    log = logging.getLogger(__name__)
    log.warning("Auto-restarting dead reader thread for session '%s'", name)
    old_stop = sess.get("reader_stop")
    if old_stop:
        old_stop.set()
    stop_event = threading.Event()
    sess["reader_stop"] = stop_event
    sess["exec_active"] = False
    new_t = threading.Thread(
        target=_bg_reader_thread,
        args=(name, stop_event),
        name=f"tmux-reader-{name}",
        daemon=True,
    )
    sess["reader_thread"] = new_t
    new_t.start()
    log.warning("Reader thread restarted for '%s' (tid=%s)", name, new_t.ident)


def _bg_reader_thread(name: str, stop_event: threading.Event) -> None:
    """
    Background thread: continuously drain PTY master fd into session history.

    Runs in a dedicated OS thread so it is never starved by asyncio event loop
    scheduling. Uses select() to block until data is available — PTY buffer
    never fills regardless of event loop load.

    Coordinates with tmux_exec via exec_active flag: backs off for one select
    cycle while exec is reading so _read_output gets clean output.
    """
    import logging
    log = logging.getLogger(__name__)
    log.debug("bg_reader_thread START name=%s", name)

    while not stop_event.is_set():
        sess = _sessions.get(name)
        if not sess:
            log.debug("bg_reader_thread EXIT name=%s: session gone", name)
            break

        # Check process exit — but keep draining until no more data
        proc_done = sess["proc"].returncode is not None

        # Back off while exec is reading
        if sess.get("exec_active"):
            time.sleep(0.01)
            continue

        master_fd = sess["master"]
        try:
            ready, _, _ = select.select([master_fd], [], [], 0.5)
        except (ValueError, OSError) as e:
            log.debug("bg_reader_thread EXIT name=%s: select error %s", name, e)
            break

        if not ready:
            if proc_done:
                # Process exited and no more data — we're done
                log.debug("bg_reader_thread EXIT name=%s: proc done, no data", name)
                break
            continue

        if sess.get("exec_active"):
            continue

        buf = b""
        while True:
            try:
                chunk = os.read(master_fd, _CHUNK)
            except (BlockingIOError, InterruptedError):
                break
            except OSError as e:
                log.debug("bg_reader_thread EXIT name=%s: read OSError %s", name, e)
                return
            if not chunk:
                break
            buf += chunk
        if buf:
            _append_history(name, _decode_pty(buf))

    log.debug("bg_reader_thread DONE name=%s", name)


def _is_alive(name: str) -> bool:
    sess = _sessions.get(name)
    return bool(sess and sess["proc"].returncode is None)


async def _await_session(name: str, timeout: float = 10.0) -> Optional[str]:
    """
    Wait for a session to be fully initialized.
    Handles the case where !tmux new and !tmux exec are pasted simultaneously.
    Returns None on success, or an error string on timeout/not-found.
    """
    # Fast path: session exists and ready
    sess = _sessions.get(name)
    if sess:
        ready = sess.get("ready")
        if not ready or ready.is_set():
            return None
        # Session exists but still initializing — wait on its ready event
        try:
            await asyncio.wait_for(ready.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return f"ERROR: Session '{name}' creation timed out."
        return None

    # Session doesn't exist yet — check if a create is in flight
    pending = _pending_creates.get(name)
    if pending:
        try:
            await asyncio.wait_for(pending.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return f"ERROR: Session '{name}' creation timed out."
        return None

    # No session and no pending create
    names = ", ".join(_sessions.keys()) if _sessions else "(none)"
    return (
        f"ERROR: No tmux session named '{name}'.\n"
        f"Active sessions: {names}"
    )


async def _do_create(name: str) -> str:
    if name in _sessions:
        return f"Session '{name}' already exists."
    if name in _pending_creates:
        return f"Session '{name}' is already being created."

    # Register pending create SYNCHRONOUSLY (before any await) so that
    # exec calls pasted simultaneously can find and wait on it.
    ready_event = asyncio.Event()
    _pending_creates[name] = ready_event

    # Spawn the actual setup as a shielded task so it survives cancellation
    # of the parent request task (cancel_active_task in routes.py cancels the
    # previous task when a new !command arrives from the same client).
    async def _setup():
        try:
            master_fd, slave_fd = pty.openpty()
            _resize_pty(master_fd)
            _make_nonblocking(master_fd)

            env = os.environ.copy()
            _SESSION_VARS = [
                "DISPLAY", "WAYLAND_DISPLAY",
                "DBUS_SESSION_BUS_ADDRESS",
                "XDG_RUNTIME_DIR", "XDG_SESSION_ID", "XDG_SESSION_TYPE",
                "SSH_AUTH_SOCK", "SSH_AGENT_PID",
                "GNOME_KEYRING_CONTROL", "GNOME_KEYRING_PID",
            ]
            for var in _SESSION_VARS:
                val = os.environ.get(var)
                if val:
                    env[var] = val

            proc = await asyncio.create_subprocess_exec(
                "/bin/bash", "--login", "-i",
                stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
                start_new_session=True,
                env=env,
            )
            os.close(slave_fd)

            stop_event = threading.Event()
            _sessions[name] = {
                "proc":          proc,
                "master":        master_fd,
                "history":       deque(maxlen=_history_limit),
                "name":          name,
                "started":       time.time(),
                "exec_active":   False,
                "reader_stop":   stop_event,
                "reader_thread": None,
                "exec_lock":     asyncio.Lock(),
                "ready":         ready_event,
            }

            await asyncio.sleep(_STARTUP_DRAIN)
            _drain_sync(master_fd)

            t = threading.Thread(
                target=_bg_reader_thread,
                args=(name, stop_event),
                name=f"tmux-reader-{name}",
                daemon=True,
            )
            _sessions[name]["reader_thread"] = t
            t.start()

            ready_event.set()
        except Exception:
            # If setup fails, clean up and signal ready (with no session)
            _sessions.pop(name, None)
            ready_event.set()
            raise
        finally:
            _pending_creates.pop(name, None)

    # Shield the setup from cancellation — even if this task is cancelled,
    # the inner coroutine runs to completion
    await asyncio.shield(_setup())

    sess = _sessions.get(name)
    if sess:
        return f"Session '{name}' created (pid {sess['proc'].pid}, reader tid {sess['reader_thread'].ident})."
    return f"ERROR: Session '{name}' setup failed."


async def _do_kill(name: str) -> str:
    sess = _sessions.pop(name, None)
    if not sess:
        return f"No session named '{name}'."

    # Stop background reader thread
    stop_event = sess.get("reader_stop")
    if stop_event:
        stop_event.set()
    # Thread will exit on next select() timeout (≤0.5s); no need to join

    proc = sess["proc"]
    try:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            proc.kill()
    except ProcessLookupError:
        pass
    try:
        os.close(sess["master"])
    except OSError:
        pass
    return f"Session '{name}' killed."


async def _do_kill_all() -> str:
    names = list(_sessions.keys())
    if not names:
        return "No active sessions."
    return "\n".join([await _do_kill(n) for n in names])


def _do_ls() -> str:
    if not _sessions:
        return "No active tmux sessions."
    lines = ["Active tmux sessions:"]
    for name, sess in _sessions.items():
        status = "running" if _is_alive(name) else "exited"
        age = int(time.time() - sess["started"])
        hist = len(sess["history"])
        t = sess.get("reader_thread")
        reader = "reader=ok" if (t and t.is_alive()) else "reader=DEAD"
        lines.append(
            f"  {name:<20}  pid={sess['proc'].pid}  {status}"
            f"  age={age}s  history={hist}/{sess['history'].maxlen} lines  {reader}"
        )
    return "\n".join(lines)


def _do_history(name: str, lines: int = 50) -> str:
    sess = _sessions.get(name)
    if not sess:
        return f"No session named '{name}'."
    _ensure_reader_alive(name)
    hist = list(sess["history"])
    tail = hist[-lines:] if len(hist) > lines else hist
    header = f"=== {name} (last {len(tail)} of {len(hist)} lines) ==="
    return header + "\n" + "\n".join(tail) if tail else header + "\n(empty)"


# ---------------------------------------------------------------------------
# Tool executors
# ---------------------------------------------------------------------------

class _TmuxNewArgs(BaseModel):
    name: str = Field(description="Session name (letters, digits, hyphens, underscores)")

async def tmux_new_executor(name: str) -> str:
    name = name.strip()
    if not _valid_name(name):
        return f"ERROR: Invalid session name '{name}'. Use letters, digits, hyphens, underscores only."
    return await _do_create(name)


class _TmuxExecArgs(BaseModel):
    session: str = Field(description="Target session name")
    command: str = Field(
        description=(
            "Shell command to send. For long-running jobs use '&' to background and tee "
            "output to a log file. Control chars: \\x03=Ctrl-C, \\x04=Ctrl-D."
        )
    )
    timeout: Optional[float] = Field(
        default=10.0,
        description="Seconds to wait for output silence (default 10, max 120).",
    )

async def tmux_exec_executor(session: str, command: str, timeout: float = 10.0) -> str:
    # Shield the ENTIRE exec flow (await_session + checks + lock + write + read)
    # from task cancellation. cancel_active_task() in routes.py cancels the
    # previous request task when a new command arrives from the same client.
    # Without shield, a rapid-fire paste (new + exec cd + exec bash) would
    # cancel the cd exec before it runs, leaving the shell in the wrong dir.
    async def _full_exec():
        # Wait for session to appear and be ready (handles rapid new+exec paste)
        err = await _await_session(session)
        if err:
            return err + f"\nCreate one with: tmux_new(name='{session}')"

        if not _is_alive(session):
            return (
                f"ERROR: Session '{session}' process has exited. "
                f"Kill and recreate: tmux_kill_session(name='{session}') "
                f"then tmux_new(name='{session}')"
            )

        # Command filter check
        block_reason = _check_command(command)
        if block_reason:
            return f"ERROR: {block_reason}\nCommand was NOT sent to the session."

        # Outbound agent blocked command check
        if _outbound_blocked_commands:
            cmd_lower = command.strip().lower()
            for pattern in _outbound_blocked_commands:
                if cmd_lower.startswith(pattern):
                    return (
                        f"ERROR: BLOCKED by OUTBOUND_AGENT_BLOCKED_COMMANDS: "
                        f"command matches blocked pattern '{pattern}'.\n"
                        f"Command was NOT sent to the session."
                    )

        t_out = max(1.0, min(float(timeout or 10.0), 120.0))
        sess = _sessions[session]
        master_fd = sess["master"]
        cmd_bytes = (command.rstrip("\n") + "\n").encode("utf-8")

        exec_lock = sess.get("exec_lock")
        if exec_lock is None:
            exec_lock = asyncio.Lock()
            sess["exec_lock"] = exec_lock

        async with exec_lock:
            _ensure_reader_alive(session)

            sess["exec_active"] = True
            await asyncio.sleep(0.05)
            try:
                try:
                    os.write(master_fd, cmd_bytes)
                except OSError as e:
                    return f"ERROR: Write to session '{session}' failed: {e}"

                output = await _read_output(master_fd, timeout=t_out)
            finally:
                sess["exec_active"] = False

            _append_history(session, f"$ {command}\n" + output)
            return output if output.strip() else "(no output)"

    return await asyncio.shield(_full_exec())


class _TmuxLsArgs(BaseModel):
    pass

async def tmux_ls_executor() -> str:
    return _do_ls()


class _TmuxKillSessionArgs(BaseModel):
    name: str = Field(description="Session name to terminate")

async def tmux_kill_session_executor(name: str) -> str:
    return await _do_kill(name.strip())


class _TmuxKillServerArgs(BaseModel):
    pass

async def tmux_kill_server_executor() -> str:
    return await _do_kill_all()


class _TmuxHistoryArgs(BaseModel):
    session: str = Field(description="Session name")
    lines: Optional[int] = Field(default=50, description="Number of recent lines to show (default 50)")

async def tmux_history_executor(session: str, lines: int = 50) -> str:
    return _do_history(session.strip(), int(lines or 50))


class _TmuxHistoryLimitArgs(BaseModel):
    n: int = Field(description="New rolling history limit (lines per session, >= 1)")

async def tmux_history_limit_executor(n: int) -> str:
    global _history_limit
    if n < 1:
        return "ERROR: History limit must be >= 1."
    _history_limit = n
    for sess in _sessions.values():
        old = list(sess["history"])
        sess["history"] = deque(old[-n:], maxlen=n)
    return f"History limit set to {n} lines. ({len(_sessions)} session(s) updated)"


class _TmuxCallLimitArgs(BaseModel):
    calls: Optional[int] = Field(
        default=None,
        description="Max calls allowed in the window (omit to read current setting)"
    )
    window_seconds: Optional[int] = Field(
        default=None,
        description="Rolling window length in seconds (omit to read current setting)"
    )

async def tmux_call_limit_executor(calls: Optional[int] = None,
                                    window_seconds: Optional[int] = None) -> str:
    """Read or update the tmux tool rate limit."""
    global _rate_limit_calls, _rate_limit_window

    if calls is None and window_seconds is None:
        # Read
        return (
            f"tmux rate limit: {_rate_limit_calls} calls per {_rate_limit_window}s\n"
            f"(auto_disable=true — tmux tools disabled on breach until agent restart)"
        )

    # Write — both must be supplied together
    if calls is None or window_seconds is None:
        return "ERROR: Provide both calls and window_seconds, or neither (to read)."
    if calls < 1:
        return "ERROR: calls must be >= 1."
    if window_seconds < 1:
        return "ERROR: window_seconds must be >= 1."

    old_calls, old_window = _rate_limit_calls, _rate_limit_window
    _rate_limit_calls = calls
    _rate_limit_window = window_seconds

    # Persist to plugins-enabled.json
    try:
        import json
        path = os.path.join(os.path.dirname(__file__), "plugins-enabled.json")
        with open(path, "r") as f:
            cfg = json.load(f)
        cfg.setdefault("rate_limits", {})["tmux"] = {
            "calls": calls,
            "window_seconds": window_seconds,
            "auto_disable": True,
        }
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        saved = " Persisted to plugins-enabled.json."
    except Exception as e:
        saved = f" WARNING: in-memory only, failed to persist: {e}"

    return (
        f"tmux rate limit: {old_calls}/{old_window}s → {calls}/{window_seconds}s.{saved}"
    )


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------

class TmuxPlugin(BasePlugin):
    """PTY shell session manager — tmux-like sessions for LLM shell interaction."""

    PLUGIN_NAME = "plugin_tmux"
    PLUGIN_VERSION = "1.1.0"
    PLUGIN_TYPE = "data_tool"
    DESCRIPTION = "PTY shell sessions with rolling history, command filtering, and rate limiting"
    DEPENDENCIES = []
    ENV_VARS = []

    def __init__(self):
        self.enabled = False

    def init(self, config: dict) -> bool:
        global _history_limit, _allowed_commands, _blocked_commands
        global _rate_limit_calls, _rate_limit_window
        global _outbound_blocked_commands
        global _READ_TIMEOUT_DEFAULT

        # History limit
        try:
            _history_limit = int(config.get("TMUX_HISTORY_LIMIT", 200))
        except (TypeError, ValueError):
            _history_limit = 200

        # Exec timeout
        try:
            _READ_TIMEOUT_DEFAULT = float(config.get("TMUX_EXEC_TIMEOUT", 10.0))
        except (TypeError, ValueError):
            _READ_TIMEOUT_DEFAULT = 10.0

        # Command filter lists — store as lowercase prefixes for fast matching
        raw_allowed = config.get("TMUX_ALLOWED_COMMANDS", [])
        raw_blocked = config.get("TMUX_BLOCKED_COMMANDS", [])
        _allowed_commands = [s.strip().lower() for s in raw_allowed if s.strip()]
        _blocked_commands = [s.strip().lower() for s in raw_blocked if s.strip()]

        # Outbound agent blocked commands — from plugin_client_api config
        try:
            import json as _json
            _path = os.path.join(os.path.dirname(__file__), "plugins-enabled.json")
            with open(_path, "r") as _f:
                _full = _json.load(_f)
            raw_outbound_blocked = (
                _full.get("plugin_config", {})
                     .get("plugin_client_api", {})
                     .get("OUTBOUND_AGENT_BLOCKED_COMMANDS", [])
            )
            _outbound_blocked_commands = [s.strip().lower() for s in raw_outbound_blocked if s.strip()]
        except Exception:
            _outbound_blocked_commands = []

        # Rate limit — read from plugins-enabled.json rate_limits.tmux if present
        try:
            import json
            path = os.path.join(os.path.dirname(__file__), "plugins-enabled.json")
            with open(path, "r") as f:
                full_cfg = json.load(f)
            rl = full_cfg.get("rate_limits", {}).get("tmux", {})
            _rate_limit_calls  = int(rl.get("calls", 30))
            _rate_limit_window = int(rl.get("window_seconds", 60))
        except Exception:
            _rate_limit_calls  = 30
            _rate_limit_window = 60

        self.enabled = True
        return True

    def shutdown(self) -> None:
        self.enabled = False
        for name in list(_sessions.keys()):
            sess = _sessions.pop(name, None)
            if not sess:
                continue
            stop_event = sess.get("reader_stop")
            if stop_event:
                stop_event.set()
            try:
                sess["proc"].kill()
            except Exception:
                pass
            try:
                os.close(sess["master"])
            except Exception:
                pass

    def get_commands(self) -> Dict[str, Any]:
        """
        Return !command handlers contributed by this plugin.

        routes.py dispatches !tmux and !tmux_call_limit through these handlers.
        Handler signature: async (args: str) -> str
        The wrapper in routes.py handles push_tok / push_done.
        """
        return {
            "tmux":           tmux_command,
            "tmux_call_limit": tmux_call_limit_command,
        }

    def get_help(self) -> str:
        """Return the !help section for this plugin."""
        return (
            "Shell Sessions (tmux):\n"
            "  !tmux new <name>                          - create a new PTY shell session\n"
            "  !tmux exec <session> <command>            - run a command, wait up to TMUX_EXEC_TIMEOUT seconds\n"
            "  !tmux exec <session> <cmd> &              - background (for long-running scripts/installs)\n"
            "  Note: for scripts that take >TMUX_EXEC_TIMEOUT, use '&' and poll with !tmux buf\n"
            "  !tmux ls                                  - list active sessions\n"
            "  !tmux kill-session <name>                 - terminate a session\n"
            "  !tmux kill-server                         - terminate all sessions\n"
            "  !tmux buf <name>                          - show output buffer (recent history) for a session\n"
            "  !tmux send-keys <name> <key>              - send raw keypress: C-c, C-d, C-z, enter\n"
            "  !tmux history-limit [n]                   - get/set rolling history line limit\n"
            "  !tmux filters                             - show ALLOWED/BLOCKED command filter lists\n"
            "  !tmux_call_limit                          - show current tmux rate limit\n"
            "  !tmux_call_limit <calls> <window_sec>     - set rate limit (auto-disables on breach)\n"
            "  !tmux_gate_write <t|f>                    - gate ALL tmux tools: true=gated, false=auto-allow\n"
            "  !tmux_<toolname>_gate_write <t|f>         - per-tool override (e.g. !tmux_exec_gate_write)\n"
            "  !tmux_call_limit_gate_read <t|f>          - gate tmux_call_limit reads\n"
            "  !tmux_call_limit_gate_write <t|f>         - gate tmux_call_limit writes\n"
            "  !tmux threads                             - show bg reader thread status (diagnostic)\n"
            "  !tmux restart-reader <name>               - force-restart bg reader thread for a session\n"
        )

    def get_tools(self) -> Dict[str, Any]:
        return {
            "lc": [
                StructuredTool.from_function(
                    coroutine=tmux_new_executor,
                    name="tmux_new",
                    description=(
                        "Create a new named PTY bash session. "
                        "Must be called before tmux_exec can use the session."
                    ),
                    args_schema=_TmuxNewArgs,
                ),
                StructuredTool.from_function(
                    coroutine=tmux_exec_executor,
                    name="tmux_exec",
                    description=(
                        "Execute a shell command in a named persistent PTY session. "
                        "State (cwd, env, background jobs) persists between calls. "
                        "For long-running commands: background with '&' and tee to a log, "
                        "then poll with 'tail logfile' + 'jobs'. "
                        "Control chars: \\x03=Ctrl-C, \\x04=Ctrl-D."
                    ),
                    args_schema=_TmuxExecArgs,
                ),
                StructuredTool.from_function(
                    coroutine=tmux_ls_executor,
                    name="tmux_ls",
                    description="List all active PTY sessions with status and history stats.",
                    args_schema=_TmuxLsArgs,
                ),
                StructuredTool.from_function(
                    coroutine=tmux_kill_session_executor,
                    name="tmux_kill_session",
                    description="Terminate a named PTY session and free its resources.",
                    args_schema=_TmuxKillSessionArgs,
                ),
                StructuredTool.from_function(
                    coroutine=tmux_kill_server_executor,
                    name="tmux_kill_server",
                    description="Terminate ALL active PTY sessions.",
                    args_schema=_TmuxKillServerArgs,
                ),
                StructuredTool.from_function(
                    coroutine=tmux_history_executor,
                    name="tmux_history",
                    description=(
                        "Show the rolling output history for a session. "
                        "Useful for reviewing past output without re-executing commands."
                    ),
                    args_schema=_TmuxHistoryArgs,
                ),
                StructuredTool.from_function(
                    coroutine=tmux_history_limit_executor,
                    name="tmux_history_limit",
                    description="Set the rolling history line limit for all sessions.",
                    args_schema=_TmuxHistoryLimitArgs,
                ),
                StructuredTool.from_function(
                    coroutine=tmux_call_limit_executor,
                    name="tmux_call_limit",
                    description=(
                        "Read or set the tmux tool rate limit. "
                        "Call with no args to show current limit. "
                        "Call with calls + window_seconds to update the limit."
                    ),
                    args_schema=_TmuxCallLimitArgs,
                ),
            ]
        }


# ---------------------------------------------------------------------------
# Public API — registered via get_commands() and dispatched by routes.py
# ---------------------------------------------------------------------------

async def tmux_command(args: str) -> str:
    """Dispatch a !tmux subcommand. Returns string for push_tok."""
    parts = args.split(maxsplit=1)
    if not parts:
        return (
            "Usage: !tmux <subcommand> [args]\n"
            "Available: new, exec, ls, kill-session, kill-server, buf, send-keys,\n"
            "           history-limit, filters, threads, restart-reader"
        )
    sub = parts[0].lower().strip()
    args = parts[1].strip() if len(parts) > 1 else ""

    if sub == "new":
        name = args.strip()
        if not name:
            return "Usage: !tmux new <name>"
        if not _valid_name(name):
            return (
                f"ERROR: Invalid session name '{name}'. "
                "Use letters, digits, hyphens, underscores only."
            )
        return await _do_create(name)

    elif sub == "exec":
        # !tmux exec <session> <command>
        parts = args.split(None, 1)
        if len(parts) < 2:
            return "Usage: !tmux exec <session> <command>"
        session_name, command = parts[0], parts[1]
        return await tmux_exec_executor(session_name, command)

    elif sub == "ls":
        return _do_ls()

    elif sub == "kill-session":
        name = args.strip()
        if not name:
            return "Usage: !tmux kill-session <name>"
        return await _do_kill(name)

    elif sub == "kill-server":
        return await _do_kill_all()

    elif sub == "buf":
        name = args.strip()
        if not name:
            return "Usage: !tmux buf <name>"
        err = await _await_session(name)
        if err:
            return err
        return _do_history(name)

    elif sub == "history-limit":
        val = args.strip()
        if not val:
            return f"Current history limit: {_history_limit} lines per session"
        try:
            n = int(val)
            if n < 1:
                raise ValueError
        except ValueError:
            return f"ERROR: Invalid value '{val}'. Must be a positive integer."
        return await tmux_history_limit_executor(n)

    elif sub == "send-keys":
        # !tmux send-keys <session> <key>
        # Supported keys: C-c  C-d  C-z  C-\  enter  <any single char>
        parts = args.split(None, 1)
        if len(parts) < 2:
            return "Usage: !tmux send-keys <session> <key>  (e.g. C-c, C-d, enter)"
        session_name, key = parts[0], parts[1].strip()
        err = await _await_session(session_name)
        if err:
            return err
        _KEY_MAP = {
            "c-c": b"\x03", "c-d": b"\x04", "c-z": b"\x1a",
            "c-\\": b"\x1c", "enter": b"\n", "return": b"\n",
        }
        raw = _KEY_MAP.get(key.lower())
        if raw is None:
            raw = key.encode("utf-8", errors="replace")
        master_fd = _sessions[session_name]["master"]
        try:
            os.write(master_fd, raw)
        except OSError as e:
            return f"ERROR: write failed: {e}"
        return f"Sent {key!r} to session '{session_name}'"

    elif sub == "filters":
        return _format_filter_status()

    elif sub == "threads":
        # Diagnostic: show bg reader thread status for all sessions
        if not _sessions:
            return "No active sessions."
        lines = ["Session reader thread status:"]
        for sname, sess in _sessions.items():
            t = sess.get("reader_thread")
            stop_ev = sess.get("reader_stop")
            if t is None:
                state = "NO THREAD"
            elif t.is_alive():
                state = f"alive (daemon={t.daemon})"
            else:
                state = "DEAD"
            stop_state = "stop_set" if (stop_ev and stop_ev.is_set()) else "running"
            proc_rc = sess["proc"].returncode
            exec_active = sess.get("exec_active", False)
            lines.append(
                f"  {sname:<20}  thread={state}  stop={stop_state}"
                f"  proc_rc={proc_rc}  exec_active={exec_active}"
            )
        return "\n".join(lines)

    elif sub == "restart-reader":
        # Force-restart the bg reader thread for a session (diagnostic/recovery)
        name = args.strip()
        if not name:
            return "Usage: !tmux restart-reader <session>"
        if name not in _sessions:
            return f"ERROR: No session '{name}'"
        sess = _sessions[name]
        old_stop = sess.get("reader_stop")
        if old_stop:
            old_stop.set()
        old_t = sess.get("reader_thread")
        if old_t and old_t.is_alive():
            old_t.join(timeout=1.0)
        stop_event = threading.Event()
        sess["reader_stop"] = stop_event
        sess["exec_active"] = False
        t = threading.Thread(
            target=_bg_reader_thread,
            args=(name, stop_event),
            name=f"tmux-reader-{name}",
            daemon=True,
        )
        t.start()
        sess["reader_thread"] = t
        return f"Reader thread restarted for session '{name}' (tid={t.ident})"

    else:
        return (
            f"Unknown tmux subcommand: '{sub}'\n"
            "Available: new, exec, ls, kill-session, kill-server, buf, send-keys,\n"
            "           history-limit, filters, threads, restart-reader"
        )


async def tmux_call_limit_command(args: str) -> str:
    """
    Dispatch !tmux_call_limit command.
    No args  → show current limit
    <n> <w>  → set calls=n window=w
    """
    args = args.strip()
    if not args:
        return await tmux_call_limit_executor()
    parts = args.split()
    if len(parts) != 2:
        return (
            "Usage: !tmux_call_limit [<calls> <window_seconds>]\n"
            "  No args: show current limit\n"
            "  Example: !tmux_call_limit 20 60"
        )
    try:
        calls = int(parts[0])
        window = int(parts[1])
    except ValueError:
        return "ERROR: Both arguments must be integers."
    return await tmux_call_limit_executor(calls=calls, window_seconds=window)
