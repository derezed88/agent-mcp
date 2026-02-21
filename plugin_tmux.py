"""
Tmux-like Shell Session Plugin for MCP Agent

Provides persistent PTY (pseudo-terminal) shell sessions.
LLMs interact via tool calls; humans manage sessions via !tmux commands.

Config keys in plugins-enabled.json → plugin_config.plugin_tmux:
    TMUX_HISTORY_LIMIT        int   Rolling history lines per session (default 200)
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
import struct
import termios
import time
from collections import deque
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from plugin_loader import BasePlugin
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Module-level state — shared by plugin instance and routes.py cmd_tmux()
# ---------------------------------------------------------------------------

# name -> {proc, master, history, name, started}
_sessions: Dict[str, Dict[str, Any]] = {}

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


def _is_alive(name: str) -> bool:
    sess = _sessions.get(name)
    return bool(sess and sess["proc"].returncode is None)


async def _do_create(name: str) -> str:
    if name in _sessions:
        return f"Session '{name}' already exists."

    master_fd, slave_fd = pty.openpty()
    _resize_pty(master_fd)
    _make_nonblocking(master_fd)

    proc = await asyncio.create_subprocess_exec(
        "/bin/bash", "--login", "-i",
        stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
        start_new_session=True,
    )
    os.close(slave_fd)

    _sessions[name] = {
        "proc":    proc,
        "master":  master_fd,
        "history": deque(maxlen=_history_limit),
        "name":    name,
        "started": time.time(),
    }

    await asyncio.sleep(_STARTUP_DRAIN)
    _drain_sync(master_fd)
    return f"Session '{name}' created (pid {proc.pid})."


async def _do_kill(name: str) -> str:
    sess = _sessions.pop(name, None)
    if not sess:
        return f"No session named '{name}'."
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
        lines.append(
            f"  {name:<20}  pid={sess['proc'].pid}  {status}"
            f"  age={age}s  history={hist}/{sess['history'].maxlen} lines"
        )
    return "\n".join(lines)


def _do_history(name: str, lines: int = 50) -> str:
    sess = _sessions.get(name)
    if not sess:
        return f"No session named '{name}'."
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
    if session not in _sessions:
        names = ", ".join(_sessions.keys()) if _sessions else "(none)"
        return (
            f"ERROR: No tmux session named '{session}'.\n"
            f"Active sessions: {names}\n"
            f"Create one with: tmux_new(name='{session}')"
        )
    if not _is_alive(session):
        return (
            f"ERROR: Session '{session}' process has exited. "
            f"Kill and recreate: tmux_kill_session(name='{session}') then tmux_new(name='{session}')"
        )

    # Command filter check (TMUX_ALLOWED/BLOCKED_COMMANDS)
    block_reason = _check_command(command)
    if block_reason:
        return f"ERROR: {block_reason}\nCommand was NOT sent to the session."

    # Outbound agent blocked command check (OUTBOUND_AGENT_BLOCKED_COMMANDS)
    if _outbound_blocked_commands:
        cmd_lower = command.strip().lower()
        for pattern in _outbound_blocked_commands:
            if cmd_lower.startswith(pattern):
                return (
                    f"ERROR: BLOCKED by OUTBOUND_AGENT_BLOCKED_COMMANDS: "
                    f"command matches blocked pattern '{pattern}'.\n"
                    f"Command was NOT sent to the session."
                )

    timeout = max(1.0, min(float(timeout or 10.0), 120.0))
    sess = _sessions[session]
    master_fd = sess["master"]

    cmd_bytes = (command.rstrip("\n") + "\n").encode("utf-8")
    try:
        os.write(master_fd, cmd_bytes)
    except OSError as e:
        return f"ERROR: Write to session '{session}' failed: {e}"

    output = await _read_output(master_fd, timeout=timeout)
    _append_history(session, f"$ {command}\n" + output)
    return output if output.strip() else "(no output)"


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

        # History limit
        try:
            _history_limit = int(config.get("TMUX_HISTORY_LIMIT", 200))
        except (TypeError, ValueError):
            _history_limit = 200

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
            "  !tmux exec <session> <command>            - run a command in a session\n"
            "  !tmux ls                                  - list active sessions\n"
            "  !tmux kill-session <name>                 - terminate a session\n"
            "  !tmux kill-server                         - terminate all sessions\n"
            "  !tmux buf <name>                          - show output buffer (recent history) for a session\n"
            "  !tmux history-limit [n]                   - get/set rolling history line limit\n"
            "  !tmux filters                             - show ALLOWED/BLOCKED command filter lists\n"
            "  !tmux_call_limit                          - show current tmux rate limit\n"
            "  !tmux_call_limit <calls> <window_sec>     - set rate limit (auto-disables on breach)\n"
            "  !tmux_gate_write <t|f>                    - gate ALL tmux tools: true=gated, false=auto-allow\n"
            "  !tmux_<toolname>_gate_write <t|f>         - per-tool override (e.g. !tmux_exec_gate_write)\n"
            "  !tmux_call_limit_gate_read <t|f>          - gate tmux_call_limit reads\n"
            "  !tmux_call_limit_gate_write <t|f>         - gate tmux_call_limit writes\n"
        )

    def get_gate_tools(self) -> Dict[str, Any]:
        """
        All tmux tools are write-gated.
        Group gate 'tmux' covers all; per-tool gates override for individual tools.
        tmux_call_limit supports both read and write gates.
        """
        entry = {"type": "tmux", "operations": ["write"]}
        return {
            "tmux_new":           {**entry, "description": "create a new PTY shell session"},
            "tmux_exec":          {**entry, "description": "execute a command in a PTY session"},
            "tmux_ls":            {**entry, "description": "list active PTY sessions"},
            "tmux_kill_session":  {**entry, "description": "terminate a named PTY session"},
            "tmux_kill_server":   {**entry, "description": "terminate all PTY sessions"},
            "tmux_history":       {**entry, "description": "show session rolling history"},
            "tmux_history_limit": {**entry, "description": "change the rolling history line limit"},
            "tmux_call_limit":    {"type": "tmux", "operations": ["read", "write"],
                                   "description": "read or set the tmux tool rate limit"},
        }

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
            "Available: new, exec, ls, kill-session, kill-server, buf, history-limit, filters"
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

    elif sub == "filters":
        return _format_filter_status()

    else:
        return (
            f"Unknown tmux subcommand: '{sub}'\n"
            "Available: new, exec, ls, kill-session, kill-server, buf, history-limit, filters"
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
