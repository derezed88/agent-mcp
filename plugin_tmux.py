"""
Tmux-like Shell Session Plugin for MCP Agent

Provides persistent PTY (pseudo-terminal) shell sessions.
LLMs interact via tool calls; humans manage sessions via !tmux commands.

Sessions live in memory with a configurable rolling history buffer.
TMUX_HISTORY_LIMIT is set in plugins-enabled.json under
plugin_config.plugin_tmux.TMUX_HISTORY_LIMIT (default 200).

Tools (all gated):
    tmux_new(name)                       — create a new shell session      [write]
    tmux_exec(session, command, timeout) — run a command, return output    [write]
    tmux_ls()                            — list active sessions             [read]
    tmux_kill_session(name)              — terminate one session            [write]
    tmux_kill_server()                   — terminate all sessions           [write]
    tmux_history(session, lines)         — show rolling history             [read]
    tmux_history_limit(n)                — change history line limit        [write]

User commands (dispatched from routes.py cmd_tmux):
    !tmux new <name>
    !tmux ls
    !tmux kill-session <name>
    !tmux kill-server
    !tmux a <name>
    !tmux history-limit <n>

Gate:
    Read  operations (tmux_ls, tmux_history)       — !tmux_gate_read  <t|f>
    Write operations (all others)                  — !tmux_gate_write <t|f>
    Both default to gated (gate ON).
"""

import asyncio
import fcntl
import os
import pty
import re
import signal
import struct
import termios
import time
from collections import deque
from typing import Any, Dict, Optional

from langchain_core.tools import StructuredTool
from plugin_loader import BasePlugin
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Module-level session registry
# Shared by the plugin instance and the !tmux command handler in routes.py.
# ---------------------------------------------------------------------------

# name -> {
#   "proc":    asyncio.subprocess.Process,
#   "master":  int  (PTY master fd),
#   "history": deque[str],
#   "name":    str,
#   "started": float,
# }
_sessions: Dict[str, Dict[str, Any]] = {}

# Rolling history limit (lines).  Updated by !tmux history-limit / tmux_history_limit.
_history_limit: int = 200

# --- Tuning constants -------------------------------------------------------
_READ_TIMEOUT_DEFAULT: float = 10.0   # seconds to wait for output silence
_DRAIN_PAUSE: float = 0.25            # seconds of silence = "command done"
_STARTUP_DRAIN: float = 0.5           # seconds to discard bash startup noise
_CHUNK: int = 4096                    # bytes per PTY read
_PTY_COLS: int = 220                  # terminal width reported to programs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _valid_name(name: str) -> bool:
    """Session names: letters, digits, hyphens, underscores only."""
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
    """Read whatever is currently buffered without blocking."""
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
    """Decode PTY bytes, stripping ANSI/VT100 escape sequences."""
    text = raw.decode("utf-8", errors="replace")
    ansi = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    text = ansi.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


async def _read_output(master_fd: int,
                       timeout: float = _READ_TIMEOUT_DEFAULT,
                       drain_pause: float = _DRAIN_PAUSE) -> str:
    """
    Async PTY drain.

    Waits up to *timeout* seconds for the first byte, then keeps reading
    with *drain_pause* silence threshold.  Returns decoded, ANSI-stripped text.
    """
    loop = asyncio.get_event_loop()
    buf = b""
    got_data = False
    deadline = loop.time() + timeout

    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        wait = drain_pause if got_data else remaining
        try:
            chunk = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: os.read(master_fd, _CHUNK)),
                timeout=wait,
            )
        except asyncio.TimeoutError:
            break
        except OSError:
            break
        if chunk:
            buf += chunk
            got_data = True
            deadline = loop.time() + drain_pause  # reset silence window

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
    """Spawn a bash shell in a new PTY; register under *name*."""
    if name in _sessions:
        return f"Session '{name}' already exists."

    master_fd, slave_fd = pty.openpty()
    _resize_pty(master_fd)
    _make_nonblocking(master_fd)

    proc = await asyncio.create_subprocess_exec(
        "/bin/bash", "--login", "-i",
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
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
    _drain_sync(master_fd)   # discard bash startup banner

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
    results = [await _do_kill(n) for n in names]
    return "\n".join(results)


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
    """Create a new named PTY shell session."""
    name = name.strip()
    if not _valid_name(name):
        return f"ERROR: Invalid session name '{name}'. Use letters, digits, hyphens, underscores only."
    return await _do_create(name)


class _TmuxExecArgs(BaseModel):
    session: str = Field(description="Target session name")
    command: str = Field(
        description=(
            "Shell command to send. For long-running jobs use '&' to background "
            "and tee output to a log file. Control chars: \\x03=Ctrl-C, \\x04=Ctrl-D."
        )
    )
    timeout: Optional[float] = Field(
        default=10.0,
        description=(
            "Seconds to wait for output silence (default 10). "
            "Increase for slow commands. Max 120."
        ),
    )


async def tmux_exec_executor(session: str, command: str, timeout: float = 10.0) -> str:
    """Send a command to a PTY session and return captured output."""
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
    """List all active PTY sessions."""
    return _do_ls()


class _TmuxKillSessionArgs(BaseModel):
    name: str = Field(description="Session name to terminate")


async def tmux_kill_session_executor(name: str) -> str:
    """Terminate a named PTY session."""
    return await _do_kill(name.strip())


class _TmuxKillServerArgs(BaseModel):
    pass


async def tmux_kill_server_executor() -> str:
    """Terminate all PTY sessions."""
    return await _do_kill_all()


class _TmuxHistoryArgs(BaseModel):
    session: str = Field(description="Session name")
    lines: Optional[int] = Field(default=50, description="Number of recent lines to show (default 50)")


async def tmux_history_executor(session: str, lines: int = 50) -> str:
    """Show the rolling history buffer for a session."""
    return _do_history(session.strip(), int(lines or 50))


class _TmuxHistoryLimitArgs(BaseModel):
    n: int = Field(description="New rolling history limit (lines per session, >= 1)")


async def tmux_history_limit_executor(n: int) -> str:
    """Set the rolling history line limit for all sessions."""
    global _history_limit
    if n < 1:
        return "ERROR: History limit must be >= 1."
    _history_limit = n
    for sess in _sessions.values():
        old = list(sess["history"])
        sess["history"] = deque(old[-n:], maxlen=n)
    return f"History limit set to {n} lines. ({len(_sessions)} session(s) updated)"


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------

class TmuxPlugin(BasePlugin):
    """PTY shell session manager — tmux-like sessions for LLM shell interaction."""

    PLUGIN_NAME = "plugin_tmux"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "data_tool"
    DESCRIPTION = "PTY shell sessions with rolling history (tmux-like)"
    DEPENDENCIES = []
    ENV_VARS = []

    def __init__(self):
        self.enabled = False

    def init(self, config: dict) -> bool:
        global _history_limit
        limit = config.get("TMUX_HISTORY_LIMIT", config.get("tmux_history_limit", 200))
        try:
            _history_limit = int(limit)
        except (TypeError, ValueError):
            _history_limit = 200
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

    def get_gate_tools(self) -> Dict[str, Any]:
        """
        Read tools: tmux_ls, tmux_history  — gated via tmux_gate_read
        Write tools: all others            — gated via tmux_gate_write
        """
        return {
            "tmux_new":           {"type": "tmux", "operations": ["write"],
                                   "description": "create a new PTY shell session"},
            "tmux_exec":          {"type": "tmux", "operations": ["write"],
                                   "description": "execute a command in a PTY session"},
            "tmux_ls":            {"type": "tmux", "operations": ["read"],
                                   "description": "list active PTY sessions"},
            "tmux_kill_session":  {"type": "tmux", "operations": ["write"],
                                   "description": "terminate a named PTY session"},
            "tmux_kill_server":   {"type": "tmux", "operations": ["write"],
                                   "description": "terminate all PTY sessions"},
            "tmux_history":       {"type": "tmux", "operations": ["read"],
                                   "description": "show session rolling history"},
            "tmux_history_limit": {"type": "tmux", "operations": ["write"],
                                   "description": "change the rolling history line limit"},
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
                        "Useful for reviewing what ran without re-executing commands."
                    ),
                    args_schema=_TmuxHistoryArgs,
                ),
                StructuredTool.from_function(
                    coroutine=tmux_history_limit_executor,
                    name="tmux_history_limit",
                    description="Set the rolling history line limit for all sessions.",
                    args_schema=_TmuxHistoryLimitArgs,
                ),
            ]
        }


# ---------------------------------------------------------------------------
# Public API for routes.py cmd_tmux()
# ---------------------------------------------------------------------------

async def tmux_command(subcommand: str, args: str) -> str:
    """
    Dispatch a !tmux subcommand from the routes layer.
    Returns a string suitable for push_tok.
    """
    sub = subcommand.lower().strip()

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

    elif sub == "ls":
        return _do_ls()

    elif sub == "kill-session":
        name = args.strip()
        if not name:
            return "Usage: !tmux kill-session <name>"
        return await _do_kill(name)

    elif sub == "kill-server":
        return await _do_kill_all()

    elif sub == "a":
        name = args.strip()
        if not name:
            return "Usage: !tmux a <name>"
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

    else:
        return (
            f"Unknown tmux subcommand: '{sub}'\n"
            "Available: new, ls, kill-session, kill-server, a, history-limit"
        )
