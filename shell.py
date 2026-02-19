"""
__shell__.py - AIOps Console Client  (Stage 4: Google Search + Buffer Scrolling)

Input model:
  The input area is a single flat string that visually wraps across
  `input_lines` screen rows.  There are no "logical rows" — the buffer
  is just text; the display chunks it by (terminal_width - 2) characters
  and always scrolls to keep the cursor visible.

Key bindings — input:
  Enter / F5 / Ctrl+G  — submit immediately
  [a]                  — allow pending AI tool request  (gate mode only)
  [r]                  — reject pending AI tool request (gate mode only)
  Backspace / Delete   — delete char before cursor
  Left / Right         — move cursor
  Up / Down            — move cursor one visual row up/down
  Home / Ctrl+A        — jump to start
  End  / Ctrl+E        — jump to end

Key bindings — output buffer scrolling:
  PgUp  / Ctrl+B       — scroll output up one page
  PgDn  / Ctrl+F       — scroll output down one page
  Ctrl+End             — jump to bottom (latest output)
  Mouse wheel up/down  — scroll output 3 lines (if mouse capture is ON)

Special client-side commands:
  !input_lines <n>              — resize input area (1-20 rows)
  !mouse [on|off]               — toggle mouse capture (OFF allows text selection)
  !gate_preview_length [n]      — get/set gate preview char limit (default 25)
  !session                      — list all active sessions (shows current)
  !session <ID> attach          — switch to a different session (immediate)
  !session <ID> delete          — delete a session from server
  !exit / !quit                 — exit the shell

Server-side commands (forwarded):
  !model                        — list available models
  !model <key>                  — switch active LLM
  !reset                        — clear conversation history
  !db <sql>                     — run SQL directly (no LLM, no gate)
  !autoAIdb <table> true|false  — toggle AI db gate per table
  !autoAIdb status              — show current gate settings
  !help                         — full help
"""

import asyncio
import curses
import json
import locale
import os
import sys
import uuid
from datetime import datetime

import httpx
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVER_HOST         = os.getenv("AISVC_HOST", "http://127.0.0.1:8765")
SUBMIT_URL          = f"{SERVER_HOST}/submit"
STREAM_URL          = f"{SERVER_HOST}/stream"
GATE_RESPONSE_URL   = f"{SERVER_HOST}/gate_response"

SESSION_FILE        = ".aiops_session_id"
DEFAULT_INPUT_LINES = 3
BORDER_CHAR         = "─"

# ---------------------------------------------------------------------------
# Session Persistence
# ---------------------------------------------------------------------------

def load_or_create_client_id() -> str:
    """Load CLIENT_ID from file, or create new one if file doesn't exist."""
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, 'r') as f:
                client_id = f.read().strip()
                if client_id:  # Validate non-empty
                    return client_id
        except Exception:
            pass  # Fall through to generate new ID
    # Generate new ID
    client_id = str(uuid.uuid4())
    save_client_id(client_id)
    return client_id

def save_client_id(client_id: str):
    """Save CLIENT_ID to file for persistence across restarts."""
    with open(SESSION_FILE, 'w') as f:
        f.write(client_id)

CLIENT_ID = load_or_create_client_id()

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self):
        self.output_lines: list[str] = []

        # ---- flat input model ----
        self.input_text: str       = ""    # entire input as one string
        self.input_cursor_pos: int = 0     # byte offset into input_text
        self.input_lines: int      = DEFAULT_INPUT_LINES  # visual rows available

        self.status: str           = "Ready"
        self.lock                  = asyncio.Lock()
        self.redraw_event          = asyncio.Event()
        self.running: bool         = True
        self.current_model: str    = "gemini25"
        # Human gate
        self.pending_gate: dict | None = None
        # Output buffer scroll  (0 = pinned to bottom; N = scrolled up N display-lines)
        self.output_scroll: int    = 0
        # Mouse capture state
        self.mouse_enabled: bool   = False
        # Gate preview truncation length (chars). Configurable via !gate_preview_length.
        self.gate_preview_length: int = 25
        # Session switching
        self.session_switch_requested: bool = False
        self.new_session_id: str | None = None

    async def append_output(self, text: str):
        async with self.lock:
            for part in text.split("\n"):
                self.output_lines.append(part)
        self.redraw_event.set()

    async def set_status(self, text: str):
        async with self.lock:
            self.status = text
        self.redraw_event.set()

    async def scroll_by(self, delta: int, output_rows: int):
        """Adjust scroll offset. delta>0 scrolls up (older), delta<0 scrolls down (newer)."""
        async with self.lock:
            # Build the full display list length so we can clamp properly.
            # (Rough estimate: each line may wrap; use line count as lower bound.)
            max_scroll = max(0, len(self.output_lines) - max(1, output_rows))
            self.output_scroll = max(0, min(max_scroll, self.output_scroll + delta))
        self.redraw_event.set()

    async def scroll_to_bottom(self):
        async with self.lock:
            self.output_scroll = 0
        self.redraw_event.set()


state = AppState()

# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

async def submit_to_server(text: str):
    payload = {
        "client_id":     CLIENT_ID,
        "text":          text,
        "default_model": state.current_model,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(SUBMIT_URL, json=payload)
            if resp.status_code != 200:
                await state.append_output(
                    f"[ERROR] Server returned {resp.status_code}: {resp.text}"
                )
    except httpx.ConnectError:
        await state.append_output(
            f"[ERROR] Cannot connect to AISvc at {SERVER_HOST}. Is the server running?"
        )
    except Exception as exc:
        await state.append_output(f"[ERROR] Submit failed: {exc}")


async def send_gate_response(gate_id: str, decision: str):
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(GATE_RESPONSE_URL, json={
                "gate_id":  gate_id,
                "decision": decision,
            })
    except Exception as exc:
        await state.append_output(f"[ERROR] Gate response failed: {exc}")


# ---------------------------------------------------------------------------
# Session Management
# ---------------------------------------------------------------------------

async def list_sessions():
    """Fetch and display all sessions from server."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{SERVER_HOST}/sessions")
            if resp.status_code == 200:
                data = resp.json()
                sessions = data.get("sessions", [])
                if not sessions:
                    await state.append_output("\n[shell] No active sessions found.\n")
                    return
                await state.append_output("\nActive sessions:")
                for s in sessions:
                    current = " (current)" if s["client_id"] == CLIENT_ID else ""
                    shorthand = s.get("shorthand_id", "?")
                    cid = s["client_id"]
                    model = s["model"]
                    history = s["history_length"]
                    peer_ip = s.get("peer_ip")
                    ip_str = f", ip={peer_ip}" if peer_ip else ""
                    await state.append_output(
                        f"  ID [{shorthand}] {cid}: model={model}, history={history} messages{ip_str}{current}"
                    )
                await state.append_output("")
            else:
                await state.append_output(f"[ERROR] Server returned {resp.status_code}: {resp.text}")
    except Exception as exc:
        await state.append_output(f"[ERROR] Failed to list sessions: {exc}")

async def delete_session(session_id: str):
    """Delete a session from the server."""
    global CLIENT_ID
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.delete(f"{SERVER_HOST}/session/{session_id}")
            if resp.status_code == 200:
                await state.append_output(f"[shell] Session {session_id[:8]}... deleted.")
                # If deleting current session, generate new ID
                if session_id == CLIENT_ID:
                    CLIENT_ID = str(uuid.uuid4())
                    save_client_id(CLIENT_ID)
                    await state.append_output(f"[shell] New session created: {CLIENT_ID[:8]}...")
            elif resp.status_code == 404:
                await state.append_output(f"[ERROR] Session {session_id[:8]}... not found.")
            else:
                await state.append_output(f"[ERROR] Failed to delete session: {resp.text}")
    except Exception as exc:
        await state.append_output(f"[ERROR] Delete failed: {exc}")

async def attach_session(session_id: str):
    """Switch to a different session."""
    global CLIENT_ID
    if session_id == CLIENT_ID:
        await state.append_output("[shell] Already attached to this session.")
        return

    # Verify session exists
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{SERVER_HOST}/sessions?client_id={session_id}")
            if resp.status_code == 200:
                data = resp.json()
                sessions = data.get("sessions", [])
                if not sessions:
                    await state.append_output(f"[ERROR] Session {session_id[:8]}... not found.")
                    return
            else:
                await state.append_output(f"[ERROR] Failed to verify session: {resp.text}")
                return
    except Exception as exc:
        await state.append_output(f"[ERROR] Failed to verify session: {exc}")
        return

    # Initiate switch
    await state.append_output(f"[shell] Switching to session {session_id[:8]}...")
    save_client_id(session_id)

    async with state.lock:
        state.session_switch_requested = True
        state.new_session_id = session_id
        state.output_lines.clear()  # Clear output for new session

    # SSE listener will pick up the switch request and reconnect


# ---------------------------------------------------------------------------
# SSE listener
# ---------------------------------------------------------------------------

async def sse_listener():
    """
    Persistent SSE connection.  Decodes \\n → real newlines in token data.
    Handles: token stream, done, error, gate_request.
    """
    global CLIENT_ID
    params          = {"client_id": CLIENT_ID}
    current_reply: list[str] = []
    current_event   = "message"

    while state.running:
        # Check for session switch request
        async with state.lock:
            if state.session_switch_requested:
                CLIENT_ID = state.new_session_id
                state.session_switch_requested = False
                state.new_session_id = None
                await state.append_output(f"\n[shell] Reconnecting to session {CLIENT_ID[:8]}...\n")

        # Update params with current CLIENT_ID (may have changed)
        params = {"client_id": CLIENT_ID}

        try:
            await state.set_status("Connecting…")
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", STREAM_URL, params=params) as resp:
                    await state.set_status("Connected")

                    async for line in resp.aiter_lines():
                        if not state.running:
                            return

                        line = line.strip()

                        if not line:
                            current_event = "message"
                            continue

                        # event: field — peel one optional space per SSE spec
                        if line.startswith("event:"):
                            ev = line[6:]
                            current_event = (ev[1:] if ev.startswith(" ") else ev).strip()
                            if current_event == "done":
                                if current_reply:
                                    await state.append_output("")
                                    current_reply.clear()
                                current_event = "message"
                            continue

                        if not line.startswith("data:"):
                            continue

                        # Peel one optional space only (preserve internal spaces)
                        raw = line[5:]
                        if raw.startswith(" "):
                            raw = raw[1:]
                        if not raw:
                            continue

                        # ---- error ----------------------------------------
                        if current_event == "error":
                            try:
                                err = json.loads(raw)
                                await state.append_output(
                                    f"\n[SERVER ERROR] {err.get('error', raw)}"
                                )
                            except Exception:
                                await state.append_output(f"\n[SERVER ERROR] {raw}")
                            current_event = "message"
                            continue

                        # ---- gate_request ----------------------------------
                        if current_event == "gate_request":
                            try:
                                gate_data = json.loads(raw)
                                async with state.lock:
                                    state.pending_gate = gate_data
                                    prev_len = state.gate_preview_length
                                tool_name = gate_data.get("tool_name", "unknown")
                                tool_args = gate_data.get("tool_args", {})
                                raw_tables = gate_data.get("tables", [])

                                def _trunc(text: str, n: int) -> str:
                                    """Truncate to n chars and append ellipsis if cut."""
                                    return text[:n] + ("…" if len(text) > n else "")

                                # Friendly label for __meta__ sentinel
                                def _fmt_table(t: str) -> str:
                                    return "(no-table: SHOW/DESCRIBE/etc.)" if t == "__meta__" else t

                                tables_str = ", ".join(_fmt_table(t) for t in raw_tables)

                                if tool_name == "db_query":
                                    sql = tool_args.get("sql", "")
                                    meta_hint = (
                                        "\n   💡 Tip: '!autoAIdb __meta__ true' to auto-allow these"
                                        if raw_tables == ["__meta__"] else ""
                                    )
                                    await state.append_output(
                                        f"\n🔒 AI tool request: {tool_name}\n"
                                        f"   SQL    : {_trunc(sql, prev_len)}\n"
                                        f"   Tables : {tables_str or '(none detected)'}"
                                        f"{meta_hint}\n"
                                        f"   ► Press [a] to allow  or  [r] to reject"
                                    )
                                elif tool_name == "google_search":
                                    query = tool_args.get("query", "")
                                    await state.append_output(
                                        f"\n🔍 AI web search request: {tool_name}\n"
                                        f"   Query  : {_trunc(query, prev_len)}\n"
                                        f"   (external network call — costs API quota)\n"
                                        f"   ► Press [a] to allow  or  [r] to reject"
                                    )
                                elif tool_name == "update_system_prompt":
                                    operation = tool_args.get("operation", "?")
                                    content   = tool_args.get("content", "")
                                    target    = tool_args.get("target", "")
                                    lines = [
                                        f"\n🔒 AI tool request: {tool_name}",
                                        f"   Operation: {operation}",
                                    ]
                                    if content:
                                        lines.append(f"   Content  : {_trunc(content, prev_len)}")
                                    if target:
                                        lines.append(f"   Target   : {_trunc(target, prev_len)}")
                                    lines.append("   ► Press [a] to allow  or  [r] to reject")
                                    await state.append_output("\n".join(lines))
                                else:
                                    await state.append_output(
                                        f"\n🔒 AI tool request: {tool_name}\n"
                                        f"   Args   : {_trunc(str(tool_args), prev_len)}\n"
                                        f"   ► Press [a] to allow  or  [r] to reject"
                                    )
                            except Exception as exc:
                                await state.append_output(f"[gate parse error] {exc}")
                            current_event = "message"
                            continue

                        # ---- regular token ---------------------------------
                        decoded = raw.replace("\\n", "\n")
                        current_reply.append(decoded)

                        async with state.lock:
                            parts = decoded.split("\n")
                            for i, part in enumerate(parts):
                                if i == 0:
                                    if state.output_lines:
                                        state.output_lines[-1] += part
                                    else:
                                        state.output_lines.append(part)
                                else:
                                    state.output_lines.append(part)
                        state.redraw_event.set()

        except httpx.ConnectError:
            await state.set_status("Server offline — retrying in 3 s…")
            await asyncio.sleep(3)
        except Exception as exc:
            if state.running:
                await state.set_status(f"SSE error: {exc} — retrying in 3 s…")
                await asyncio.sleep(3)


# ---------------------------------------------------------------------------
# User input handler
# ---------------------------------------------------------------------------

async def user_input_handler(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True

    if stripped.startswith("!"):
        parts = stripped[1:].split(maxsplit=1)
        cmd   = parts[0].lower()
        arg   = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("exit", "quit"):
            state.running = False
            return False

        if cmd == "input_lines":
            try:
                n = int(arg)
                if 1 <= n <= 20:
                    async with state.lock:
                        state.input_lines = n
                    await state.append_output(f"[shell] Input area resized to {n} lines.")
                else:
                    await state.append_output("[shell] !input_lines: value must be 1–20.")
            except ValueError:
                await state.append_output("[shell] Usage: !input_lines <integer>")
            return True

        if cmd == "mouse":
            target = arg.lower()
            if target == "on":
                async with state.lock:
                    state.mouse_enabled = True
                await state.append_output("[shell] Mouse capture ON (scrolling enabled, selection disabled).")
            elif target == "off":
                async with state.lock:
                    state.mouse_enabled = False
                await state.append_output("[shell] Mouse capture OFF (scrolling disabled, selection enabled).")
            else:
                # Toggle
                async with state.lock:
                    state.mouse_enabled = not state.mouse_enabled
                    new_mode = state.mouse_enabled
                await state.append_output(f"[shell] Mouse capture {'ON' if new_mode else 'OFF'}.")
            return True

        if cmd == "gate_preview_length":
            if arg:
                try:
                    n = int(arg)
                    if n >= 1:
                        async with state.lock:
                            state.gate_preview_length = n
                        await state.append_output(f"[shell] Gate preview length set to {n} chars.")
                    else:
                        await state.append_output("[shell] !gate_preview_length: value must be >= 1.")
                except ValueError:
                    await state.append_output("[shell] Usage: !gate_preview_length <integer>")
            else:
                async with state.lock:
                    current = state.gate_preview_length
                await state.append_output(f"[shell] Gate preview length: {current} chars.")
            return True

        if cmd == "session":
            if not arg:
                # List all sessions
                await list_sessions()
            else:
                parts = arg.split(maxsplit=1)
                session_id = parts[0]
                action = parts[1] if len(parts) > 1 else "attach"

                if action == "delete":
                    await delete_session(session_id)
                elif action == "attach":
                    await attach_session(session_id)
                else:
                    await state.append_output(f"[shell] Unknown action: {action}. Use 'attach' or 'delete'.")
            return True

        if cmd == "model" and arg:
            state.current_model = arg

    ts = datetime.now().strftime("%H:%M:%S")
    await state.append_output(f"\n[{ts}] You: {stripped}")
    await state.append_output("")   # blank line so LLM response starts fresh
    await submit_to_server(stripped)
    return True


# ---------------------------------------------------------------------------
# Curses rendering
# ---------------------------------------------------------------------------

def _chunk_input(text: str, usable: int) -> list[str]:
    """
    Split flat input text into display chunks of `usable` width.
    Always returns at least one element (empty string if text is empty).
    """
    if usable <= 0:
        return [text] if text else [""]
    if not text:
        return [""]
    chunks = []
    while text:
        chunks.append(text[:usable])
        text = text[usable:]
    return chunks


def _cursor_to_screen(pos: int, usable: int, input_lines: int,
                      border_row: int) -> tuple[int, int]:
    """
    Convert flat cursor position → (screen_row, screen_col).
    Also returns start_chunk (scroll offset) so the cursor chunk is visible.
    """
    if usable <= 0:
        return border_row + 1, 2
    cur_chunk     = pos // usable
    cur_col_in_c  = pos % usable
    # Scroll: ensure cur_chunk is within [start_chunk, start_chunk + input_lines)
    start_chunk   = max(0, cur_chunk - (input_lines - 1))
    screen_row    = border_row + 1 + (cur_chunk - start_chunk)
    screen_col    = 2 + cur_col_in_c          # 2 = len("> " or "  ")
    return screen_row, screen_col, start_chunk


def _draw(stdscr, snap: dict):
    stdscr.erase()
    max_y, max_x = stdscr.getmaxyx()

    input_lines = snap["input_lines"]
    border_row  = max_y - input_lines - 1
    output_rows = border_row

    if output_rows < 1 or border_row < 1:
        stdscr.refresh()
        return

    # ---- Output area -------------------------------------------------------
    display: list[str] = []
    for line in snap["output_lines"]:
        if not line:
            display.append("")
            continue
        while len(line) > max_x:
            display.append(line[:max_x])
            line = line[max_x:]
        display.append(line)

    total_display = len(display)
    scroll        = snap["output_scroll"]

    if scroll == 0:
        # Pinned to bottom — show newest lines
        visible = display[-output_rows:] if total_display >= output_rows else display
    else:
        # Scrolled up — anchor is 'scroll' lines from the bottom
        end   = max(0, total_display - scroll)
        start = max(0, end - output_rows)
        visible = display[start:end]

    for row, line in enumerate(visible):
        if row >= output_rows:
            break
        try:
            stdscr.addstr(row, 0, line[:max_x])
        except Exception:
            pass

    # ---- Separator ---------------------------------------------------------
    if snap["pending_gate"]:
        status_str = " ⚠  AI GATE PENDING — [a]llow / [r]eject "
        attr = curses.A_BOLD | curses.A_REVERSE
    else:
        base_status = snap["status"]
        if scroll > 0:
            lines_above = max(0, total_display - output_rows - scroll)
            base_status = f"{base_status}  [↑{scroll} scrolled — ↑{lines_above} more above]"
        elif total_display > output_rows:
            lines_above = total_display - output_rows
            base_status = f"{base_status}  [↑{lines_above} above — PgUp to scroll]"
        status_str = f" {base_status} "
        attr = curses.A_BOLD

    fill_len = max(0, max_x - len(status_str))
    sep_line = (status_str + BORDER_CHAR * fill_len)[:max_x]
    try:
        stdscr.addstr(border_row, 0, sep_line, attr)
    except Exception:
        pass

    # ---- Input area --------------------------------------------------------
    usable = max_x - 2         # 2 chars for "> " / "  " prefix

    if snap["pending_gate"]:
        hint = "  (input blocked — resolve gate above)"[:max_x]
        try:
            stdscr.addstr(border_row + 1, 0, hint, curses.A_DIM)
        except Exception:
            pass
    else:
        text      = snap["input_text"]
        pos       = snap["input_cursor_pos"]
        chunks    = _chunk_input(text, usable)

        # Scroll offset so cursor chunk is always visible
        cur_chunk    = pos // usable if usable > 0 else 0
        start_chunk  = max(0, cur_chunk - (input_lines - 1))

        for i in range(input_lines):
            row = border_row + 1 + i
            if row >= max_y:
                break
            chunk_idx = start_chunk + i
            chunk     = chunks[chunk_idx] if chunk_idx < len(chunks) else ""
            prefix    = "> " if (start_chunk == 0 and i == 0) else "  "
            try:
                stdscr.addstr(row, 0, (prefix + chunk)[:max_x])
            except Exception:
                pass

        # Cursor
        col_in_chunk = pos % usable if usable > 0 else 0
        screen_row   = border_row + 1 + (cur_chunk - start_chunk)
        screen_col   = 2 + col_in_chunk
        try:
            stdscr.move(
                min(screen_row, max_y - 1),
                min(screen_col, max_x - 1),
            )
        except Exception:
            pass

    stdscr.refresh()


def _snapshot(st: AppState) -> dict:
    return {
        "output_lines":     list(st.output_lines),
        "output_scroll":    st.output_scroll,
        "input_text":       st.input_text,
        "input_cursor_pos": st.input_cursor_pos,
        "input_lines":      st.input_lines,
        "status":           st.status,
        "pending_gate":     st.pending_gate,
    }


# ---------------------------------------------------------------------------
# Input loop
# ---------------------------------------------------------------------------

async def input_loop(stdscr):
    stdscr.nodelay(True)
    curses.cbreak()
    stdscr.keypad(True)
    curses.noecho()

    # Track local mouse state to detect changes from !mouse command
    current_mouse_mode = True

    # Enable mouse events for wheel scrolling initially
    try:
        curses.mousemask(
            curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION
        )
    except Exception:
        pass

    loop = asyncio.get_event_loop()

    def _do_submit():
        text = state.input_text.strip()
        if not text:
            return

        async def _task():
            async with state.lock:
                state.input_text       = ""
                state.input_cursor_pos = 0

            # Handle multi-line paste: split by newlines and submit each line
            lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
            for line in lines:
                line = line.strip()
                if line:  # Skip empty lines
                    await user_input_handler(line)

        loop.create_task(_task())

    while state.running:
        # ---- Sync mouse state ----------------------------------------------
        # If the user toggled mouse mode via "!mouse off", we must disable
        # the mask so the terminal performs native selection.
        if state.mouse_enabled != current_mouse_mode:
            current_mouse_mode = state.mouse_enabled
            if current_mouse_mode:
                try:
                    curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
                    sys.stdout.write("\033[?1003h")
                    sys.stdout.flush()
                except Exception: pass
            else:
                try:
                    curses.mousemask(0)
                    sys.stdout.write("\033[?1003l")
                    sys.stdout.flush()
                except Exception: pass

        try:
            ch = stdscr.getch()
        except Exception:
            ch = -1

        if ch == -1:
            await asyncio.sleep(0.02)
            continue

        # ---- Detect paste: collect multiple rapid characters --------------
        # If characters are arriving rapidly (paste), collect them all first
        paste_buffer = []
        if ch != -1:
            paste_buffer.append(ch)
            # Check if more characters are immediately available (paste scenario)
            try:
                while True:
                    next_ch = stdscr.getch()
                    if next_ch == -1:
                        break
                    paste_buffer.append(next_ch)
                    if len(paste_buffer) > 1000:  # Safety limit
                        break
            except Exception:
                pass

        # If we collected multiple characters, it's likely a paste
        if len(paste_buffer) > 5:  # Threshold for paste detection
            # Convert to string and handle newlines
            paste_text = ''.join(chr(c) if 32 <= c <= 126 or c in (10, 13) else '' for c in paste_buffer)
            # Add to current input
            async with state.lock:
                current_text = state.input_text
                pos = state.input_cursor_pos
                state.input_text = current_text[:pos] + paste_text + current_text[pos:]
                state.input_cursor_pos = pos + len(paste_text)

            # If paste contains newlines, trigger submission
            if '\n' in paste_text or '\r' in paste_text:
                loop.create_task(state.scroll_to_bottom())
                _do_submit()

            state.redraw_event.set()
            continue

        # Process single character normally
        ch = paste_buffer[0] if paste_buffer else -1
        if ch == -1:
            continue

        # ---- Compute output rows for scroll page size ----------------------
        max_y, max_x = stdscr.getmaxyx()
        output_rows  = max(1, max_y - state.input_lines - 1)
        page_size    = max(1, output_rows - 1)

        # ---- Scroll output buffer ------------------------------------------
        if ch == curses.KEY_PPAGE or ch == 2:      # PgUp / Ctrl+B
            loop.create_task(state.scroll_by(page_size, output_rows))
            state.redraw_event.set()
            continue

        if ch == curses.KEY_NPAGE or ch == 6:      # PgDn / Ctrl+F
            loop.create_task(state.scroll_by(-page_size, output_rows))
            state.redraw_event.set()
            continue

        if ch == 533 or ch == 566:                 # Ctrl+End (terminal-dependent codes)
            loop.create_task(state.scroll_to_bottom())
            state.redraw_event.set()
            continue

        # ---- Mouse events --------------------------------------------------
        if ch == curses.KEY_MOUSE:
            try:
                _, mx, my, mz, bstate = curses.getmouse()
                # Button 4 = wheel up, Button 5 = wheel down
                # (some terminals report as BUTTON4_PRESSED / BUTTON5_PRESSED)
                scroll_up   = bstate & 0x00080000   # BUTTON4_PRESSED
                scroll_down = bstate & 0x00800000   # BUTTON5_PRESSED
                if scroll_up:
                    loop.create_task(state.scroll_by(3, output_rows))
                    state.redraw_event.set()
                elif scroll_down:
                    loop.create_task(state.scroll_by(-3, output_rows))
                    state.redraw_event.set()
            except Exception:
                pass
            continue

        # ---- Human gate mode -----------------------------------------------
        async with state.lock:
            gate = state.pending_gate

        if gate is not None:
            if ch in (ord('a'), ord('A')):
                async with state.lock:
                    state.pending_gate = None
                await state.append_output("[shell] ✓ Allowed — AI tool proceeding.")
                loop.create_task(send_gate_response(gate["gate_id"], "allow"))
            elif ch in (ord('r'), ord('R')):
                async with state.lock:
                    state.pending_gate = None
                await state.append_output("[shell] ✗ Rejected — AI tool blocked.")
                loop.create_task(send_gate_response(gate["gate_id"], "reject"))
            state.redraw_event.set()
            continue

        # ---- Normal mode ---------------------------------------------------
        # Compute usable width for Up/Down movement
        usable = max(1, max_x - 2)

        async with state.lock:
            text = state.input_text
            pos  = state.input_cursor_pos

        # Submit
        if ch in (curses.KEY_ENTER, 10, 13, curses.KEY_F5, 7):
            # Any input submission jumps to bottom
            loop.create_task(state.scroll_to_bottom())
            _do_submit()

        # Backspace / Delete
        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            if pos > 0:
                async with state.lock:
                    state.input_text       = text[:pos-1] + text[pos:]
                    state.input_cursor_pos = pos - 1

        elif ch == curses.KEY_DC:   # Delete key — delete char after cursor
            if pos < len(text):
                async with state.lock:
                    state.input_text = text[:pos] + text[pos+1:]
                # cursor stays

        # Left
        elif ch == curses.KEY_LEFT:
            if pos > 0:
                async with state.lock:
                    state.input_cursor_pos = pos - 1

        # Right
        elif ch == curses.KEY_RIGHT:
            if pos < len(text):
                async with state.lock:
                    state.input_cursor_pos = pos + 1

        # Up — move cursor one visual row up
        elif ch == curses.KEY_UP:
            new_pos = max(0, pos - usable)
            async with state.lock:
                state.input_cursor_pos = new_pos

        # Down — move cursor one visual row down
        elif ch == curses.KEY_DOWN:
            new_pos = min(len(text), pos + usable)
            async with state.lock:
                state.input_cursor_pos = new_pos

        # Home / Ctrl+A
        elif ch in (curses.KEY_HOME, 1):
            async with state.lock:
                state.input_cursor_pos = 0

        # End / Ctrl+E
        elif ch in (curses.KEY_END, 5):
            async with state.lock:
                state.input_cursor_pos = len(state.input_text)

        # Printable ASCII
        elif 32 <= ch <= 126:
            async with state.lock:
                state.input_text       = text[:pos] + chr(ch) + text[pos:]
                state.input_cursor_pos = pos + 1

        state.redraw_event.set()

    state.running = False


# ---------------------------------------------------------------------------
# Render loop
# ---------------------------------------------------------------------------

async def render_loop(stdscr):
    while state.running:
        await state.redraw_event.wait()
        state.redraw_event.clear()
        snap = _snapshot(state)
        try:
            _draw(stdscr, snap)
        except Exception:
            pass
        await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def async_main(stdscr):
    curses.curs_set(1)

    await state.append_output(
        "AIOps Shell  |  PgUp/PgDn=scroll  !mouse=selection toggle  !help=commands  !exit=quit"
    )
    await state.append_output(
        f"Session: {CLIENT_ID[:8]}…   Server: {SERVER_HOST}   Model: {state.current_model}"
    )
    await state.append_output("")

    await asyncio.gather(
        sse_listener(),
        input_loop(stdscr),
        render_loop(stdscr),
    )


def main():
    # Set locale so curses handles UTF-8 and non-ASCII characters correctly.
    locale.setlocale(locale.LC_ALL, "")

    # Enable xterm-style extended mouse reporting (button 4/5 = wheel up/down)
    # Must be written BEFORE curses.wrapper takes over stdout.
    try:
        import sys
        sys.stdout.write("\033[?1003h")
        sys.stdout.flush()
    except Exception:
        pass

    try:
        curses.wrapper(lambda stdscr: asyncio.run(async_main(stdscr)))
    except KeyboardInterrupt:
        pass
    finally:
        # Restore terminal: disable extended mouse reporting
        try:
            import sys
            sys.stdout.write("\033[?1003l")
            sys.stdout.flush()
        except Exception:
            pass
        print("\nAIOps Shell exited.")


if __name__ == "__main__":
    main()