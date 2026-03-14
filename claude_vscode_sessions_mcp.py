#!/usr/bin/env python3
"""
claude_vscode_sessions_mcp: MCP server bridging Claude Code VSCode chat to Google Drive.

Tools exposed to Claude Code:
  claude_sessions_list   - list local Claude Code sessions (always local, no agent-mcp needed)
  claude_sessions_read   - read session content into context (full or agent-summarized, no Drive write)
  gdrive_list            - list Drive files
  gdrive_read            - read a Drive file (full | summary | extract)
  gdrive_snippet_save    - save verbatim content to a Drive topic file
  gdrive_sessions_export - export selected sessions to Drive (full or summarized)

Drive operations fall back to direct Google API when agent-mcp is unreachable or returns 404.
Session listing always runs locally.
Session export summary mode delegates to agent-mcp /vscode/sessions/read when available.
"""

import os
import sys
import json
import glob
import asyncio
import httpx
from typing import Optional
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Direct Drive fallback — imported lazily so missing google libs don't break
# server startup when agent-mcp is the primary path.
_drive = None

def _get_drive():
    global _drive
    if _drive is None:
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location(
            "drive",
            pathlib.Path(__file__).parent / "drive.py",
        )
        _drive = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_drive)
    return _drive

import mcp.server.stdio
from mcp.server import Server
from mcp.types import Tool, TextContent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AGENT_MCP_URL   = os.getenv("AGENT_MCP_URL", "http://localhost:8767").rstrip("/")
AGENT_MCP_TOKEN = os.getenv("AGENT_MCP_TOKEN", "")
CLAUDE_PROJECTS_DIR = os.path.expanduser("~/.claude/projects")


def _utc_ts_to_local_date(ts: str) -> str:
    """Convert a UTC ISO 8601 timestamp to a local YYYY-MM-DD date string."""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d")
    except Exception:
        return ts[:10]

app = Server("claude_vscode_sessions_mcp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auth_headers() -> dict:
    if AGENT_MCP_TOKEN:
        return {"Authorization": f"Bearer {AGENT_MCP_TOKEN}"}
    return {}


def _agent_mcp_unavailable(exc: Exception) -> bool:
    """Return True for any error that should trigger direct Drive fallback."""
    if isinstance(exc, httpx.ConnectError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        # 404 = endpoint not implemented in this agent-mcp version
        return exc.response.status_code == 404
    return False


async def _api_get(path: str, params: dict = None) -> dict:
    url = f"{AGENT_MCP_URL}{path}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(url, headers=_auth_headers(), params=params or {})
        resp.raise_for_status()
        return resp.json()


async def _api_post(path: str, body: dict) -> dict:
    url = f"{AGENT_MCP_URL}{path}"
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, headers=_auth_headers(), json=body)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Local session listing (runs without agent-mcp)
# ---------------------------------------------------------------------------

def _local_sessions_list(date_filter: str = "", project_filter: str = "") -> list:
    sessions = []
    pattern = os.path.join(CLAUDE_PROJECTS_DIR, "*", "*.jsonl")

    for jsonl_path in glob.glob(pattern):
        # Skip subagent files (nested deeper than project/session.jsonl)
        rel = jsonl_path.replace(CLAUDE_PROJECTS_DIR + "/", "")
        if rel.count("/") > 1:
            continue

        session_id = os.path.splitext(os.path.basename(jsonl_path))[0]
        project_dir = os.path.basename(os.path.dirname(jsonl_path))
        cwd = None
        first_ts = None
        last_ts = None
        title = None
        message_count = 0

        try:
            with open(jsonl_path) as f:
                for raw in f:
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        continue

                    ts = obj.get("timestamp")
                    if ts:
                        if first_ts is None:
                            first_ts = ts
                        last_ts = ts

                    if obj.get("type") in ("user", "assistant"):
                        message_count += 1

                    if cwd is None:
                        cwd = obj.get("cwd")

                    if title is None and obj.get("type") == "user":
                        content = obj.get("message", {}).get("content", [])
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text":
                                text = c["text"].strip()
                                # Skip IDE-injected context blocks
                                if text and "<ide_" not in text:
                                    title = text[:200]
                                    break
        except Exception:
            continue

        if date_filter and not _utc_ts_to_local_date(last_ts or first_ts or "").startswith(date_filter):
            continue
        if project_filter and project_filter.lower() not in (cwd or project_dir).lower():
            continue

        try:
            file_bytes = os.path.getsize(jsonl_path)
        except Exception:
            file_bytes = 0

        sessions.append({
            "session_id": session_id,
            "project_path": cwd or project_dir,
            "title": title or "(no title)",
            "first_timestamp": first_ts or "",
            "last_timestamp": last_ts or "",
            "message_count": message_count,
            "file_bytes": file_bytes,
        })

    sessions.sort(key=lambda s: s["last_timestamp"] or s["first_timestamp"])
    # Filter out orphaned snapshot-only files (no actual chat messages)
    return [s for s in sessions if s["message_count"] > 0]


def _format_sessions(sessions: list) -> str:
    if not sessions:
        return "No sessions found."
    lines = [f"Found {len(sessions)} session(s):\n"]
    for s in sessions:
        kb = s.get("file_bytes", 0) / 1024
        size_str = f"{kb:.0f}kB"
        lines.append(
            f"  [{_utc_ts_to_local_date(s['last_timestamp'] or s['first_timestamp']) if (s['last_timestamp'] or s['first_timestamp']) else '?'}] "
            f"{s['project_path'].split('/')[-1]:<28} "
            f"ID: {s['session_id']}  "
            f"{size_str:>7}  "
            f"\"{s['title'][:50]}\""
        )
    return "\n".join(lines)


def _read_session_text_local(jsonl_path: str) -> str:
    """Extract user+assistant text from a JSONL session file."""
    body_text = ""
    try:
        with open(jsonl_path) as fh:
            for raw in fh:
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                if obj.get("type") not in ("user", "assistant"):
                    continue
                role = obj.get("message", {}).get("role", "")
                msg_content = obj.get("message", {}).get("content", [])
                text = ""
                if isinstance(msg_content, str):
                    text = msg_content
                elif isinstance(msg_content, list):
                    for c in msg_content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            text += c["text"]
                if text.strip():
                    body_text += f"\n[{role.upper()}]\n{text.strip()}\n"
    except Exception as e:
        body_text = f"(could not read session file: {e})"
    return body_text


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="claude_sessions_list",
            description=(
                "List all local Claude Code chat sessions across all VSCode projects. "
                "Returns session IDs, titles (first user message), timestamps, and project paths. "
                "Use this before claude_sessions_read (to read/summarize) or gdrive_sessions_export (to save to Drive). "
                "Runs locally — does not require agent-mcp."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Filter by date prefix, e.g. '2026-02-24'. Leave empty for all."
                    },
                    "project": {
                        "type": "string",
                        "description": "Filter by partial project path, e.g. 'agent-mcp'. Leave empty for all."
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="claude_sessions_read",
            description=(
                "Read one or more local Claude Code sessions into the current chat context. "
                "No Drive write — content is returned directly here. "
                "Use this to review, summarize, or reason over past sessions without saving anywhere. "
                "mode='full' (default/preferred): returns verbatim user+assistant text; you summarize in-context. "
                "mode='summary': delegates summarization to agent-mcp LLM before returning — use only if the "
                "session is too large to fit in context. "
                "model: agent-mcp model key for mode='summary' (e.g. 'nuc11Localtokens', 'gemini25fl'). "
                "Empty = agent-mcp default model. "
                "Workflow: call claude_sessions_list first to find session IDs, then call this tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of session UUIDs or 8-char prefixes (from claude_sessions_list)."
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["full", "summary"],
                        "description": "full (default): verbatim text, you summarize. summary: agent-mcp pre-summarizes (extra API call)."
                    },
                    "model": {
                        "type": "string",
                        "description": "agent-mcp model key for mode='summary'. Empty = agent-mcp default."
                    }
                },
                "required": ["session_ids"]
            }
        ),
        Tool(
            name="gdrive_list",
            description="List files in the configured Google Drive folder.",
            inputSchema={
                "type": "object",
                "properties": {
                    "folder_id": {
                        "type": "string",
                        "description": "Drive folder ID. Leave empty to use configured default."
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="gdrive_read",
            description=(
                "Read a file from Google Drive into the current chat context. "
                "mode='full' returns all content verbatim. "
                "mode='summary' returns an LLM-generated summary preserving technical details. "
                "mode='extract' returns only content matching extract_prompt (e.g. 'iptables commands only')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "Drive file ID (from gdrive_list)"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["full", "summary", "extract"],
                        "description": "Retrieval mode: full|summary|extract (default: full)"
                    },
                    "extract_prompt": {
                        "type": "string",
                        "description": "Required when mode=extract. What to extract, e.g. 'only iptables commands'"
                    }
                },
                "required": ["file_id"]
            }
        ),
        Tool(
            name="gdrive_snippet_save",
            description=(
                "Save verbatim content to a named topic file in Google Drive. "
                "If file_id is provided, appends to an existing file. "
                "Otherwise creates a new file named 'snippet-<topic>.txt'. "
                "Use this to preserve specific commands, syntax, or technical exchanges exactly."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The verbatim text content to save"
                    },
                    "topic": {
                        "type": "string",
                        "description": "Topic name for the file (e.g. 'linux-networking', 'iptables'). Used as filename when creating."
                    },
                    "file_id": {
                        "type": "string",
                        "description": "Existing Drive file ID to append to (optional)"
                    },
                    "folder_id": {
                        "type": "string",
                        "description": "Drive folder ID override (optional)"
                    }
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="gdrive_sessions_export",
            description=(
                "Export one or more Claude Code chat sessions to a single file in Google Drive. "
                "Use this whenever the user asks to push, export, save, or archive a session to Drive — "
                "including when they ask agent-mcp to summarize and save a session. "
                "Pass session_ids from claude_sessions_list. "
                "mode='full' writes raw conversation verbatim — zero Claude tokens spent on session content. "
                "mode='summary' generates LLM summaries per session preserving technical details. "
                "summarizer='claude' — Claude Code in VSCode summarizes in-context (uses VSCode tokens). "
                "summarizer='agent' — agent-mcp reads, summarizes, and writes to Drive entirely off-context "
                "(zero Claude tokens spent on session content; requires agent-mcp). "
                "model selects which agent-mcp model to use when summarizer='agent' (e.g. 'summarizer', 'gemini25fl'). "
                "Sessions are assembled in chronological order with title headers."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of session UUIDs to export (from claude_sessions_list)"
                    },
                    "filename": {
                        "type": "string",
                        "description": "Drive filename (optional, defaults to 'claude-sessions-YYYY-MM-DD.txt')"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["full", "summary"],
                        "description": "Export mode: full|summary (default: full)"
                    },
                    "summarizer": {
                        "type": "string",
                        "enum": ["claude", "agent"],
                        "description": (
                            "Who summarizes when mode=summary. "
                            "'claude' — Claude Code summarizes in-context (default). "
                            "'agent' — agent-mcp LLM summarizes off-context (saves VSCode tokens)."
                        )
                    },
                    "model": {
                        "type": "string",
                        "description": (
                            "agent-mcp model key for summarization when summarizer='agent'. "
                            "E.g. 'nuc11Local', 'gemini25fl'. Leave empty for agent-mcp default."
                        )
                    },
                    "folder_id": {
                        "type": "string",
                        "description": "Drive folder ID override (optional)"
                    }
                },
                "required": ["session_ids"]
            }
        ),
    ]


# ---------------------------------------------------------------------------
# Tool call handler
# ---------------------------------------------------------------------------

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        result = await _dispatch(name, arguments)
    except Exception as e:
        result = f"ERROR: {e}"

    return [TextContent(type="text", text=str(result))]


async def _dispatch(name: str, args: dict) -> str:
    # -----------------------------------------------------------------------
    # claude_sessions_list — always local
    # -----------------------------------------------------------------------
    if name == "claude_sessions_list":
        date_filter    = args.get("date", "")
        project_filter = args.get("project", "")
        sessions = await asyncio.to_thread(
            _local_sessions_list, date_filter, project_filter
        )
        return _format_sessions(sessions)

    # -----------------------------------------------------------------------
    # claude_sessions_read — returns content directly, no Drive write
    # -----------------------------------------------------------------------
    elif name == "claude_sessions_read":
        import pathlib
        session_ids = args.get("session_ids", [])
        if not session_ids:
            return "ERROR: session_ids is required"
        mode  = args.get("mode", "full")
        model = args.get("model", "")

        all_sessions = await asyncio.to_thread(_local_sessions_list, "", "")
        session_map  = {s["session_id"]: s for s in all_sessions}

        # Support 8-char prefix matching
        def _resolve_id(sid: str) -> Optional[str]:
            if sid in session_map:
                return sid
            matches = [k for k in session_map if k.startswith(sid)]
            return matches[0] if len(matches) == 1 else None

        parts   = []
        missing = []
        for sid in session_ids:
            full_id = _resolve_id(sid)
            if not full_id:
                missing.append(sid)
                continue
            s = session_map[full_id]
            header = (
                f"{'='*70}\n"
                f"Session: {s['title'][:80]}\n"
                f"Project: {s['project_path']}\n"
                f"Date:    {s.get('first_timestamp', 'unknown')}\n"
                f"ID:      {s['session_id']}\n"
                f"{'='*70}\n"
            )
            jsonl_path = None
            for p in pathlib.Path(CLAUDE_PROJECTS_DIR).rglob(f"{full_id}.jsonl"):
                jsonl_path = str(p)
                break

            if mode == "summary":
                try:
                    params = {"session_ids": full_id, "mode": "summary"}
                    if model:
                        params["model"] = model
                    data = await _api_get("/vscode/sessions/read", params)
                    body_text = data.get("content", "")
                    # Strip agent-mcp's own header block if present
                    if body_text.startswith("="):
                        stripped, in_hdr, eq_count = [], True, 0
                        for line in body_text.split("\n"):
                            if in_hdr and line.startswith("="):
                                eq_count += 1
                                if eq_count >= 2:
                                    in_hdr = False
                                continue
                            if not in_hdr:
                                stripped.append(line)
                        body_text = "\n".join(stripped)
                except Exception as e:
                    body_text = (
                        f"[agent-mcp {'unavailable' if _agent_mcp_unavailable(e) else f'error: {e}'}"
                        f" — returning full text]\n\n"
                        + (_read_session_text_local(jsonl_path) if jsonl_path else "(file not found)")
                    )
            else:
                body_text = (
                    _read_session_text_local(jsonl_path) if jsonl_path
                    else "(session file not found)"
                )

            parts.append(header + body_text)

        if not parts:
            return "ERROR: No matching sessions found."

        out = "\n\n".join(parts)
        if missing:
            out += f"\n\n[NOTE: {len(missing)} session ID(s) not found: {', '.join(missing)}]"
        return out

    # -----------------------------------------------------------------------
    # gdrive_list
    # -----------------------------------------------------------------------
    elif name == "gdrive_list":
        folder_id = args.get("folder_id", "")
        try:
            params = {}
            if folder_id:
                params["folder_id"] = folder_id
            data = await _api_get("/vscode/drive/list", params)
            return data.get("result", json.dumps(data))
        except Exception as e:
            if _agent_mcp_unavailable(e):
                drv = _get_drive()
                return await drv.run_drive_op("list", None, None, None, folder_id or None)
            raise

    # -----------------------------------------------------------------------
    # gdrive_read
    # -----------------------------------------------------------------------
    elif name == "gdrive_read":
        file_id = args.get("file_id", "").strip()
        if not file_id:
            return "ERROR: file_id is required"
        mode = args.get("mode", "full")
        try:
            body = {"file_id": file_id, "mode": mode}
            if args.get("extract_prompt"):
                body["extract_prompt"] = args["extract_prompt"]
            data = await _api_post("/vscode/drive/read", body)
            return data.get("content", json.dumps(data))
        except Exception as e:
            if _agent_mcp_unavailable(e):
                drv = _get_drive()
                content = await drv.run_drive_op("read", file_id, None, None, None)
                if mode != "full":
                    content = (
                        f"[agent-mcp offline — returning full content; {mode} mode unavailable]\n\n"
                        + content
                    )
                return content
            raise

    # -----------------------------------------------------------------------
    # gdrive_snippet_save
    # -----------------------------------------------------------------------
    elif name == "gdrive_snippet_save":
        content   = args.get("content", "")
        topic     = args.get("topic", "")
        file_id   = args.get("file_id", "")
        folder_id = args.get("folder_id", "")
        try:
            body = {"content": content, "topic": topic, "file_id": file_id}
            if folder_id:
                body["folder_id"] = folder_id
            data = await _api_post("/vscode/drive/snippet/save", body)
            action = data.get("action", "?")
            if action == "created":
                return f"Created Drive file: {data.get('filename')} — {data.get('result')}"
            elif action == "appended":
                return f"Appended to Drive file ID {data.get('file_id')} — {data.get('result')}"
            return json.dumps(data)
        except Exception as e:
            if _agent_mcp_unavailable(e):
                drv = _get_drive()
                if file_id:
                    return await drv.run_drive_op("append", file_id, None, content, folder_id or None)
                else:
                    fname = f"snippet-{topic}.txt" if topic else "snippet.txt"
                    return await drv.run_drive_op("create", None, fname, content, folder_id or None)
            raise

    # -----------------------------------------------------------------------
    # gdrive_sessions_export
    # -----------------------------------------------------------------------
    elif name == "gdrive_sessions_export":
        session_ids = args.get("session_ids", [])
        if not session_ids:
            return "ERROR: session_ids list is required"
        mode       = args.get("mode", "full")
        summarizer = args.get("summarizer", "claude")
        model      = args.get("model", "")
        folder_id  = args.get("folder_id", "")
        filename   = args.get("filename", "")

        from datetime import datetime, timezone
        import pathlib

        sessions = await asyncio.to_thread(_local_sessions_list, "", "")
        selected = [s for s in sessions if s["session_id"] in session_ids]
        missing  = len(session_ids) - len(selected)

        parts = []
        for s in sorted(selected, key=lambda x: x["first_timestamp"] or ""):
            header = (
                f"{'='*70}\n"
                f"Session: {s['title'][:80]}\n"
                f"Project: {s['project_path']}\n"
                f"Date:    {s.get('first_timestamp', 'unknown')}\n"
                f"ID:      {s['session_id']}\n"
                f"{'='*70}\n"
            )

            # Locate the JSONL file
            jsonl_path = None
            for p in pathlib.Path(CLAUDE_PROJECTS_DIR).rglob(f"{s['session_id']}.jsonl"):
                jsonl_path = str(p)
                break

            body_text = ""

            if mode == "summary" and summarizer == "agent":
                # Delegate summarization to agent-mcp — no VSCode tokens consumed
                try:
                    params = {"session_ids": s["session_id"], "mode": "summary"}
                    if model:
                        params["model"] = model
                    data = await _api_get("/vscode/sessions/read", params)
                    # agent-mcp returns the full assembled section (header + text)
                    # Use just the content portion so we apply our own header
                    body_text = data.get("content", "")
                    # Strip the agent-mcp header block if present (starts with ===)
                    if body_text.startswith("="):
                        lines = body_text.split("\n")
                        # Find end of header block (second === line)
                        in_header = True
                        stripped_lines = []
                        eq_count = 0
                        for line in lines:
                            if in_header and line.startswith("="):
                                eq_count += 1
                                if eq_count >= 2:
                                    in_header = False
                                continue
                            if not in_header:
                                stripped_lines.append(line)
                        body_text = "\n".join(stripped_lines)
                except Exception as e:
                    if _agent_mcp_unavailable(e):
                        body_text = (
                            f"[agent-mcp unavailable for summarization — using full text]\n\n"
                            + (_read_session_text_local(jsonl_path) if jsonl_path else "(file not found)")
                        )
                    else:
                        body_text = (
                            f"[agent-mcp summarization error: {e} — using full text]\n\n"
                            + (_read_session_text_local(jsonl_path) if jsonl_path else "(file not found)")
                        )
            else:
                # mode="full" OR mode="summary" with summarizer="claude":
                # Return raw text. For summarizer="claude", Claude Code sees this
                # text in-context and summarizes before the Drive write.
                if jsonl_path:
                    body_text = _read_session_text_local(jsonl_path)
                else:
                    body_text = "(session file not found)"

            parts.append(header + body_text)

        if not parts:
            return "ERROR: No matching sessions found locally."

        combined = "\n\n".join(parts)
        if missing:
            combined += f"\n\n[NOTE: {missing} session(s) not found]"

        fname = filename or f"claude-sessions-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.txt"
        drv = _get_drive()
        result = await drv.run_drive_op("create", None, fname, combined, folder_id or None)
        msg = f"Exported {len(parts)} session(s) to Drive: {fname} — {result}"
        if missing:
            msg += f"\n(Missing {missing} session(s))"
        return msg

    else:
        return f"ERROR: Unknown tool '{name}'"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
