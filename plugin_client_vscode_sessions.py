"""
VSCode Sessions Plugin for MCP Agent

Exposes HTTP API endpoints for reading local Claude Code JSONL session data.
Drive writes are NOT handled here — callers receive session content directly
and decide where to store it (e.g. the gdrive-mcp MCP server writes to Drive).

Endpoints:
  GET  /vscode/sessions/list     - list sessions with ID, title, timestamps, size
  GET  /vscode/sessions/read     - return assembled text for one or more sessions

LangChain tools (for LLM autonomous use):
  vscode_sessions_list  - list sessions with optional date/project filter
  vscode_sessions_read  - read one or more sessions (full or summarized)
"""

import os
import json
import glob
import asyncio
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse

from plugin_loader import BasePlugin
from config import log

_API_KEY: str = os.getenv("API_KEY", "")

CLAUDE_PROJECTS_DIR = os.path.expanduser("~/.claude/projects")


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _check_auth(request: Request) -> bool:
    if not _API_KEY:
        return True
    auth = request.headers.get("Authorization", "")
    return auth == f"Bearer {_API_KEY}"

def _auth_error() -> JSONResponse:
    return JSONResponse({"error": "Unauthorized"}, status_code=401)


# ---------------------------------------------------------------------------
# JSONL session parsing helpers
# ---------------------------------------------------------------------------

def _extract_session_title(jsonl_path: str) -> Optional[str]:
    """Return the first typed user text message as the session title."""
    try:
        with open(jsonl_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("type") == "user":
                    content = obj.get("message", {}).get("content", [])
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            text = c["text"].strip()
                            # Skip IDE-injected context blocks
                            if text and "<ide_" not in text:
                                return text[:200]
    except Exception:
        pass
    return None


def _extract_session_meta(jsonl_path: str) -> dict:
    """Return metadata for a single JSONL session file."""
    session_id = os.path.splitext(os.path.basename(jsonl_path))[0]
    cwd = None
    first_ts = None
    last_ts = None
    message_count = 0

    try:
        with open(jsonl_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
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
    except Exception:
        pass

    title = _extract_session_title(jsonl_path)

    # Decode project dir name (-home-markj-projects-foo → /home/markj/projects/foo)
    project_dir = os.path.basename(os.path.dirname(jsonl_path))
    project_path = project_dir.replace("-", "/").lstrip("/")
    if not project_path.startswith("/"):
        project_path = "/" + project_path

    try:
        file_size = os.path.getsize(jsonl_path)
    except Exception:
        file_size = 0

    return {
        "session_id": session_id,
        "jsonl_path": jsonl_path,
        "project_path": cwd or project_path,
        "title": title or "(no title)",
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "message_count": message_count,
        "file_bytes": file_size,
    }


def _list_all_sessions(date_filter: Optional[str] = None,
                       project_filter: Optional[str] = None) -> List[dict]:
    """
    Scan all JSONL session files across all Claude Code projects on this machine.

    Args:
        date_filter:    ISO date prefix e.g. "2026-02-24" — filters by first_timestamp
        project_filter: partial project path match e.g. "agent-mcp"
    """
    sessions = []
    pattern = os.path.join(CLAUDE_PROJECTS_DIR, "*", "*.jsonl")

    for jsonl_path in glob.glob(pattern):
        # Skip subagent files (nested under session-id subdirs)
        parts = jsonl_path.replace(CLAUDE_PROJECTS_DIR + "/", "").split("/")
        if len(parts) > 2:
            continue

        meta = _extract_session_meta(jsonl_path)

        if date_filter and meta.get("first_timestamp"):
            if not meta["first_timestamp"].startswith(date_filter):
                continue

        if project_filter and project_filter.lower() not in meta["project_path"].lower():
            continue

        sessions.append(meta)

    sessions.sort(key=lambda s: s.get("first_timestamp") or "")
    # Filter out orphaned snapshot-only files (no actual chat messages)
    return [s for s in sessions if s.get("message_count", 0) > 0]


def _extract_session_text(jsonl_path: str) -> str:
    """
    Extract human-readable user+assistant text from a JSONL session.
    Strips tool calls and tool results — keeps only conversational turns.
    """
    lines = []
    try:
        with open(jsonl_path) as f:
            for raw in f:
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue

                if obj.get("type") not in ("user", "assistant"):
                    continue

                role = obj.get("message", {}).get("role", "")
                content = obj.get("message", {}).get("content", [])

                if isinstance(content, str):
                    if content.strip():
                        lines.append(f"[{role.upper()}] {content.strip()}")
                    continue

                for c in content:
                    if not isinstance(c, dict):
                        continue
                    if c.get("type") == "text" and c.get("text", "").strip():
                        lines.append(f"[{role.upper()}] {c['text'].strip()}")
    except Exception as e:
        lines.append(f"[ERROR reading session: {e}]")

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Endpoint: GET /vscode/sessions/list
# ---------------------------------------------------------------------------

async def endpoint_sessions_list(request: Request) -> JSONResponse:
    """
    List all local Claude Code JSONL sessions on this machine.

    Query params:
        date    - ISO date prefix filter e.g. "2026-02-24"
        project - partial project path filter e.g. "agent-mcp"

    Returns session_id, title, first/last timestamps, message_count, file_bytes.
    """
    if not _check_auth(request):
        return _auth_error()

    date_filter    = request.query_params.get("date")
    project_filter = request.query_params.get("project")

    try:
        sessions = await asyncio.to_thread(_list_all_sessions, date_filter, project_filter)
        # Strip internal jsonl_path from response
        for s in sessions:
            s.pop("jsonl_path", None)
        return JSONResponse({"sessions": sessions, "count": len(sessions)})
    except Exception as e:
        log.exception("sessions_list failed")
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Endpoint: GET /vscode/sessions/read
# ---------------------------------------------------------------------------

async def endpoint_sessions_read(request: Request) -> JSONResponse:
    """
    Return assembled text content for one or more sessions.
    The caller is responsible for storing or forwarding the content.

    Query params:
        session_ids - comma-separated list of session UUIDs
        mode        - "full" (default) or "summary" (LLM summary via dispatch_llm)
    """
    if not _check_auth(request):
        return _auth_error()

    raw_ids = request.query_params.get("session_ids", "")
    session_ids = [s.strip() for s in raw_ids.split(",") if s.strip()]
    if not session_ids:
        return JSONResponse({"error": "session_ids required (comma-separated)"}, status_code=400)

    mode  = request.query_params.get("mode", "full").strip().lower()
    model = request.query_params.get("model", "").strip() or None

    try:
        all_sessions = await asyncio.to_thread(_list_all_sessions)
        session_map = {s["session_id"]: s for s in all_sessions}

        # Support 8-char prefix matching (IDs shown truncated in list output)
        def _resolve(sid: str):
            if sid in session_map:
                return sid
            matches = [k for k in session_map if k.startswith(sid)]
            return matches[0] if len(matches) == 1 else None

        sections = []
        missing = []

        for sid in session_ids:
            full_id = _resolve(sid)
            meta = session_map.get(full_id) if full_id else None
            if not meta:
                missing.append(sid)
                continue
            sid = full_id  # use canonical ID for logging

            header = (
                f"{'='*70}\n"
                f"Session: {meta['title']}\n"
                f"Project: {meta['project_path']}\n"
                f"Date:    {meta.get('first_timestamp', 'unknown')}\n"
                f"ID:      {meta['session_id']}\n"
                f"{'='*70}\n"
            )

            raw_text = await asyncio.to_thread(_extract_session_text, meta["jsonl_path"])

            if mode == "summary":
                try:
                    from agents import llm_call
                    instruction = (
                        "Summarize this Claude Code chat session. "
                        "Preserve specific commands, syntax, config values, and technical details verbatim. "
                        "Organize into clearly labeled topic sections."
                    )
                    prompt = f"{instruction}\n\n---\n{raw_text}\n---"
                    session_text = await llm_call(
                        model=model, prompt=prompt, mode="text",
                        sys_prompt="none", history="none",
                    ) or raw_text
                except Exception as e:
                    log.warning("llm summary failed for %s: %s — using full text", sid, e)
                    session_text = raw_text
            else:
                session_text = raw_text

            sections.append(header + session_text)

        if not sections:
            return JSONResponse({
                "error": "No valid sessions found",
                "missing_session_ids": missing,
            }, status_code=404)

        content = "\n\n".join(sections)
        if missing:
            content += f"\n\n[NOTE: {len(missing)} session(s) not found: {', '.join(missing)}]"

        return JSONResponse({
            "mode": mode,
            "sessions_returned": len(sections),
            "sessions_missing": len(missing),
            "content": content,
        })

    except Exception as e:
        log.exception("sessions_read failed")
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# !vscode command handler (for all agent-mcp chat clients: shell, Slack, API)
# ---------------------------------------------------------------------------

async def _cmd_vscode(args: str) -> str:
    """
    !vscode <subcommand> [options]

    Subcommands:
      list  [date=YYYY-MM-DD] [project=name]
            List local Claude Code sessions with IDs and titles.

      read  <id>[,<id>...] [mode=full|summary] [model=<key>]
            Pull session text into chat context.
            mode=summary uses agent-mcp LLM (model= selects which one).

    Examples:
      !vscode list
      !vscode list date=2026-02-24 project=agent-mcp
      !vscode read a1b2c3d4
      !vscode read a1b2c3d4,e5f6g7h8 mode=summary model=nuc11Local
    """
    parts = args.strip().split()
    if not parts:
        return (
            "Usage: !vscode <subcommand> [options]\n"
            "  !vscode list [date=YYYY-MM-DD] [project=name]\n"
            "  !vscode read <id>[,<id>...] [mode=full|summary] [model=<key>]"
        )

    sub = parts[0].lower()
    rest = parts[1:]

    # Parse key=value options
    opts = {}
    positional = []
    for token in rest:
        if "=" in token:
            k, _, v = token.partition("=")
            opts[k.lower()] = v
        else:
            positional.append(token)

    # ------------------------------------------------------------------
    # !vscode list
    # ------------------------------------------------------------------
    if sub == "list":
        date_filter    = opts.get("date", "")
        project_filter = opts.get("project", "")
        sessions = await asyncio.to_thread(_list_all_sessions, date_filter or None, project_filter or None)
        if not sessions:
            return "No sessions found."
        lines = [f"Found {len(sessions)} session(s):\n"]
        for s in sessions:
            ts = (s.get("first_timestamp") or "")[:10]
            proj = (s.get("project_path") or "").split("/")[-1]
            sid  = s["session_id"][:8]
            title = (s.get("title") or "(no title)")[:60]
            lines.append(f"  [{ts}] {proj:<28} ID: {sid}...  \"{title}\"")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # !vscode read
    # ------------------------------------------------------------------
    elif sub == "read":
        # IDs: first positional token (comma-separated) or id= option
        raw_ids = opts.get("id", "") or (positional[0] if positional else "")
        if not raw_ids:
            return "ERROR: session ID(s) required.\nUsage: !vscode read <id>[,<id>...] [mode=full|summary] [model=<key>]"

        session_ids = [s.strip() for s in raw_ids.split(",") if s.strip()]
        mode  = opts.get("mode", "full").lower()
        model = opts.get("model", "").strip() or None

        all_sessions = await asyncio.to_thread(_list_all_sessions)
        session_map = {s["session_id"]: s for s in all_sessions}

        # Support 8-char prefix matching (IDs shown truncated in !vscode list)
        def _resolve(sid: str):
            if sid in session_map:
                return sid
            matches = [k for k in session_map if k.startswith(sid)]
            return matches[0] if len(matches) == 1 else None

        sections = []
        missing  = []

        for sid in session_ids:
            full_id = _resolve(sid)
            if not full_id:
                missing.append(sid)
                continue

            meta = session_map[full_id]
            header = (
                f"{'='*70}\n"
                f"Session: {meta['title']}\n"
                f"Project: {meta['project_path']}\n"
                f"Date:    {meta.get('first_timestamp', 'unknown')}\n"
                f"ID:      {meta['session_id']}\n"
                f"{'='*70}\n"
            )

            raw_text = await asyncio.to_thread(_extract_session_text, meta["jsonl_path"])

            if mode == "summary":
                try:
                    from agents import llm_call
                    instruction = (
                        "Summarize this Claude Code chat session. "
                        "Preserve specific commands, syntax, config values, and technical details verbatim. "
                        "Organize into clearly labeled topic sections."
                    )
                    prompt = f"{instruction}\n\n---\n{raw_text}\n---"
                    session_text = await llm_call(
                        model=model, prompt=prompt, mode="text",
                        sys_prompt="none", history="none",
                    ) or raw_text
                except Exception as e:
                    log.warning("!vscode read summary failed for %s: %s — using full text", full_id, e)
                    session_text = raw_text
            else:
                session_text = raw_text

            sections.append(header + session_text)

        out = "\n\n".join(sections)
        if missing:
            out += f"\n\n[NOTE: {len(missing)} session ID(s) not found: {', '.join(missing)}]"
        return out if out.strip() else "No sessions found."

    else:
        return (
            f"Unknown subcommand '{sub}'.\n"
            "Usage: !vscode list | !vscode read"
        )


# ---------------------------------------------------------------------------
# LangChain tool arg schemas and executors
# ---------------------------------------------------------------------------

class _VscodeSessionsListArgs(BaseModel):
    date: Optional[str] = Field(default="", description="Filter by date prefix e.g. '2026-02-24'. Leave empty for all.")
    project: Optional[str] = Field(default="", description="Filter by partial project path e.g. 'agent-mcp'. Leave empty for all.")


class _VscodeSessionsReadArgs(BaseModel):
    session_ids: str = Field(description="Comma-separated session UUIDs or 8-char prefixes (from vscode_sessions_list).")
    mode: Optional[str] = Field(default="full", description="'full' — verbatim text. 'summary' — LLM-summarized (uses model param).")
    model: Optional[str] = Field(default="", description="agent-mcp model key for summarization e.g. 'nuc11Local', 'gemini25fl'. Empty = default model.")


async def _vscode_sessions_list_executor(date: str = "", project: str = "") -> str:
    sessions = await asyncio.to_thread(_list_all_sessions, date or None, project or None)
    if not sessions:
        return "No sessions found."
    lines = [f"Found {len(sessions)} session(s):\n"]
    for s in sessions:
        ts    = (s.get("first_timestamp") or "")[:10]
        proj  = (s.get("project_path") or "").split("/")[-1]
        sid   = s["session_id"][:8]
        title = (s.get("title") or "(no title)")[:60]
        lines.append(f"  [{ts}] {proj:<28} ID: {sid}...  \"{title}\"")
    return "\n".join(lines)


async def _vscode_sessions_read_executor(session_ids: str, mode: str = "full", model: str = "") -> str:
    ids = [s.strip() for s in session_ids.split(",") if s.strip()]
    if not ids:
        return "ERROR: session_ids required."

    all_sessions = await asyncio.to_thread(_list_all_sessions)
    session_map = {s["session_id"]: s for s in all_sessions}

    def _resolve(sid: str):
        if sid in session_map:
            return sid
        matches = [k for k in session_map if k.startswith(sid)]
        return matches[0] if len(matches) == 1 else None

    resolved_model = model.strip() or None
    sections = []
    missing  = []

    for sid in ids:
        full_id = _resolve(sid)
        if not full_id:
            missing.append(sid)
            continue
        meta = session_map[full_id]
        header = (
            f"{'='*70}\n"
            f"Session: {meta['title']}\n"
            f"Project: {meta['project_path']}\n"
            f"Date:    {meta.get('first_timestamp', 'unknown')}\n"
            f"ID:      {meta['session_id']}\n"
            f"{'='*70}\n"
        )
        raw_text = await asyncio.to_thread(_extract_session_text, meta["jsonl_path"])
        if mode == "summary":
            try:
                from agents import llm_call
                instruction = (
                    "Summarize this Claude Code chat session. "
                    "Preserve specific commands, syntax, config values, and technical details verbatim. "
                    "Organize into clearly labeled topic sections."
                )
                prompt = f"{instruction}\n\n---\n{raw_text}\n---"
                session_text = await llm_call(
                    model=resolved_model, prompt=prompt, mode="text",
                    sys_prompt="none", history="none",
                ) or raw_text
            except Exception as e:
                log.warning("vscode_sessions_read summary failed for %s: %s — using full text", full_id, e)
                session_text = raw_text
        else:
            session_text = raw_text
        sections.append(header + session_text)

    out = "\n\n".join(sections)
    if missing:
        out += f"\n\n[NOTE: {len(missing)} session ID(s) not found: {', '.join(missing)}]"
    return out if out.strip() else "No sessions found."


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------

class VscodeSessionsPlugin(BasePlugin):
    """Expose local Claude Code JSONL session data via HTTP API."""

    PLUGIN_NAME    = "plugin_client_vscode_sessions"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE    = "client_interface"
    DESCRIPTION    = "Local Claude Code session listing and content retrieval API"
    DEPENDENCIES   = []
    ENV_VARS       = []

    def __init__(self):
        self.enabled = False

    def init(self, config: dict) -> bool:
        if not os.path.isdir(CLAUDE_PROJECTS_DIR):
            log.warning(
                "plugin_client_vscode_sessions: ~/.claude/projects not found "
                "— Claude Code not installed on this machine?"
            )
            return False
        self.enabled = True
        log.info("plugin_client_vscode_sessions: initialized OK")
        return True

    def shutdown(self) -> None:
        self.enabled = False

    def get_config(self) -> dict:
        """No dedicated port — routes are added to the shared Starlette app."""
        return {"port": None, "name": "VSCode Sessions API (shared server)"}

    def get_routes(self) -> list:
        return [
            Route("/vscode/sessions/list", endpoint_sessions_list, methods=["GET"]),
            Route("/vscode/sessions/read", endpoint_sessions_read, methods=["GET"]),
        ]

    def get_commands(self) -> dict:
        return {"vscode": _cmd_vscode}

    def get_tools(self) -> Dict[str, Any]:
        return {
            "lc": [
                StructuredTool.from_function(
                    coroutine=_vscode_sessions_list_executor,
                    name="vscode_sessions_list",
                    description=(
                        "List local Claude Code chat sessions on this machine. "
                        "Returns session IDs (use 8-char prefix with vscode_sessions_read), "
                        "titles, dates, and project paths. "
                        "Filter by date= (e.g. '2026-02-24') or project= (e.g. 'agent-mcp')."
                    ),
                    args_schema=_VscodeSessionsListArgs,
                ),
                StructuredTool.from_function(
                    coroutine=_vscode_sessions_read_executor,
                    name="vscode_sessions_read",
                    description=(
                        "Read one or more local Claude Code sessions into context. "
                        "Pass comma-separated session IDs or 8-char prefixes from vscode_sessions_list. "
                        "mode='full' returns verbatim user+assistant text. "
                        "mode='summary' uses an LLM to summarize (specify model= key e.g. 'nuc11Local'). "
                        "Use this to bring past VSCode session context into the current conversation."
                    ),
                    args_schema=_VscodeSessionsReadArgs,
                ),
            ]
        }

    def get_help(self) -> str:
        return (
            "VSCode Sessions:\n"
            "  !vscode list [date=YYYY-MM-DD] [project=name]\n"
            "      List local Claude Code sessions with IDs and titles.\n"
            "  !vscode read <id>[,<id>...] [mode=full|summary] [model=<key>]\n"
            "      Pull session text into chat. mode=summary uses agent-mcp LLM.\n"
            "  GET /vscode/sessions/list?date=2026-02-24&project=agent-mcp\n"
            "      HTTP API: list sessions with metadata.\n"
            "  GET /vscode/sessions/read?session_ids=uuid1,uuid2&mode=full|summary[&model=key]\n"
            "      HTTP API: return assembled session text.\n"
        )
