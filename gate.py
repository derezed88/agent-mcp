import uuid
import asyncio
import os
import re
from state import push_gate, pending_gates, auto_aidb_state, tool_gate_state, sessions
from database import extract_table_names
from tools import get_gate_tools_by_type

# How long (seconds) to wait for an api- client to respond to a gate before auto-rejecting.
# AgentClient with an approval directive responds within milliseconds, so 2s is ample.
_API_GATE_TIMEOUT: float = float(os.getenv("API_GATE_TIMEOUT", "2.0"))

def is_sql_write_operation(sql: str) -> bool:
    """Check if SQL is a write operation (INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, TRUNCATE)"""
    sql_upper = sql.strip().upper()
    write_keywords = ["INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "TRUNCATE", "REPLACE"]
    return any(sql_upper.startswith(kw) for kw in write_keywords)

def is_drive_write_operation(operation: str) -> bool:
    """Check if Google Drive operation is a write operation (create, append, delete)"""
    return operation.lower() in ["create", "append", "delete"]

async def check_human_gate(client_id: str, tool_name: str, tool_args: dict) -> bool:
    """
    Check if a tool call requires human approval.

    For llama proxy clients (client_id starts with "llama-") and Slack clients (starts with "slack-"),
    gates are auto-rejected since there's no interactive approval mechanism.
    The LLM receives instructive feedback.

    For shell.py clients, gates trigger interactive approval prompts.
    """
    # Check if this is a non-interactive client (no gate support)
    is_llama_proxy = client_id.startswith("llama-")
    is_slack_client = client_id.startswith("slack-")
    is_api_client = client_id.startswith("api-")
    is_non_interactive = is_llama_proxy or is_slack_client

    # @<model> temp switch: admin explicitly delegated this turn — auto-allow all gates
    if sessions.get(client_id, {}).get("_temp_model_active", False):
        return True

    def should_auto_reject(needs_gate: bool) -> bool:
        """Auto-reject gates for non-interactive clients (llama proxy, slack)"""
        return needs_gate and is_non_interactive

    async def do_api_gate(gate_data: dict) -> bool:
        """
        For api- clients: push gate to SSE queue then wait _API_GATE_TIMEOUT seconds
        for the AgentClient to respond via POST /api/v1/gate/{gate_id}.
        If no response arrives in time, auto-reject.
        AgentClient with auto_approve_gates responds within milliseconds when configured.
        """
        gate_id = gate_data["gate_id"]
        pending_gates[gate_id] = {"event": asyncio.Event(), "decision": None}
        await push_gate(client_id, gate_data)
        try:
            await asyncio.wait_for(pending_gates[gate_id]["event"].wait(), timeout=_API_GATE_TIMEOUT)
            decision = pending_gates[gate_id].pop("decision", "reject")
        except asyncio.TimeoutError:
            decision = "reject"
        pending_gates.pop(gate_id, None)
        return decision == "allow"

    if tool_name == "db_query":
        sql = tool_args.get("sql", "")
        tables = extract_table_names(sql)
        is_write = is_sql_write_operation(sql)
        permission_type = "write" if is_write else "read"

        # Get default permission setting (wildcard "*")
        default_perms = auto_aidb_state.get("*", {})
        default_allowed = default_perms.get(permission_type, False)

        if not tables:
            # No specific table, check __meta__ permissions (or default)
            table_perms = auto_aidb_state.get("__meta__", {})
            # Use table-specific setting, fall back to default
            needs_gate = not table_perms.get(permission_type, default_allowed)
            display_tables = ["__meta__"]
        else:
            # Check if any table requires gating for this permission type
            needs_gate = False
            for t in tables:
                table_perms = auto_aidb_state.get(t, {})
                # Use table-specific setting, fall back to default
                if not table_perms.get(permission_type, default_allowed):
                    needs_gate = True
                    break
            display_tables = tables

        if not needs_gate:
            return True

        # Auto-reject for llama proxy clients (no interactive approval available)
        if should_auto_reject(needs_gate):
            return False

        gate_data = {
            "gate_id": str(uuid.uuid4()),
            "tool_name": tool_name,
            "tool_args": {**tool_args, "operation_type": permission_type},
            "tables": display_tables,
        }

        if is_api_client:
            return await do_api_gate(gate_data)

        gate_id = gate_data["gate_id"]
        pending_gates[gate_id] = {"event": asyncio.Event(), "decision": None}
        await push_gate(client_id, gate_data)
        await pending_gates[gate_id]["event"].wait()
        decision = pending_gates[gate_id].pop("decision", "reject")
        pending_gates.pop(gate_id, None)
        return decision == "allow"

    if tool_name in ("get_system_info", "llm_list", "help", "sysprompt_list", "sysprompt_read",
                     "llm_call", "llm_timeout", "stream", "tool_preview_length"):
        return True

    # at_llm: always treated as write (worst-case) — gated via at_llm_gate_write
    if tool_name == "at_llm":
        default_perms = tool_gate_state.get("*", {})
        tool_perms = tool_gate_state.get("at_llm", {})
        if tool_perms.get("write", default_perms.get("write", False)):
            return True
        if is_non_interactive:
            return False
        gate_data = {
            "gate_id": str(uuid.uuid4()),
            "tool_name": tool_name,
            "tool_args": {**tool_args, "operation_type": "write"},
            "tables": [],
        }
        if is_api_client:
            return await do_api_gate(gate_data)
        gate_id = gate_data["gate_id"]
        pending_gates[gate_id] = {"event": asyncio.Event(), "decision": None}
        await push_gate(client_id, gate_data)
        await pending_gates[gate_id]["event"].wait()
        decision = pending_gates[gate_id].pop("decision", "reject")
        pending_gates.pop(gate_id, None)
        return decision == "allow"

    # gate_list: read-only, gatable via gate_list_gate_read
    if tool_name == "gate_list":
        default_perms = tool_gate_state.get("*", {})
        tool_perms = tool_gate_state.get("gate_list", {})
        if tool_perms.get("read", default_perms.get("read", False)):
            return True
        if is_non_interactive:
            return False
        gate_data = {
            "gate_id": str(uuid.uuid4()),
            "tool_name": tool_name,
            "tool_args": {"operation_type": "read"},
            "tables": [],
        }
        if is_api_client:
            return await do_api_gate(gate_data)
        gate_id = gate_data["gate_id"]
        pending_gates[gate_id] = {"event": asyncio.Event(), "decision": None}
        await push_gate(client_id, gate_data)
        await pending_gates[gate_id]["event"].wait()
        decision = pending_gates[gate_id].pop("decision", "reject")
        pending_gates.pop(gate_id, None)
        return decision == "allow"

    # limit_depth_list: read-only, gatable via limit_depth_list_gate_read
    if tool_name == "limit_depth_list":
        default_perms = tool_gate_state.get("*", {})
        tool_perms = tool_gate_state.get("limit_depth_list", {})
        if tool_perms.get("read", default_perms.get("read", True)):
            return True
        if is_non_interactive:
            return False
        gate_data = {
            "gate_id": str(uuid.uuid4()),
            "tool_name": tool_name,
            "tool_args": {"operation_type": "read"},
            "tables": [],
        }
        if is_api_client:
            return await do_api_gate(gate_data)
        gate_id = gate_data["gate_id"]
        pending_gates[gate_id] = {"event": asyncio.Event(), "decision": None}
        await push_gate(client_id, gate_data)
        await pending_gates[gate_id]["event"].wait()
        decision = pending_gates[gate_id].pop("decision", "reject")
        pending_gates.pop(gate_id, None)
        return decision == "allow"

    # limit_depth_set: write operation, gatable via limit_depth_set_gate_write
    if tool_name == "limit_depth_set":
        default_perms = tool_gate_state.get("*", {})
        tool_perms = tool_gate_state.get("limit_depth_set", {})
        if tool_perms.get("write", default_perms.get("write", False)):
            return True
        if is_non_interactive:
            return False
        gate_data = {
            "gate_id": str(uuid.uuid4()),
            "tool_name": tool_name,
            "tool_args": {**tool_args, "operation_type": "write"},
            "tables": [],
        }
        if is_api_client:
            return await do_api_gate(gate_data)
        gate_id = gate_data["gate_id"]
        pending_gates[gate_id] = {"event": asyncio.Event(), "decision": None}
        await push_gate(client_id, gate_data)
        await pending_gates[gate_id]["event"].wait()
        decision = pending_gates[gate_id].pop("decision", "reject")
        pending_gates.pop(gate_id, None)
        return decision == "allow"

    # limit_rate_list: read-only, gatable via limit_rate_list_gate_read (default: auto-allowed)
    if tool_name == "limit_rate_list":
        default_perms = tool_gate_state.get("*", {})
        tool_perms = tool_gate_state.get("limit_rate_list", {})
        if tool_perms.get("read", default_perms.get("read", True)):
            return True
        if is_non_interactive:
            return False
        gate_data = {
            "gate_id": str(uuid.uuid4()),
            "tool_name": tool_name,
            "tool_args": {"operation_type": "read"},
            "tables": [],
        }
        if is_api_client:
            return await do_api_gate(gate_data)
        gate_id = gate_data["gate_id"]
        pending_gates[gate_id] = {"event": asyncio.Event(), "decision": None}
        await push_gate(client_id, gate_data)
        await pending_gates[gate_id]["event"].wait()
        decision = pending_gates[gate_id].pop("decision", "reject")
        pending_gates.pop(gate_id, None)
        return decision == "allow"

    # limit_rate_set: write operation, gatable via limit_rate_set_gate_write (default: gated)
    if tool_name == "limit_rate_set":
        default_perms = tool_gate_state.get("*", {})
        tool_perms = tool_gate_state.get("limit_rate_set", {})
        if tool_perms.get("write", default_perms.get("write", False)):
            return True
        if is_non_interactive:
            return False
        gate_data = {
            "gate_id": str(uuid.uuid4()),
            "tool_name": tool_name,
            "tool_args": {**tool_args, "operation_type": "write"},
            "tables": [],
        }
        if is_api_client:
            return await do_api_gate(gate_data)
        gate_id = gate_data["gate_id"]
        pending_gates[gate_id] = {"event": asyncio.Event(), "decision": None}
        await push_gate(client_id, gate_data)
        await pending_gates[gate_id]["event"].wait()
        decision = pending_gates[gate_id].pop("decision", "reject")
        pending_gates.pop(gate_id, None)
        return decision == "allow"

    # limit_max_iteration_list: read-only, gatable via limit_max_iteration_list_gate_read (default: auto-allowed)
    if tool_name == "limit_max_iteration_list":
        default_perms = tool_gate_state.get("*", {})
        tool_perms = tool_gate_state.get("limit_max_iteration_list", {})
        if tool_perms.get("read", default_perms.get("read", True)):
            return True
        if is_non_interactive:
            return False
        gate_data = {
            "gate_id": str(uuid.uuid4()),
            "tool_name": tool_name,
            "tool_args": {**tool_args, "operation_type": "read"},
            "tables": [],
        }
        if is_api_client:
            return await do_api_gate(gate_data)
        gate_id = gate_data["gate_id"]
        pending_gates[gate_id] = {"event": asyncio.Event(), "decision": None}
        await push_gate(client_id, gate_data)
        await pending_gates[gate_id]["event"].wait()
        decision = pending_gates[gate_id].pop("decision", "reject")
        pending_gates.pop(gate_id, None)
        return decision == "allow"

    # limit_max_iteration_set: write operation, gatable via limit_max_iteration_set_gate_write (default: gated)
    if tool_name == "limit_max_iteration_set":
        default_perms = tool_gate_state.get("*", {})
        tool_perms = tool_gate_state.get("limit_max_iteration_set", {})
        if tool_perms.get("write", default_perms.get("write", False)):
            return True
        if is_non_interactive:
            return False
        gate_data = {
            "gate_id": str(uuid.uuid4()),
            "tool_name": tool_name,
            "tool_args": {**tool_args, "operation_type": "write"},
            "tables": [],
        }
        if is_api_client:
            return await do_api_gate(gate_data)
        gate_id = gate_data["gate_id"]
        pending_gates[gate_id] = {"event": asyncio.Event(), "decision": None}
        await push_gate(client_id, gate_data)
        await pending_gates[gate_id]["event"].wait()
        decision = pending_gates[gate_id].pop("decision", "reject")
        pending_gates.pop(gate_id, None)
        return decision == "allow"

    # sysprompt write operations — gated
    if tool_name in ("sysprompt_write", "sysprompt_delete", "sysprompt_copy_dir", "sysprompt_set_dir"):
        default_perms = tool_gate_state.get("*", {})
        tool_perms = tool_gate_state.get("sysprompt_write", {})
        if tool_perms.get("write", default_perms.get("write", False)):
            return True
        if is_non_interactive:
            return False
        gate_data = {
            "gate_id": str(uuid.uuid4()),
            "tool_name": tool_name,
            "tool_args": {**tool_args, "operation_type": "write"},
            "tables": [],
        }
        if is_api_client:
            return await do_api_gate(gate_data)
        gate_id = gate_data["gate_id"]
        pending_gates[gate_id] = {"event": asyncio.Event(), "decision": None}
        await push_gate(client_id, gate_data)
        await pending_gates[gate_id]["event"].wait()
        decision = pending_gates[gate_id].pop("decision", "reject")
        pending_gates.pop(gate_id, None)
        return decision == "allow"

    # session: read gate for "list", write gate for "delete"
    if tool_name == "session":
        action = tool_args.get("action", "list")
        perm_type = "write" if action == "delete" else "read"
        default_perms = tool_gate_state.get("*", {})
        tool_perms = tool_gate_state.get("session", {})
        if tool_perms.get(perm_type, default_perms.get(perm_type, False)):
            return True
        if is_non_interactive:
            return False
        gate_data = {
            "gate_id": str(uuid.uuid4()),
            "tool_name": tool_name,
            "tool_args": {**tool_args, "operation_type": perm_type},
            "tables": [],
        }
        if is_api_client:
            return await do_api_gate(gate_data)
        gate_id = gate_data["gate_id"]
        pending_gates[gate_id] = {"event": asyncio.Event(), "decision": None}
        await push_gate(client_id, gate_data)
        await pending_gates[gate_id]["event"].wait()
        decision = pending_gates[gate_id].pop("decision", "reject")
        pending_gates.pop(gate_id, None)
        return decision == "allow"

    # model: list is auto-allowed; set requires write gate
    if tool_name == "model":
        action = tool_args.get("action", "list")
        if action == "list":
            return True
        default_perms = tool_gate_state.get("*", {})
        tool_perms = tool_gate_state.get("model", {})
        if tool_perms.get("write", default_perms.get("write", False)):
            return True
        if is_non_interactive:
            return False
        gate_data = {
            "gate_id": str(uuid.uuid4()),
            "tool_name": tool_name,
            "tool_args": {**tool_args, "operation_type": "write"},
            "tables": [],
        }
        if is_api_client:
            return await do_api_gate(gate_data)
        gate_id = gate_data["gate_id"]
        pending_gates[gate_id] = {"event": asyncio.Event(), "decision": None}
        await push_gate(client_id, gate_data)
        await pending_gates[gate_id]["event"].wait()
        decision = pending_gates[gate_id].pop("decision", "reject")
        pending_gates.pop(gate_id, None)
        return decision == "allow"

    # reset: write gate
    if tool_name == "reset":
        default_perms = tool_gate_state.get("*", {})
        tool_perms = tool_gate_state.get("reset", {})
        if tool_perms.get("write", default_perms.get("write", False)):
            return True
        if is_non_interactive:
            return False
        gate_data = {
            "gate_id": str(uuid.uuid4()),
            "tool_name": tool_name,
            "tool_args": {**tool_args, "operation_type": "write"},
            "tables": [],
        }
        if is_api_client:
            return await do_api_gate(gate_data)
        gate_id = gate_data["gate_id"]
        pending_gates[gate_id] = {"event": asyncio.Event(), "decision": None}
        await push_gate(client_id, gate_data)
        await pending_gates[gate_id]["event"].wait()
        decision = pending_gates[gate_id].pop("decision", "reject")
        pending_gates.pop(gate_id, None)
        return decision == "allow"

    # tmux: all registered tmux tools — checked dynamically from plugin registry.
    # Group gate "tmux" covers all tools of this type; per-tool gates override.
    # Tools with only ["write"] ops use write gate; tools with ["read","write"] ops
    # infer perm_type from whether any mutation args are present.
    if tool_name in get_gate_tools_by_type("tmux"):
        from tools import get_all_gate_tools as _all_gt
        tool_meta = _all_gt().get(tool_name, {})
        ops = tool_meta.get("operations", ["write"])
        if "read" in ops and "write" in ops:
            # Mixed read/write: infer from presence of non-None args (tmux_call_limit pattern)
            has_write_args = bool(tool_args and any(v is not None for v in tool_args.values()))
            perm_type = "write" if has_write_args else "read"
        else:
            perm_type = ops[0] if ops else "write"

        default_perms = tool_gate_state.get("*", {})
        group_perms  = tool_gate_state.get("tmux", {})
        per_tool     = tool_gate_state.get(tool_name, {})
        # per-tool overrides group; group overrides wildcard
        allowed = per_tool.get(perm_type,
                      group_perms.get(perm_type,
                          default_perms.get(perm_type, False)))
        if allowed:
            return True
        if is_non_interactive:
            return False
        gate_data = {
            "gate_id": str(uuid.uuid4()),
            "tool_name": tool_name,
            "tool_args": {**tool_args, "operation_type": perm_type},
            "tables": [],
        }
        if is_api_client:
            return await do_api_gate(gate_data)
        gate_id = gate_data["gate_id"]
        pending_gates[gate_id] = {"event": asyncio.Event(), "decision": None}
        await push_gate(client_id, gate_data)
        await pending_gates[gate_id]["event"].wait()
        decision = pending_gates[gate_id].pop("decision", "reject")
        pending_gates.pop(gate_id, None)
        return decision == "allow"

    # All registered extract tools are read-only — checked dynamically from plugin registry
    if tool_name in get_gate_tools_by_type("extract"):
        default_perms = tool_gate_state.get("*", {})
        tool_perms = tool_gate_state.get(tool_name, {})
        # Use tool-specific setting, fall back to default
        if tool_perms.get("read", default_perms.get("read", False)):
            return True

        # Auto-reject for non-interactive clients (llama proxy, slack)
        if is_non_interactive:
            return False

        url = tool_args.get("url", "")
        query = tool_args.get("query", "")
        gate_data = {
            "gate_id": str(uuid.uuid4()),
            "tool_name": tool_name,
            "tool_args": {"url": url, "query": query, "operation_type": "read"},
            "tables": [],
        }

        if is_api_client:
            return await do_api_gate(gate_data)

        gate_id = gate_data["gate_id"]
        pending_gates[gate_id] = {"event": asyncio.Event(), "decision": None}
        await push_gate(client_id, gate_data)
        await pending_gates[gate_id]["event"].wait()
        decision = pending_gates[gate_id].pop("decision", "reject")
        pending_gates.pop(gate_id, None)
        return decision == "allow"

    # All registered search tools are read-only — checked dynamically from plugin registry
    if tool_name in get_gate_tools_by_type("search"):
        default_perms = tool_gate_state.get("*", {})
        tool_perms = tool_gate_state.get(tool_name, {})
        # Use tool-specific setting, fall back to default
        if tool_perms.get("read", default_perms.get("read", False)):
            return True

        # Auto-reject for non-interactive clients (llama proxy, slack)
        if is_non_interactive:
            return False

        query = tool_args.get("query", "")
        gate_data = {
            "gate_id": str(uuid.uuid4()),
            "tool_name": tool_name,
            "tool_args": {"query": query, "operation_type": "read"},
            "tables": [],
        }

        if is_api_client:
            return await do_api_gate(gate_data)

        gate_id = gate_data["gate_id"]
        pending_gates[gate_id] = {"event": asyncio.Event(), "decision": None}
        await push_gate(client_id, gate_data)
        await pending_gates[gate_id]["event"].wait()
        decision = pending_gates[gate_id].pop("decision", "reject")
        return decision == "allow"

    if tool_name == "google_drive":
        op = tool_args.get("operation", "")
        is_write = is_drive_write_operation(op)
        permission_type = "write" if is_write else "read"

        default_perms = tool_gate_state.get("*", {})
        tool_perms = tool_gate_state.get("google_drive", {})
        # Use tool-specific setting, fall back to default
        if tool_perms.get(permission_type, default_perms.get(permission_type, False)):
            return True

        # Auto-reject for non-interactive clients (llama proxy, slack)
        if is_non_interactive:
            return False

        fname = tool_args.get("file_name") or tool_args.get("file_id") or ""
        gate_data = {
            "gate_id": str(uuid.uuid4()),
            "tool_name": tool_name,
            "tool_args": {"operation": op, "details": fname, "operation_type": permission_type},
            "tables": [],
        }

        if is_api_client:
            return await do_api_gate(gate_data)

        gate_id = gate_data["gate_id"]
        pending_gates[gate_id] = {"event": asyncio.Event(), "decision": None}
        await push_gate(client_id, gate_data)
        await pending_gates[gate_id]["event"].wait()
        decision = pending_gates[gate_id].pop("decision", "reject")
        return decision == "allow"

    # agent_call: write gate (sends message to remote agent — outbound action)
    if tool_name == "agent_call":
        default_perms = tool_gate_state.get("*", {})
        tool_perms = tool_gate_state.get("agent_call", {})
        if tool_perms.get("write", default_perms.get("write", False)):
            return True
        if is_non_interactive:
            return False
        gate_data = {
            "gate_id": str(uuid.uuid4()),
            "tool_name": tool_name,
            "tool_args": {**tool_args, "operation_type": "write"},
            "tables": [],
        }
        if is_api_client:
            return await do_api_gate(gate_data)
        gate_id = gate_data["gate_id"]
        pending_gates[gate_id] = {"event": asyncio.Event(), "decision": None}
        await push_gate(client_id, gate_data)
        await pending_gates[gate_id]["event"].wait()
        decision = pending_gates[gate_id].pop("decision", "reject")
        pending_gates.pop(gate_id, None)
        return decision == "allow"

    # sleep: write gate (sleeping affects execution flow)
    if tool_name == "sleep":
        default_perms = tool_gate_state.get("*", {})
        tool_perms = tool_gate_state.get("sleep", {})
        if tool_perms.get("read", default_perms.get("read", False)):
            return True
        if is_non_interactive:
            return False
        seconds = tool_args.get("seconds", "?")
        gate_data = {
            "gate_id": str(uuid.uuid4()),
            "tool_name": tool_name,
            "tool_args": {"seconds": seconds, "operation_type": "read"},
            "tables": [],
        }
        if is_api_client:
            return await do_api_gate(gate_data)
        gate_id = gate_data["gate_id"]
        pending_gates[gate_id] = {"event": asyncio.Event(), "decision": None}
        await push_gate(client_id, gate_data)
        await pending_gates[gate_id]["event"].wait()
        decision = pending_gates[gate_id].pop("decision", "reject")
        pending_gates.pop(gate_id, None)
        return decision == "allow"

    return True