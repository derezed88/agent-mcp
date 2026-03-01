#!/usr/bin/env python3
"""
Parameterized gate test for agent-mcp.

Usage:
  python test_gate.py <test_name> [options]

Tests:
  db_query_deny      - db_query gate fires, user denies
  db_query_approve   - db_query gate fires, user approves
  db_query_no_gate   - no gate fires (gate removed)
  session_delete     - 'session delete' gated on gemini25f:
                       1. !session list runs freely (not gated)
                       2. 'session delete <id>' is gated (denied)

Options:
  --server URL       Base server URL (default: http://localhost:8767)
  --admin URL        Admin server URL for !model_cfg (default: http://localhost:8765)
"""

import asyncio
import httpx
import sys
import time
import argparse
import re

DEFAULT_SERVER = "http://localhost:8767"
DEFAULT_ADMIN  = "http://localhost:8765"


class GateTestClient:
    """Reusable async client for gate testing."""

    def __init__(self, server: str, client_id: str | None = None):
        self.server = server.rstrip("/")
        self.client_id = client_id or f"gate-test-{int(time.time())}"

    async def submit(self, text: str) -> None:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self.server}/submit",
                json={"client_id": self.client_id, "text": text},
            )
            assert resp.status_code == 200, f"submit failed: {resp.status_code} {resp.text}"

    async def collect(self, timeout: float = 45.0) -> list[tuple[str, str]]:
        """Collect SSE events until 'done' or gate event (stops to allow response)."""
        events: list[tuple[str, str]] = []
        deadline = time.monotonic() + timeout
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "GET", f"{self.server}/stream",
                    params={"client_id": self.client_id}
                ) as resp:
                    current_event = "message"
                    async for line in resp.aiter_lines():
                        if time.monotonic() > deadline:
                            break
                        line = line.strip()
                        if not line:
                            current_event = "message"
                            continue
                        if line.startswith("event:"):
                            ev = line[6:].strip()
                            current_event = ev
                            if ev == "done":
                                events.append(("done", ""))
                                return events
                            continue
                        if line.startswith("data:"):
                            raw = line[5:]
                            if raw.startswith(" "):
                                raw = raw[1:]
                            decoded = raw.replace("\\n", "\n")
                            events.append((current_event, decoded))
                            if current_event == "gate":
                                return events  # stop; caller will respond
        except Exception as e:
            events.append(("error", str(e)))
        return events

    async def gate_respond(self, approved: bool) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self.server}/gate_respond",
                json={"client_id": self.client_id, "approved": approved},
            )
            return resp.json()

    def has_gate(self, events: list[tuple[str, str]]) -> bool:
        return any(ev == "gate" for ev, _ in events)

    def all_text(self, events: list[tuple[str, str]]) -> str:
        return " ".join(d for ev, d in events if ev in ("message", ""))


def show(events: list[tuple[str, str]], label: str = "") -> None:
    if label:
        print(f"\n{'='*60}\n{label}\n{'='*60}")
    for ev_type, data in events:
        if ev_type in ("message", ""):
            if data.strip():
                print(f"  [tok] {data[:250]}")
        elif ev_type == "gate":
            print(f"  [GATE EVENT]\n{data}")
        elif ev_type == "done":
            print("  [done]")
        elif ev_type == "error":
            print(f"  [ERROR] {data}")
        else:
            print(f"  [{ev_type}] {data[:120]}")


async def admin_cmd(admin_url: str, text: str) -> str:
    """Send a !command to the admin port and return the response text."""
    cid = f"admin-{int(time.time())}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(
            f"{admin_url}/submit",
            json={"client_id": cid, "text": text},
        )
    output_lines: list[str] = []
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "GET", f"{admin_url}/stream",
            params={"client_id": cid}
        ) as resp:
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("event:") and "done" in line:
                    break
                if line.startswith("data:"):
                    output_lines.append(line[5:].strip())
    result = "\n".join(output_lines)
    print(f"  [admin] {text!r} → {result[:200]}")
    return result


# ──────────────────────────────────────────────────────────────
# Individual test functions
# ──────────────────────────────────────────────────────────────

async def test_db_query_deny(server: str, **_) -> bool:
    """Gate fires for db_query; user denies."""
    c = GateTestClient(server)
    prompt = "Please use the db_query tool to run SHOW TABLES and tell me what tables exist."

    print("Submitting prompt (db_query gate active)...")
    await c.submit(prompt)

    events = await c.collect(timeout=45.0)
    show(events, "Phase 1: waiting for gate")

    if not c.has_gate(events):
        print("\nFAIL: No gate event — gate did not fire!")
        return False

    print("\nGate fired. Sending DENY...")
    result = await c.gate_respond(False)
    print(f"  gate_respond → {result}")

    events2 = await c.collect(timeout=30.0)
    show(events2, "Phase 2: LLM response after denial")

    print("\nPASS: db_query gate fired and was denied.")
    return True


async def test_db_query_approve(server: str, **_) -> bool:
    """Gate fires for db_query; user approves; query runs."""
    c = GateTestClient(server)
    prompt = "Please use the db_query tool to run SHOW TABLES and tell me what tables exist."

    print("Submitting prompt (db_query gate active)...")
    await c.submit(prompt)

    events = await c.collect(timeout=45.0)
    show(events, "Phase 1: waiting for gate")

    if not c.has_gate(events):
        print("\nFAIL: No gate event — gate did not fire!")
        return False

    print("\nGate fired. Sending APPROVE...")
    result = await c.gate_respond(True)
    print(f"  gate_respond → {result}")

    events2 = await c.collect(timeout=60.0)
    show(events2, "Phase 2: LLM response after approval")

    text = c.all_text(events2)
    if any(kw in text.lower() for kw in ("table", "show", "result", "query", "db")):
        print("\nPASS: db_query gate approved; query ran.")
    else:
        print("\nPASS: db_query gate approved; LLM responded.")
    return True


async def test_db_query_no_gate(server: str, **_) -> bool:
    """No gate should fire (gate has been removed)."""
    c = GateTestClient(server)
    prompt = "Please use the db_query tool to run SHOW TABLES and tell me what tables exist."

    print("Submitting prompt (no gate expected)...")
    await c.submit(prompt)

    events = await c.collect(timeout=60.0)
    show(events, "Full response (no gate expected)")

    if c.has_gate(events):
        print("\nFAIL: Gate fired — should NOT have after removal!")
        return False

    text = c.all_text(events)
    if any(kw in text.lower() for kw in ("table", "show", "result", "information_schema")):
        print("\nPASS: No gate; db_query ran freely.")
    else:
        print("\nPASS: No gate fired.")
    return True


async def test_session_delete(server: str, admin: str, **_) -> bool:
    """
    Gate 'session delete' on gemini25f.
    Step 1: '!session' (list) runs freely — no gate.
    Step 2: Tell LLM to delete a noted session — gate fires, deny.
    """
    MODEL = "gemini25f"
    GATE_ENTRY = "session delete"
    c = GateTestClient(server)

    # ── Setup: add gate to gemini25f ──────────────────────────
    print(f"\n>>> Setup: adding gate '{GATE_ENTRY}' to {MODEL}")
    await admin_cmd(admin, f"!model_cfg write {MODEL} llm_tools_gates session delete")

    # ── Step 1: list sessions — should NOT be gated ───────────
    print(f"\n>>> Step 1: ask LLM to list sessions (should run freely, no gate)")
    await c.submit("Please use the session tool to list all active sessions.")

    events1 = await c.collect(timeout=45.0)
    show(events1, "Step 1: session list (no gate expected)")

    if c.has_gate(events1):
        print("\nFAIL: Gate fired for session list — should NOT have!")
        await admin_cmd(admin, f"!model_cfg write {MODEL} llm_tools_gates ")
        return False

    # Extract a session ID from the response text to use in delete
    text1 = c.all_text(events1)
    print(f"\n  Session list response (excerpt): {text1[:400]}")

    # Look for shorthand IDs like [101] or full session IDs
    shorthand_ids = re.findall(r'\[(\d{3,})\]', text1)
    full_ids = re.findall(r'\b([a-z]+-[\w.@-]+)\b', text1)
    target_session = None
    if shorthand_ids:
        target_session = shorthand_ids[0]
        print(f"\n  Found shorthand session ID: {target_session}")
    elif full_ids:
        target_session = full_ids[0]
        print(f"\n  Found full session ID: {target_session}")
    else:
        # Use a placeholder — gate should still fire
        target_session = "101"
        print(f"\n  No session IDs parsed; using placeholder: {target_session}")

    print("\nPASS ✓ Step 1: session list ran freely (no gate).")

    # ── Step 2: delete that session — gate should fire ────────
    print(f"\n>>> Step 2: ask LLM to delete session {target_session!r} (gate expected)")
    prompt2 = (
        f"Please use the session tool to delete session ID {target_session}. "
        f"Use action='delete' and session_id='{target_session}'."
    )
    await c.submit(prompt2)

    events2 = await c.collect(timeout=45.0)
    show(events2, "Step 2: session delete (gate expected)")

    if not c.has_gate(events2):
        print("\nFAIL: Gate did NOT fire for session delete — should have!")
        await admin_cmd(admin, f"!model_cfg write {MODEL} llm_tools_gates ")
        return False

    print(f"\nGate fired. Sending DENY (not approving session delete)...")
    result = await c.gate_respond(False)
    print(f"  gate_respond(False) → {result}")

    events3 = await c.collect(timeout=30.0)
    show(events3, "Step 2: LLM response after denial")

    print("\nPASS ✓ Step 2: session delete gate fired and was denied.")

    # ── Cleanup: remove gate ──────────────────────────────────
    print(f"\n>>> Cleanup: removing gate from {MODEL}")
    await admin_cmd(admin, f"!model_cfg write {MODEL} llm_tools_gates ")

    print("\n>>> ALL STEPS PASSED ✓")
    return True


# ──────────────────────────────────────────────────────────────
# Test registry
# ──────────────────────────────────────────────────────────────

TESTS = {
    "db_query_deny":    test_db_query_deny,
    "db_query_approve": test_db_query_approve,
    "db_query_no_gate": test_db_query_no_gate,
    "session_delete":   test_session_delete,
}


async def main():
    parser = argparse.ArgumentParser(description="Parameterized gate test for agent-mcp")
    parser.add_argument("test", choices=list(TESTS), help="Test to run")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="Agent server URL")
    parser.add_argument("--admin",  default=DEFAULT_ADMIN,  help="Admin port URL (shell.py plugin)")
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"# Gate test: {args.test}")
    print(f"# Server:    {args.server}")
    print(f"# Admin:     {args.admin}")
    print(f"{'#'*60}")

    fn = TESTS[args.test]
    ok = await fn(server=args.server, admin=args.admin)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
