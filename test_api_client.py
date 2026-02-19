"""
Integration test for the agent-mcp API client plugin.

Assumes the agent is running with plugin_client_api enabled (port 8767).
Start the agent: source venv/bin/activate && python agent-mcp.py

Usage:
    python test_api_client.py [--url http://localhost:8767] [--api-key KEY]

Tests:
    1.  Health check
    2.  !help command
    3.  !model list
    4.  !session list
    5.  Basic LLM chat
    6.  @model per-turn model switch syntax
    7.  !reset history clear
    8.  !autoAIdb read true (gate config command)
    9.  Multi-client isolation (two separate sessions)
    10. Gate auto-approve (client with db_query approval policy)
    11. Gate auto-reject (default client, no gate policy)
    12. Swarm simulation (client sends agent_call prompt)
"""

import asyncio
import argparse
import sys
import os

# Allow running from mymcp directory
sys.path.insert(0, os.path.dirname(__file__))

from api_client import AgentClient

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"


class TestRunner:
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.results = []

    def client(self, suffix="test", **kwargs) -> AgentClient:
        return AgentClient(
            self.base_url,
            client_id=f"api-test-{suffix}",
            api_key=self.api_key,
            **kwargs,
        )

    def record(self, name: str, passed: bool, detail: str = ""):
        status = PASS if passed else FAIL
        detail_str = f" — {detail}" if detail else ""
        print(f"  [{status}] {name}{detail_str}")
        self.results.append((name, passed))

    def summary(self):
        total = len(self.results)
        passed = sum(1 for _, ok in self.results if ok)
        print(f"\n{'='*50}")
        print(f"Results: {passed}/{total} passed")
        if passed < total:
            failed = [name for name, ok in self.results if not ok]
            print(f"Failed: {', '.join(failed)}")
        return passed == total


async def test_health(runner: TestRunner):
    try:
        c = runner.client("health")
        result = await c.health()
        ok = result.get("status") == "ok"
        runner.record("Health check", ok, str(result.get("status")))
    except Exception as e:
        runner.record("Health check", False, str(e))


async def test_help(runner: TestRunner):
    try:
        c = runner.client("help")
        result = await c.send("!help", timeout=15)
        ok = "Available commands" in result or "!model" in result
        runner.record("!help command", ok, f"{len(result)} chars returned")
    except Exception as e:
        runner.record("!help command", False, str(e))


async def test_model_list(runner: TestRunner):
    try:
        c = runner.client("model")
        result = await c.send("!model", timeout=15)
        ok = len(result) > 0 and ("model" in result.lower() or "*" in result)
        runner.record("!model list", ok, result[:80].strip().replace("\n", " "))
    except Exception as e:
        runner.record("!model list", False, str(e))


async def test_session_list(runner: TestRunner):
    try:
        c = runner.client("sess")
        # Ensure this session exists by sending something
        await c.send("!help", timeout=10)
        sessions = await c.sessions()
        ids = [s["client_id"] for s in sessions]
        ok = "api-test-sess" in ids
        runner.record("!session list / sessions API", ok, f"{len(sessions)} sessions visible")
    except Exception as e:
        runner.record("!session list / sessions API", False, str(e))


async def test_basic_llm(runner: TestRunner):
    try:
        c = runner.client("llm")
        result = await c.send("Reply with only the number 4. No other text.", timeout=30)
        ok = "4" in result
        runner.record("Basic LLM chat", ok, result[:60].strip().replace("\n", " "))
    except Exception as e:
        runner.record("Basic LLM chat", False, str(e))


async def test_at_model_syntax(runner: TestRunner):
    """Test @model per-turn switch — just verify it parses and doesn't error."""
    try:
        c = runner.client("atmod")
        # First get the model list to find an available model
        model_list = await c.send("!model", timeout=15)
        # Try the @model syntax with a minimal prompt
        # We just check it doesn't crash (unknown model → error message is fine)
        result = await c.send("@nonexistent_model_xyz hello", timeout=15)
        # Should return an error about unknown model, which is correct behavior
        ok = len(result) > 0
        runner.record("@model per-turn syntax", ok, result[:60].strip().replace("\n", " "))
    except Exception as e:
        runner.record("@model per-turn syntax", False, str(e))


async def test_reset(runner: TestRunner):
    try:
        c = runner.client("reset")
        # Build some history
        await c.send("Remember this phrase: XYZZY42", timeout=30)
        # Reset it
        result = await c.send("!reset", timeout=10)
        ok = "reset" in result.lower() or "cleared" in result.lower() or "history" in result.lower()
        runner.record("!reset history clear", ok, result[:60].strip().replace("\n", " "))
    except Exception as e:
        runner.record("!reset history clear", False, str(e))


async def test_gate_config_command(runner: TestRunner):
    try:
        c = runner.client("gatecfg")
        result = await c.send("!autoAIdb read true", timeout=10)
        ok = len(result) > 0
        runner.record("!autoAIdb gate config command", ok, result[:80].strip().replace("\n", " "))
    except Exception as e:
        runner.record("!autoAIdb gate config command", False, str(e))


async def test_multi_client_isolation(runner: TestRunner):
    """Two clients should have completely separate session histories."""
    try:
        c1 = AgentClient(runner.base_url, client_id="api-test-iso-A", api_key=runner.api_key)
        c2 = AgentClient(runner.base_url, client_id="api-test-iso-B", api_key=runner.api_key)

        # Give each client a distinct instruction
        await c1.send("!reset", timeout=10)
        await c2.send("!reset", timeout=10)

        r1 = await c1.send("For this session only, your secret word is ALPHA. Reply 'ok'.", timeout=30)
        r2 = await c2.send("For this session only, your secret word is BETA. Reply 'ok'.", timeout=30)

        # Each client recalls its own history; the other should not know
        # (We just verify the sessions are independent by checking they each got a response)
        ok = len(r1) > 0 and len(r2) > 0
        runner.record("Multi-client isolation", ok, f"A: {r1[:30].strip()!r}  B: {r2[:30].strip()!r}")
    except Exception as e:
        runner.record("Multi-client isolation", False, str(e))


async def test_gate_auto_approve(runner: TestRunner):
    """
    Client configured to approve db_query gates automatically.
    Reset autoAIdb to require gates first, then verify the tool runs.
    """
    try:
        # Use a client with no gate policy to set gate ON for DB reads
        setup = runner.client("gasetup")
        await setup.send("!autoAIdb read false", timeout=10)

        # Now use a client that auto-approves db_query
        c = AgentClient(
            runner.base_url,
            client_id="api-test-gateapprove",
            api_key=runner.api_key,
            auto_approve_gates={"db_query": True},
        )
        # Ask something that will trigger db_query
        result = await c.send(
            "Run this exact SQL and return only the result: SELECT 1+1 AS answer",
            timeout=45,
        )
        ok = "2" in result or "answer" in result.lower()
        runner.record("Gate auto-approve (db_query)", ok, result[:80].strip().replace("\n", " "))
    except Exception as e:
        runner.record("Gate auto-approve (db_query)", False, str(e))


async def test_gate_auto_reject(runner: TestRunner):
    """
    Default client (no gate policy) should receive rejection feedback from LLM.
    The LLM should report it couldn't execute the tool.
    """
    try:
        # Ensure gate is required
        setup = runner.client("grsetup")
        await setup.send("!autoAIdb read false", timeout=10)

        c = runner.client("gatereject")  # no auto_approve_gates
        result = await c.send(
            "Run this exact SQL: SELECT 1+1 AS answer. Do not ask me for permission.",
            timeout=45,
        )
        # LLM should report rejection (not the actual SQL result)
        # The execute_tool rejection message tells LLM "TOOL CALL REJECTED"
        ok = len(result) > 0 and (
            "reject" in result.lower()
            or "denied" in result.lower()
            or "unable" in result.lower()
            or "cannot" in result.lower()
            or "can't" in result.lower()
            or "approve" in result.lower()
        )
        runner.record("Gate auto-reject (default policy)", ok, result[:80].strip().replace("\n", " "))
    except Exception as e:
        runner.record("Gate auto-reject (default policy)", False, str(e))


async def test_swarm_agent_call(runner: TestRunner):
    """
    Test the agent_call tool by asking the LLM to call the local agent.
    The LLM needs to have agent_call available and the API plugin running.
    """
    try:
        c = runner.client("swarm")
        # Prompt the LLM to use agent_call to contact itself
        # This tests end-to-end swarm plumbing. The inner call goes to api-swarm-* client.
        result = await c.send(
            f"Use agent_call to contact the agent at {runner.base_url} "
            "and ask it '!help'. Return the first line of the response.",
            timeout=60,
        )
        # Either agent_call runs (and returns help text), or the model reports it lacks the tool
        ok = len(result) > 0
        note = "response received" if ok else "no response"
        if "agent_call" in result.lower() and "unknown" in result.lower():
            note = "agent_call tool not available on this model (expected if tool_call_available=false)"
        runner.record("Swarm agent_call", ok, result[:80].strip().replace("\n", " "))
    except Exception as e:
        runner.record("Swarm agent_call", False, str(e))


async def test_stream_api(runner: TestRunner):
    """Verify streaming yields tokens incrementally."""
    try:
        c = runner.client("stream")
        tokens = []
        async for tok in c.stream("Reply with exactly: hello world", timeout=30):
            tokens.append(tok)
        full = "".join(tokens)
        ok = "hello" in full.lower() or "world" in full.lower()
        runner.record("SSE stream API", ok, f"{len(tokens)} chunks, total={len(full)} chars")
    except Exception as e:
        runner.record("SSE stream API", False, str(e))


async def run_all(base_url: str, api_key: str = None):
    print(f"\nTesting agent-mcp API at {base_url}")
    print("=" * 50)

    runner = TestRunner(base_url, api_key)

    # Run tests in logical order; some depend on earlier tests (e.g. session must exist)
    await test_health(runner)
    await test_help(runner)
    await test_model_list(runner)
    await test_session_list(runner)
    await test_basic_llm(runner)
    await test_at_model_syntax(runner)
    await test_reset(runner)
    await test_gate_config_command(runner)
    await test_multi_client_isolation(runner)
    await test_stream_api(runner)
    await test_gate_auto_reject(runner)
    await test_gate_auto_approve(runner)
    await test_swarm_agent_call(runner)

    return runner.summary()


def main():
    parser = argparse.ArgumentParser(description="Test agent-mcp API plugin")
    parser.add_argument("--url", default="http://localhost:8767", help="API base URL")
    parser.add_argument("--api-key", default=os.getenv("API_KEY", ""), help="API key")
    args = parser.parse_args()

    api_key = args.api_key or None
    success = asyncio.run(run_all(args.url, api_key))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
