"""
Read-type special command tests for agent-mcp.

Tests every !command that returns status/info without side-effects and
does not involve tmux. Each test verifies the command succeeds and the
response contains expected content markers.

Usage:
    cd /path/to/mymcp
    source venv/bin/activate
    python test/test_special_commands_read.py [--url http://localhost:8767] [--api-key KEY]

Tests (30 total):
     1. !help
     2. !model
     3. !session
     4. !get_system_info
     5. !llm_list
     6. !llm_call
     7. !stream
     8. !tool_preview_length
     9. !maxctx
    10. !maxusers
    11. !sessiontimeout
    12. !gate_list
    13. !db_query_gate_status
    14. !limit_depth_list
    15. !limit_rate_list
    16. !limit_max_iteration_list
    17. !sysprompt_list self
    18. !sysprompt_read self
    19. !search_ddgs_gate_read  (no-arg status)
    20. !search_google_gate_read
    21. !search_tavily_gate_read
    22. !search_xai_gate_read
    23. !url_extract_gate_read
    24. !google_drive_gate_read
    25. !db_query_gate_read     (wildcard read status)
    26. !db_query_gate_write
    27. !gate_list_gate_read
    28. !session_gate_read
    29. !sleep_gate_read
    30. !limit_depth_list_gate_read
"""

import asyncio
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api_client import AgentClient

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


class TestRunner:
    def __init__(self, url: str, api_key: str = None):
        self.url = url
        self.api_key = api_key
        self.results = []
        # Single persistent client — all commands share one session
        self._client = AgentClient(url, client_id="api-rdtest", api_key=api_key)

    async def cmd(self, command: str, timeout: int = 15) -> str:
        return await self._client.send(command, timeout=timeout)

    def record(self, name: str, passed: bool, detail: str = ""):
        status = PASS if passed else FAIL
        detail_str = f" — {detail}" if detail else ""
        print(f"  [{status}] {name}{detail_str}")
        self.results.append((name, passed))

    def summary(self) -> bool:
        total = len(self.results)
        passed = sum(1 for _, ok in self.results if ok)
        print(f"\n{'='*55}")
        print(f"Results: {passed}/{total} passed")
        if passed < total:
            failed = [n for n, ok in self.results if not ok]
            print(f"Failed:  {', '.join(failed)}")
        return passed == total


# ---------------------------------------------------------------------------
# Helper: run a command and assert the output contains all expected strings
# ---------------------------------------------------------------------------
async def check(runner: TestRunner, name: str, command: str,
                must_contain: list[str], timeout: int = 15):
    try:
        result = await runner.cmd(command, timeout=timeout)
        ok = all(s.lower() in result.lower() for s in must_contain)
        snippet = result[:100].strip().replace("\n", " ")
        runner.record(name, ok, snippet)
    except Exception as e:
        runner.record(name, False, str(e))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_help(runner: TestRunner):
    await check(runner, "!help", "!help",
                ["available commands", "!model", "!reset"])

async def test_model(runner: TestRunner):
    await check(runner, "!model", "!model",
                ["model"])          # lists models; '*' marks active one

async def test_session(runner: TestRunner):
    await check(runner, "!session", "!session",
                ["api-rdtest"])     # our own session must appear

async def test_get_system_info(runner: TestRunner):
    await check(runner, "!get_system_info", "!get_system_info",
                ["local_time", "status"])

async def test_llm_list(runner: TestRunner):
    await check(runner, "!llm_list", "!llm_list",
                ["available llm models"])

async def test_llm_call(runner: TestRunner):
    await check(runner, "!llm_call", "!llm_call",
                ["tool_call_available"])

async def test_stream(runner: TestRunner):
    await check(runner, "!stream", "!stream",
                ["agent_call streaming"])

async def test_tool_preview_length(runner: TestRunner):
    await check(runner, "!tool_preview_length", "!tool_preview_length",
                ["tool preview length"])

async def test_maxctx(runner: TestRunner):
    await check(runner, "!maxctx", "!maxctx",
                ["agent_max_ctx"])

async def test_maxusers(runner: TestRunner):
    await check(runner, "!maxusers", "!maxusers",
                ["max_users"])

async def test_sessiontimeout(runner: TestRunner):
    await check(runner, "!sessiontimeout", "!sessiontimeout",
                ["session_idle_timeout"])

async def test_gate_list(runner: TestRunner):
    await check(runner, "!gate_list", "!gate_list",
                ["gate"])

async def test_db_query_gate_status(runner: TestRunner):
    await check(runner, "!db_query_gate_status", "!db_query_gate_status",
                ["db gate", "read", "write"])

async def test_limit_depth_list(runner: TestRunner):
    await check(runner, "!limit_depth_list", "!limit_depth_list",
                ["depth"])

async def test_limit_rate_list(runner: TestRunner):
    await check(runner, "!limit_rate_list", "!limit_rate_list",
                ["rate limits", "tool type"])

async def test_limit_max_iteration_list(runner: TestRunner):
    await check(runner, "!limit_max_iteration_list", "!limit_max_iteration_list",
                ["max tool iterations"])

async def test_sysprompt_list(runner: TestRunner):
    await check(runner, "!sysprompt_list self", "!sysprompt_list self",
                ["system prompt"])

async def test_sysprompt_read(runner: TestRunner):
    # Should return non-empty prompt content — just check something came back
    try:
        result = await runner.cmd("!sysprompt_read self", timeout=15)
        ok = len(result.strip()) > 20
        runner.record("!sysprompt_read self", ok,
                      result[:80].strip().replace("\n", " "))
    except Exception as e:
        runner.record("!sysprompt_read self", False, str(e))

# --- Gate read-status queries (no-arg = show current setting) ---
# These commands set gate ON/OFF when given an argument; without one they
# return the current status.  We call them with no extra arg and expect
# a response that mentions the tool name or "gate" / "auto-allow" / "gated".

async def _gate_read_status(runner: TestRunner, cmd_str: str):
    """Generic: run a _gate_read command with no arg and verify a status reply."""
    try:
        result = await runner.cmd(cmd_str, timeout=15)
        ok = len(result.strip()) > 0 and any(
            kw in result.lower()
            for kw in ("gate", "auto-allow", "gated", "true", "false", "on", "off",
                       "usage", "require", "allow")
        )
        runner.record(cmd_str, ok, result[:100].strip().replace("\n", " "))
    except Exception as e:
        runner.record(cmd_str, False, str(e))

async def test_search_ddgs_gate_read(runner):
    await _gate_read_status(runner, "!search_ddgs_gate_read")

async def test_search_google_gate_read(runner):
    await _gate_read_status(runner, "!search_google_gate_read")

async def test_search_tavily_gate_read(runner):
    await _gate_read_status(runner, "!search_tavily_gate_read")

async def test_search_xai_gate_read(runner):
    await _gate_read_status(runner, "!search_xai_gate_read")

async def test_url_extract_gate_read(runner):
    await _gate_read_status(runner, "!url_extract_gate_read")

async def test_google_drive_gate_read(runner):
    await _gate_read_status(runner, "!google_drive_gate_read")

async def test_db_query_gate_read(runner):
    await _gate_read_status(runner, "!db_query_gate_read")

async def test_db_query_gate_write(runner):
    await _gate_read_status(runner, "!db_query_gate_write")

async def test_gate_list_gate_read(runner):
    await _gate_read_status(runner, "!gate_list_gate_read")

async def test_session_gate_read(runner):
    await _gate_read_status(runner, "!session_gate_read")

async def test_sleep_gate_read(runner):
    await _gate_read_status(runner, "!sleep_gate_read")

async def test_limit_depth_list_gate_read(runner):
    await _gate_read_status(runner, "!limit_depth_list_gate_read")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

async def run_all(url: str, api_key: str = None):
    print(f"\nRead-type special command tests")
    print(f"  agent : {url}")
    print("=" * 55)

    runner = TestRunner(url, api_key)

    await test_help(runner)
    await test_model(runner)
    await test_session(runner)
    await test_get_system_info(runner)
    await test_llm_list(runner)
    await test_llm_call(runner)
    await test_stream(runner)
    await test_tool_preview_length(runner)
    await test_maxctx(runner)
    await test_maxusers(runner)
    await test_sessiontimeout(runner)
    await test_gate_list(runner)
    await test_db_query_gate_status(runner)
    await test_limit_depth_list(runner)
    await test_limit_rate_list(runner)
    await test_limit_max_iteration_list(runner)
    await test_sysprompt_list(runner)
    await test_sysprompt_read(runner)
    await test_search_ddgs_gate_read(runner)
    await test_search_google_gate_read(runner)
    await test_search_tavily_gate_read(runner)
    await test_search_xai_gate_read(runner)
    await test_url_extract_gate_read(runner)
    await test_google_drive_gate_read(runner)
    await test_db_query_gate_read(runner)
    await test_db_query_gate_write(runner)
    await test_gate_list_gate_read(runner)
    await test_session_gate_read(runner)
    await test_sleep_gate_read(runner)
    await test_limit_depth_list_gate_read(runner)

    return runner.summary()


def main():
    parser = argparse.ArgumentParser(
        description="Test read-type special commands on agent-mcp"
    )
    parser.add_argument("--url", default="http://localhost:8767", help="Agent base URL")
    parser.add_argument("--api-key", default=os.getenv("API_KEY", ""), help="API key")
    args = parser.parse_args()

    api_key = args.api_key or None
    success = asyncio.run(run_all(args.url, api_key))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
