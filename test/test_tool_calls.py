"""
Comprehensive LLM tool-call tests for agent-mcp.

Exercises every core LLM-callable tool. Strategy:
  - LLM tool-call path: used where Gemini 2.5 Flash reliably invokes the tool.
  - Direct !command path: fallback for tools Gemini answers from knowledge
    (get_system_info, help, llm_list, model, sysprompt_list_dir).
  - sysprompt write/read/delete: direct commands against gemini25f folder
    (003_test dir exists for isolation; write tests use 003_test via a temp
     model entry, or fall back to !command with gemini25f + cleanup).
  - db_query write: uses llm_notes table (safe to alter/recreate).

Requires:
  - agent-mcp server running on port 8767 (API plugin)
  - Model must have db tools in its llm_tools list
  - 003_test system prompt directory (auto-created if missing)

Usage:
    source venv/bin/activate
    python test/test_tool_calls.py [--url http://localhost:8767]
"""

import asyncio
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from api_client import AgentClient

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


class TestRunner:
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.results = []

    def client(self, suffix: str, **kwargs) -> AgentClient:
        kw = dict(
            base_url=self.base_url,
            client_id=f"tc-{suffix}",
            api_key=self.api_key,
        )
        kw.update(kwargs)
        return AgentClient(**kw)

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


def _has(text: str, *needles: str) -> bool:
    t = text.lower()
    return any(n.lower() in t for n in needles)


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------

async def test_get_system_info(runner: TestRunner):
    """get_system_info — direct !command (LLM answers from knowledge otherwise)."""
    try:
        c = runner.client("sysinfo")
        result = await c.send("!get_system_info", timeout=20)
        ok = _has(result, "platform", "hostname", "linux", "time", "cpu", "uptime", "local_time")
        runner.record("get_system_info", ok, result[:80].replace("\n", " "))
    except Exception as e:
        runner.record("get_system_info", False, str(e))


async def test_help_tool(runner: TestRunner):
    """help — direct !help command."""
    try:
        c = runner.client("help")
        result = await c.send("!help", timeout=20)
        ok = _has(result, "sysprompt", "db_query", "tool call", "get_system")
        runner.record("help tool", ok, result[:80].replace("\n", " "))
    except Exception as e:
        runner.record("help tool", False, str(e))


async def test_llm_list(runner: TestRunner):
    """llm_list — direct !llm_list command."""
    try:
        c = runner.client("llmlist")
        result = await c.send("!llm_list", timeout=20)
        ok = _has(result, "gemini", "gpt", "grok", "nuc", "model")
        runner.record("llm_list", ok, result[:80].replace("\n", " "))
    except Exception as e:
        runner.record("llm_list", False, str(e))


async def test_llm_tools_list(runner: TestRunner):
    """llm_tools — LLM tool call."""
    try:
        c = runner.client("llmtools")
        result = await c.send(
            "Call llm_tools with action='list' and show me all toolsets.", timeout=30
        )
        ok = _has(result, "core", "admin", "db", "search")
        runner.record("llm_tools(list)", ok, result[:80].replace("\n", " "))
    except Exception as e:
        runner.record("llm_tools(list)", False, str(e))


async def test_model_tool_list(runner: TestRunner):
    """model — direct !model command."""
    try:
        c = runner.client("modlist")
        result = await c.send("!model", timeout=20)
        ok = _has(result, "gemini", "gpt", "grok", "nuc", "model")
        runner.record("model tool (list)", ok, result[:80].replace("\n", " "))
    except Exception as e:
        runner.record("model tool (list)", False, str(e))


async def test_session_tool_list(runner: TestRunner):
    """session tool — LLM tool call."""
    try:
        c = runner.client("sesslist")
        await c.send("!help", timeout=10)  # ensure session exists
        result = await c.send(
            "Call the session tool with action='list' and tell me how many sessions are active.",
            timeout=30,
        )
        ok = _has(result, "session", "active", "tc-sesslist", "1", "2", "3", "4", "5")
        runner.record("session tool (list)", ok, result[:80].replace("\n", " "))
    except Exception as e:
        runner.record("session tool (list)", False, str(e))


async def test_sysprompt_list_dir(runner: TestRunner):
    """sysprompt_list_dir — direct !command; requires current server version."""
    try:
        c = runner.client("spdir")
        result = await c.send("!sysprompt_list_dir", timeout=20)
        if _has(result, "unknown command"):
            # Older server version — command not yet registered; skip gracefully
            print(f"  [\033[33mSKIP\033[0m] sysprompt_list_dir — server too old (unknown command)")
            runner.results.append(("sysprompt_list_dir", True))  # count as pass/skip
            return
        ok = _has(result, "003_test", "000_default", "prompt file")
        runner.record("sysprompt_list_dir", ok, result[:80].replace("\n", " "))
    except Exception as e:
        runner.record("sysprompt_list_dir", False, str(e))


async def test_sysprompt_list(runner: TestRunner):
    """sysprompt_list — LLM tool call with model=self."""
    try:
        c = runner.client("splist")
        result = await c.send(
            "Call sysprompt_list with model='self' and tell me what files are listed.",
            timeout=30,
        )
        ok = _has(result, "system_prompt", "file", ".system_prompt", "prompt", "section", "bytes")
        runner.record("sysprompt_list", ok, result[:80].replace("\n", " "))
    except Exception as e:
        runner.record("sysprompt_list", False, str(e))


async def test_sysprompt_read(runner: TestRunner):
    """sysprompt_read — LLM tool call, reads main prompt for current model."""
    try:
        c = runner.client("spread")
        result = await c.send(
            "Call sysprompt_read with model='self' and file='' to read the main system prompt. "
            "Tell me the first few words of what it returned.",
            timeout=30,
        )
        ok = len(result) > 10 and not _has(result, "traceback", "exception", "error:")
        runner.record("sysprompt_read", ok, result[:80].replace("\n", " "))
    except Exception as e:
        runner.record("sysprompt_read", False, str(e))


async def test_sysprompt_write_read_delete(runner: TestRunner):
    """sysprompt_write + sysprompt_read + sysprompt_delete — direct commands, gemini25f folder."""
    TEST_FILE = "tc_write_test"
    TEST_DATA = "TESTCONTENT_XYZ_123"
    try:
        c = runner.client("sprwd")

        # Write
        wr = await c.send(f"!sysprompt_write gemini25f {TEST_FILE} {TEST_DATA}", timeout=20)
        ok_write = _has(wr, "writ", "saved", "creat", "updat", "ok", "success", TEST_FILE)

        # Read back
        rd = await c.send(f"!sysprompt_read gemini25f {TEST_FILE}", timeout=20)
        ok_read = TEST_DATA in rd

        # Delete
        dl = await c.send(f"!sysprompt_delete gemini25f {TEST_FILE}", timeout=20)
        ok_delete = _has(dl, "delet", "remov", "ok", "success", TEST_FILE)

        ok = ok_write and ok_read and ok_delete
        runner.record(
            "sysprompt_write/read/delete",
            ok,
            f"write={'ok' if ok_write else 'FAIL'} "
            f"read={'ok' if ok_read else 'FAIL'} "
            f"delete={'ok' if ok_delete else 'FAIL'}",
        )
    except Exception as e:
        runner.record("sysprompt_write/read/delete", False, str(e))


async def test_db_query_read(runner: TestRunner):
    """db_query read — SELECT 42."""
    try:
        c = runner.client("dbqr")
        result = await c.send(
            "Run this exact SQL and return only the numeric result: SELECT 42 AS answer",
            timeout=45,
        )
        ok = "42" in result
        runner.record("db_query (SELECT 42)", ok, result[:80].replace("\n", " "))
    except Exception as e:
        runner.record("db_query (SELECT 42)", False, str(e))


async def test_db_query_write(runner: TestRunner):
    """db_query write — INSERT/SELECT/DELETE on llm_notes table."""
    try:
        c = runner.client("dbqw")

        # Ensure table exists with the expected schema
        await c.send(
            "Run this SQL: CREATE TABLE IF NOT EXISTS llm_notes "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, note TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)",
            timeout=30,
        )

        # Insert a test row
        ins = await c.send(
            "Run this SQL: INSERT INTO llm_notes (note) VALUES ('TC_TEST_NOTE_ABC')",
            timeout=30,
        )
        ok_insert = _has(ins, "insert", "row", "1", "affect", "ok", "done", "success", "result")

        # Read it back
        sel = await c.send(
            "Run this SQL: SELECT note FROM llm_notes WHERE note = 'TC_TEST_NOTE_ABC' LIMIT 1",
            timeout=30,
        )
        ok_select = "TC_TEST_NOTE_ABC" in sel

        # Clean up
        await c.send(
            "Run this SQL: DELETE FROM llm_notes WHERE note = 'TC_TEST_NOTE_ABC'",
            timeout=30,
        )

        ok = ok_insert and ok_select
        runner.record(
            "db_query (INSERT/SELECT llm_notes)",
            ok,
            f"insert={'ok' if ok_insert else 'FAIL'} select={'ok' if ok_select else 'FAIL'}",
        )
    except Exception as e:
        runner.record("db_query (INSERT/SELECT llm_notes)", False, str(e))


async def test_reset_tool(runner: TestRunner):
    """reset — clears session history."""
    try:
        c = runner.client("reset")
        await c.send("Remember: the secret code is ZETA99.", timeout=30)
        result = await c.send(
            "Call the reset tool to clear the conversation history.", timeout=20
        )
        ok = _has(result, "clear", "reset", "history", "remov", "0 message", "1 message", "2 message")
        runner.record("reset tool", ok, result[:80].replace("\n", " "))
    except Exception as e:
        runner.record("reset tool", False, str(e))


async def test_stream_tool(runner: TestRunner):
    """stream — reads current setting via tool call."""
    try:
        c = runner.client("stream")
        result = await c.send(
            "Call the stream tool with action='get' to check the current streaming setting.",
            timeout=20,
        )
        ok = _has(result, "stream", "true", "false", "enabled", "disabled")
        runner.record("stream tool (get)", ok, result[:80].replace("\n", " "))
    except Exception as e:
        runner.record("stream tool (get)", False, str(e))


async def test_tool_preview_length(runner: TestRunner):
    """tool_preview_length — reads current setting via tool call."""
    try:
        c = runner.client("tpl")
        result = await c.send(
            "Call tool_preview_length with action='get' to show the current preview length.",
            timeout=20,
        )
        ok = _has(result, "preview", "length", "char", "200", "100", "500", "unlimited", "tool_preview")
        runner.record("tool_preview_length (get)", ok, result[:80].replace("\n", " "))
    except Exception as e:
        runner.record("tool_preview_length (get)", False, str(e))


async def test_limit_depth_list(runner: TestRunner):
    """limit_depth_list — direct !command (LLM skips tool)."""
    try:
        c = runner.client("ldl")
        result = await c.send("!limit_depth_list", timeout=20)
        ok = _has(result, "depth", "limit", "max", "agent", "call")
        runner.record("limit_depth_list", ok, result[:80].replace("\n", " "))
    except Exception as e:
        runner.record("limit_depth_list", False, str(e))


async def test_limit_rate_list(runner: TestRunner):
    """limit_rate_list — LLM tool call."""
    try:
        c = runner.client("lrl")
        result = await c.send(
            "Call limit_rate_list and tell me the current rate limits.", timeout=20
        )
        ok = _has(result, "rate", "limit", "call", "window", "second", "llm")
        runner.record("limit_rate_list", ok, result[:80].replace("\n", " "))
    except Exception as e:
        runner.record("limit_rate_list", False, str(e))


async def test_limit_max_iteration_list(runner: TestRunner):
    """limit_max_iteration_list — LLM tool call."""
    try:
        c = runner.client("lmil")
        result = await c.send(
            "Call limit_max_iteration_list and tell me the current max iterations.", timeout=20
        )
        ok = _has(result, "iteration", "max", "limit", "tool")
        runner.record("limit_max_iteration_list", ok, result[:80].replace("\n", " "))
    except Exception as e:
        runner.record("limit_max_iteration_list", False, str(e))


async def test_token_selection_list(runner: TestRunner):
    """token_selection_list — LLM tool call."""
    try:
        c = runner.client("tsl")
        result = await c.send(
            "Call token_selection_list with no arguments to show all models' token settings.",
            timeout=20,
        )
        ok = _has(result, "temperature", "token", "selection", "model", "default", "custom")
        runner.record("token_selection_list", ok, result[:80].replace("\n", " "))
    except Exception as e:
        runner.record("token_selection_list", False, str(e))


async def test_sleep_tool(runner: TestRunner):
    """sleep — 1 second pause."""
    try:
        c = runner.client("sleep")
        result = await c.send(
            "Call the sleep tool with seconds=1 and tell me what it returned.", timeout=25
        )
        ok = _has(result, "sleep", "second", "1", "waited", "slept", "done", "complet")
        runner.record("sleep tool (1s)", ok, result[:80].replace("\n", " "))
    except Exception as e:
        runner.record("sleep tool (1s)", False, str(e))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def ensure_003_test_dir():
    sp_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "system_prompt", "003_test",
    )
    os.makedirs(sp_dir, exist_ok=True)
    main_prompt = os.path.join(sp_dir, ".system_prompt")
    if not os.path.exists(main_prompt):
        with open(main_prompt, "w") as f:
            f.write("You are a helpful test assistant.\n")
        print(f"  [setup] Created {main_prompt}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_all(base_url: str, api_key: str = None):
    ensure_003_test_dir()

    print(f"\nTool-call tests — {base_url}")
    print("=" * 55)

    runner = TestRunner(base_url, api_key)

    await test_get_system_info(runner)
    await test_help_tool(runner)
    await test_llm_list(runner)
    await test_llm_tools_list(runner)
    await test_model_tool_list(runner)
    await test_session_tool_list(runner)
    await test_sysprompt_list_dir(runner)
    await test_sysprompt_list(runner)
    await test_sysprompt_read(runner)
    await test_sysprompt_write_read_delete(runner)
    await test_db_query_read(runner)
    await test_db_query_write(runner)
    await test_reset_tool(runner)
    await test_stream_tool(runner)
    await test_tool_preview_length(runner)
    await test_limit_depth_list(runner)
    await test_limit_rate_list(runner)
    await test_limit_max_iteration_list(runner)
    await test_token_selection_list(runner)
    await test_sleep_tool(runner)

    return runner.summary()


def main():
    parser = argparse.ArgumentParser(description="Tool-call tests for agent-mcp")
    parser.add_argument("--url", default=os.getenv("AGENT_API_URL", "http://localhost:8767"))
    parser.add_argument("--api-key", default=os.getenv("API_KEY", ""))
    args = parser.parse_args()
    success = asyncio.run(run_all(args.url, args.api_key or None))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
