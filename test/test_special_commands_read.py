"""
Read-type special command tests for agent-mcp.

Tests every !command that returns status/info without side-effects and
does not involve tmux. Each test verifies the command succeeds and the
response contains expected content markers.

Usage:
    cd /path/to/mymcp
    source venv/bin/activate
    python test/test_special_commands_read.py [--url http://localhost:8767] [--api-key KEY]

Tests (14 total):
     1. !help
     2. !model
     3. !session
     4. !get_system_info
     5. !llm_list
     6. !llm_tools list
     7. !model_cfg list
     8. !model_cfg read <default_model>
     9. !sysprompt_cfg list
    10. !sysprompt_cfg read self
    11. !config_cfg list
    12. !limits_cfg list
    13. !limits_cfg read rate
    14. !limits_cfg read depth
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

async def test_llm_tools_list(runner: TestRunner):
    await check(runner, "!llm_tools list", "!llm_tools list",
                ["toolset"])

async def test_model_cfg_list(runner: TestRunner):
    await check(runner, "!model_cfg list", "!model_cfg list",
                ["model"])

async def test_model_cfg_read(runner: TestRunner):
    # Read the first available model — we don't know the name, so just check it returns fields
    await check(runner, "!model_cfg read", "!model_cfg read",
                ["model_id"])

async def test_sysprompt_cfg_list(runner: TestRunner):
    await check(runner, "!sysprompt_cfg list", "!sysprompt_cfg list",
                ["system prompt", "prompt"])

async def test_sysprompt_cfg_read(runner: TestRunner):
    # Should return non-empty prompt content — just check something came back
    try:
        result = await runner.cmd("!sysprompt_cfg read self", timeout=15)
        ok = len(result.strip()) > 20
        runner.record("!sysprompt_cfg read self", ok,
                      result[:80].strip().replace("\n", " "))
    except Exception as e:
        runner.record("!sysprompt_cfg read self", False, str(e))

async def test_config_cfg_list(runner: TestRunner):
    await check(runner, "!config_cfg list", "!config_cfg list",
                ["config"])

async def test_limits_cfg_list(runner: TestRunner):
    await check(runner, "!limits_cfg list", "!limits_cfg list",
                ["limit"])

async def test_limits_cfg_read_rate(runner: TestRunner):
    await check(runner, "!limits_cfg read rate", "!limits_cfg read rate",
                ["rate"])

async def test_limits_cfg_read_depth(runner: TestRunner):
    await check(runner, "!limits_cfg read depth", "!limits_cfg read depth",
                ["depth"])


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
    await test_llm_tools_list(runner)
    await test_model_cfg_list(runner)
    await test_model_cfg_read(runner)
    await test_sysprompt_cfg_list(runner)
    await test_sysprompt_cfg_read(runner)
    await test_config_cfg_list(runner)
    await test_limits_cfg_list(runner)
    await test_limits_cfg_read_rate(runner)
    await test_limits_cfg_read_depth(runner)

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
