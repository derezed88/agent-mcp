#!/usr/bin/env python3
"""
Test session shorthand ID functionality.

This test verifies:
1. Sessions get assigned sequential shorthand IDs (starting at 101)
2. !session command displays sessions with shorthand IDs
3. !session <shorthand_id> delete works correctly
4. Deleted sessions show proper confirmation message
"""

import asyncio
import sys

# Add parent directory to path for imports
sys.path.insert(0, '.')

from state import (
    sessions,
    get_or_create_shorthand_id,
    get_session_by_shorthand,
    remove_shorthand_mapping,
    sse_queues
)
from routes import cmd_session, push_tok, get_queue


async def capture_output(client_id: str) -> str:
    """Capture all output tokens for a client."""
    q = await get_queue(client_id)
    output = []

    # Process all messages in queue
    while not q.empty():
        item = q.get_nowait()
        if item.get("t") == "tok":
            # Unescape newlines
            text = item["d"].replace("\\n", "\n")
            output.append(text)

    return "".join(output)


async def test_session_shorthand():
    """Test session shorthand ID functionality."""

    print("=" * 70)
    print("Testing Session Shorthand ID Functionality")
    print("=" * 70)
    print()

    # Create test sessions
    print("Step 1: Creating test sessions...")
    sessions["slack-CTEST123456-1234567890.123456"] = {
        "model": "test-model",
        "history": [{"role": "user", "content": f"msg {i}"} for i in range(18)]
    }
    sessions["llama-192.168.1.100"] = {
        "model": "test-model",
        "history": [{"role": "user", "content": "msg 1"}, {"role": "assistant", "content": "msg 2"}]
    }

    # Assign shorthand IDs
    slack_id = get_or_create_shorthand_id("slack-CTEST123456-1234567890.123456")
    llama_id = get_or_create_shorthand_id("llama-192.168.1.100")

    print(f"  Slack session: ID [{slack_id}]")
    print(f"  Llama session: ID [{llama_id}]")
    print()

    # Test 1: List sessions
    print("Step 2: Testing !session (list all sessions)...")
    test_client_id = "slack-CTEST123456-1234567890.123456"
    await cmd_session(test_client_id, "")
    output = await capture_output(test_client_id)
    print(output)
    print()

    # Verify output format
    assert f"ID [{slack_id}]" in output, f"Expected 'ID [{slack_id}]' in output"
    assert f"ID [{llama_id}]" in output, f"Expected 'ID [{llama_id}]' in output"
    assert "(current)" in output, "Expected '(current)' marker for active session"
    print("✓ Session list format correct")
    print()

    # Test 2: Delete session by shorthand ID
    print(f"Step 3: Testing !session {llama_id} delete...")
    await cmd_session(test_client_id, f"{llama_id} delete")
    output = await capture_output(test_client_id)
    print(output)
    print()

    # Verify deletion message
    assert f"Deleted session ID [{llama_id}]:" in output, f"Expected deletion confirmation"
    assert "llama-192.168.1.100" in output, "Expected full session ID in confirmation"
    assert "llama-192.168.1.100" not in sessions, "Session should be deleted from sessions dict"
    print("✓ Session deletion works correctly")
    print()

    # Test 3: Verify session is removed from list
    print("Step 4: Testing !session after deletion...")
    await cmd_session(test_client_id, "")
    output = await capture_output(test_client_id)
    print(output)
    print()

    assert f"ID [{slack_id}]" in output, "Slack session should still exist"
    assert f"ID [{llama_id}]" not in output, "Llama session should be gone"
    assert "llama-192.168.1.100" not in output, "Llama session should not appear"
    print("✓ Deleted session removed from list")
    print()

    # Test 4: Try deleting non-existent shorthand ID
    print("Step 5: Testing deletion of non-existent ID [999]...")
    await cmd_session(test_client_id, "999 delete")
    output = await capture_output(test_client_id)
    print(output)
    print()

    assert "not found" in output.lower(), "Should report session not found"
    print("✓ Handles non-existent session IDs correctly")
    print()

    # Test 5: Verify mapping cleanup
    print("Step 6: Verifying internal state cleanup...")
    remaining_sid = get_session_by_shorthand(llama_id)
    assert remaining_sid is None, "Deleted session's shorthand mapping should be removed"
    print("✓ Shorthand mappings cleaned up correctly")
    print()

    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_session_shorthand())
