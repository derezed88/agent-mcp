#!/bin/bash
# Test script to verify llama proxy auto-gate rejection

echo "=== Testing Llama Proxy Auto-Gate Rejection ==="
echo ""
echo "This tests that gated tool calls are automatically rejected"
echo "for llama proxy clients (instead of hanging indefinitely)."
echo ""

# Ensure server is running
if ! curl -s http://localhost:11434/v1/models > /dev/null 2>&1; then
    echo "ERROR: Llama proxy not running on port 11434"
    echo "Start with: python main.py --llama-proxy"
    exit 1
fi

echo "✓ Llama proxy is running"
echo ""

# Test 1: Google Drive request (should auto-reject)
echo "Test 1: Google Drive read request (should auto-reject)"
echo "Sending: 'List files in my Google Drive'"
echo ""

RESPONSE=$(curl -s -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "grok-4",
    "messages": [{"role": "user", "content": "List files in my Google Drive"}],
    "stream": false
  }')

echo "Response:"
echo "$RESPONSE" | jq -r '.choices[0].message.content' | head -20
echo ""

# Check if response contains rejection message
if echo "$RESPONSE" | grep -q "TOOL CALL REJECTED"; then
    echo "✓ Gate was auto-rejected (correct behavior)"
else
    echo "✗ Gate rejection not detected (check if Drive gate is enabled)"
fi

echo ""
echo "=== Test Complete ==="
echo ""
echo "Expected behavior:"
echo "  1. LLM attempts google_drive tool call"
echo "  2. Gate auto-rejects (no user prompt)"
echo "  3. LLM receives detailed rejection message"
echo "  4. LLM responds with acknowledgment and alternatives"
echo ""
echo "To enable Drive access without gates:"
echo "  Send: !autogate drive read true"
