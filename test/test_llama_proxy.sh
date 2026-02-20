#!/bin/bash
# Test script for llama proxy functionality in main.py

LLAMA_PROXY_URL="http://localhost:8765"

echo "=== Testing Llama Proxy Special Commands ==="
echo ""

echo "Test 1: !help command"
curl -s "$LLAMA_PROXY_URL/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "prompt": "!help"
  }' | jq -r '.response' || echo "Failed"
echo ""
echo "---"
echo ""

echo "Test 2: !model command (list models)"
curl -s "$LLAMA_PROXY_URL/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "prompt": "!model"
  }' | jq -r '.response' || echo "Failed"
echo ""
echo "---"
echo ""

echo "Test 3: OpenAI-compatible format with !help"
curl -s "$LLAMA_PROXY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [
      {"role": "user", "content": "!help"}
    ]
  }' | jq -r '.choices[0].message.content' || echo "Failed"
echo ""
echo "---"
echo ""

echo "Test 4: !autoaidb status"
curl -s "$LLAMA_PROXY_URL/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "prompt": "!autoaidb status"
  }' | jq -r '.response' || echo "Failed"
echo ""
echo "---"
echo ""

echo "Test 5: Health check"
curl -s "$LLAMA_PROXY_URL/health" | jq '.' || echo "Failed"
echo ""
echo "---"
echo ""

echo "=== Tests Complete ==="
echo ""
echo "To run the server in llama proxy mode:"
echo "  python main.py --llama-proxy --llama-host <remote-llama-server-ip>"
echo ""
echo "To test with a real llama server, update LLAMA_PROXY_URL and run:"
echo "  bash test_llama_proxy.sh"
