#!/bin/bash
# Test if immediate disconnects happen locally or only through Pinggy

echo "=== Testing Immediate Disconnect Issue ==="
echo ""

# Test 1: Local non-streaming request
echo "Test 1: Local non-streaming (should work fast)..."
time curl -s -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [{"role": "user", "content": "!model"}],
    "stream": false
  }' | jq -r '.choices[0].message.content // .error // .' | head -20
echo ""
echo "---"
echo ""

# Test 2: Local streaming request
echo "Test 2: Local streaming (watch for immediate output)..."
timeout 10 curl -s -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [{"role": "user", "content": "!help"}],
    "stream": true
  }' | head -30
echo ""
echo "---"
echo ""

# Test 3: Ollama format
echo "Test 3: Ollama format (api/generate)..."
time curl -s -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "prompt": "!model",
    "stream": true
  }' | head -20
echo ""
echo "---"
echo ""

# Test 4: Very simple prompt
echo "Test 4: Simple prompt through v1/chat/completions..."
time curl -v -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mcp",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": true
  }' 2>&1 | head -50

echo ""
echo "=== Check server logs for errors ==="
echo "Look for [LLAMA] entries and any Python tracebacks"
