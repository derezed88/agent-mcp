#!/bin/bash
# Test that the model parameter from external clients is ignored

echo "=== Testing Model Parameter Ignore Feature ==="
echo ""
echo "This test sends a model name that doesn't exist in LLM_REGISTRY."
echo "Expected: Request should succeed using DEFAULT_MODEL instead of failing."
echo ""

SERVER="http://localhost:11434"

# First check if server is responding
echo "Checking if server is running..."
if ! curl -s --max-time 2 "$SERVER/api/tags" > /dev/null 2>&1; then
    echo "ERROR: Server not responding at $SERVER"
    echo "Start with: python main.py --llama-proxy"
    exit 1
fi
echo "✓ Server is running"
echo ""

# Test with non-existent model name
echo "Test: Sending request with non-existent model name..."
echo "Model in request: Qwen2.5-Coder-7B-Instruct-abliterated-Q4_K_M.gguf"
echo ""

response=$(timeout 10 curl -s "$SERVER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-Coder-7B-Instruct-abliterated-Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "!model"}],
    "stream": true
  }' 2>&1)

echo "Response:"
echo "$response" | head -20
echo ""

# Check for error
if echo "$response" | grep -q "Unknown model"; then
    echo "✗ FAILED: Model parameter was NOT ignored!"
    echo "   The server tried to use the model from the request."
    exit 1
elif echo "$response" | grep -q "error"; then
    echo "✗ FAILED: Got an error (but not 'Unknown model')"
    echo "   Check server logs for details."
    exit 1
else
    echo "✓ SUCCESS: Model parameter was ignored!"
    echo "   Server used session model instead of request model."
fi
