#!/bin/bash
# Test Ollama client compatibility

echo "=== Testing Ollama Client Compatibility ==="
echo ""

SERVER="http://localhost:11434"

# Test 1: Version endpoint
echo "Test 1: GET /api/version"
echo "----------------------------------------"
response=$(curl -s "$SERVER/api/version")
echo "$response" | python3 -m json.tool 2>&1 || echo "$response"
echo ""

# Test 2: List models (api/tags)
echo "Test 2: GET /api/tags (Ollama model list)"
echo "----------------------------------------"
response=$(curl -s "$SERVER/api/tags")
echo "$response" | python3 -m json.tool
echo ""

# Test 3: Generate with streaming (Ollama format)
echo "Test 3: POST /api/generate (Ollama generate endpoint)"
echo "----------------------------------------"
echo "Sending request with prompt: '!model'"
echo ""

timeout 10 curl -s -X POST "$SERVER/api/generate" \
  -H "Content-Type: application/json" \
  -H "User-Agent: ollama/0.1.0" \
  -d '{
    "model": "test-model",
    "prompt": "!model",
    "stream": true
  }' | head -10

echo ""
echo "---"
echo ""

# Test 4: Chat endpoint (Ollama chat format)
echo "Test 4: POST /api/chat (Ollama chat endpoint)"
echo "----------------------------------------"
echo "Sending chat message: 'hello'"
echo ""

timeout 10 curl -s -X POST "$SERVER/api/chat" \
  -H "Content-Type: application/json" \
  -H "User-Agent: ollama/0.1.0" \
  -d '{
    "model": "test-model",
    "messages": [
      {"role": "user", "content": "hello"}
    ],
    "stream": true
  }' | head -10

echo ""
echo "---"
echo ""

# Test 5: Verify Ollama-specific format
echo "Test 5: Checking Ollama response format"
echo "----------------------------------------"

response=$(timeout 5 curl -s -X POST "$SERVER/api/generate" \
  -H "Content-Type: application/json" \
  -H "User-Agent: ollama/0.1.0" \
  -d '{
    "model": "test",
    "prompt": "!help",
    "stream": true
  }' | head -3)

echo "First 3 response lines:"
echo "$response"
echo ""

if echo "$response" | grep -q '"response":' && echo "$response" | grep -q '"done":'; then
    echo "✓ Ollama format detected (has 'response' and 'done' fields)"
else
    echo "✗ Not Ollama format - may be OpenAI format or error"
fi

echo ""
echo "---"
echo ""

# Compare with OpenAI endpoint
echo "Test 6: Compare with OpenAI endpoint"
echo "----------------------------------------"

openai_response=$(timeout 5 curl -s -X POST "$SERVER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [{"role": "user", "content": "!help"}],
    "stream": true
  }' | head -3)

echo "OpenAI format (first 3 lines):"
echo "$openai_response"
echo ""

if echo "$openai_response" | grep -q 'data:' && echo "$openai_response" | grep -q 'delta'; then
    echo "✓ OpenAI format detected (SSE with 'delta' field)"
else
    echo "✗ Unexpected format"
fi

echo ""
echo "=== Summary ==="
echo ""
echo "Ollama endpoints:"
echo "  - /api/version       - Server version"
echo "  - /api/tags          - List models"
echo "  - /api/generate      - Generate completions"
echo "  - /api/chat          - Chat completions"
echo ""
echo "Response format:"
echo "  - Ollama: NDJSON with 'response', 'done', 'model', 'created_at'"
echo "  - OpenAI: SSE with 'delta', 'choices', 'finish_reason'"
echo ""
echo "Client detection:"
echo "  - User-Agent: 'ollama' → Ollama format"
echo "  - Path: /api/* → Ollama format"
echo "  - Path: /v1/* → OpenAI format"
