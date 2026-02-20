#!/bin/bash
# Test Enchanted app compatibility (hybrid v1/api endpoints)

echo "=== Testing Enchanted App Compatibility ==="
echo ""

SERVER="http://localhost:11434"

# Test 1: HEAD /v1/ (health check)
echo "Test 1: HEAD /v1/ (Enchanted health check)"
echo "----------------------------------------"
http_code=$(curl -s -o /dev/null -w "%{http_code}" -X HEAD "$SERVER/v1/")
echo "HTTP Status: $http_code"
if [ "$http_code" = "200" ]; then
    echo "✓ Health check passed"
else
    echo "✗ Health check failed (expected 200, got $http_code)"
fi
echo ""

# Test 2: GET /v1/ (health check with body)
echo "Test 2: GET /v1/ (Health check)"
echo "----------------------------------------"
response=$(curl -s "$SERVER/v1/")
echo "$response" | python3 -m json.tool
echo ""
if echo "$response" | grep -q '"status".*"ok"'; then
    echo "✓ Returns health status"
else
    echo "✗ Missing status field"
fi
echo ""

# Test 3: GET /v1/api/tags (hybrid model list)
echo "Test 3: GET /v1/api/tags (Enchanted model list)"
echo "----------------------------------------"
response=$(curl -s "$SERVER/v1/api/tags")
echo "$response" | python3 -m json.tool
echo ""

# Check format
if echo "$response" | grep -q '"models":'; then
    echo "✓ Returns Ollama format (models array)"

    # Verify it's Ollama format, not OpenAI
    if echo "$response" | grep -q '"object".*"list"'; then
        echo "✗ WARNING: Returned OpenAI format instead of Ollama format"
    else
        echo "✓ Correctly using Ollama format for hybrid path"
    fi
else
    echo "✗ Wrong format - expected {\"models\": [...]}"
fi
echo ""

# Test 4: POST /v1/api/generate (hybrid generate endpoint)
echo "Test 4: POST /v1/api/generate (Enchanted generate)"
echo "----------------------------------------"
echo "Sending request with prompt: '!model'"
echo ""

response=$(timeout 5 curl -s -X POST "$SERVER/v1/api/generate" \
  -H "Content-Type: application/json" \
  -H "User-Agent: Enchanted/1.0" \
  -d '{
    "model": "test",
    "prompt": "!model",
    "stream": true
  }' | head -5)

echo "First 5 lines:"
echo "$response"
echo ""

if echo "$response" | grep -q '"response":' && echo "$response" | grep -q '"done":'; then
    echo "✓ Ollama NDJSON format detected"
else
    echo "✗ Expected Ollama format with 'response' and 'done' fields"
fi
echo ""

# Test 5: POST /v1/api/chat (hybrid chat endpoint)
echo "Test 5: POST /v1/api/chat (Enchanted chat)"
echo "----------------------------------------"

response=$(timeout 5 curl -s -X POST "$SERVER/v1/api/chat" \
  -H "Content-Type: application/json" \
  -H "User-Agent: Enchanted/1.0" \
  -d '{
    "model": "test",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": true
  }' | head -5)

echo "First 5 lines:"
echo "$response"
echo ""

if echo "$response" | grep -q '"response":'; then
    echo "✓ Ollama format for chat"
else
    echo "✗ Expected Ollama format"
fi
echo ""

# Test 6: Verify pure OpenAI path still works
echo "Test 6: GET /v1/models (Pure OpenAI path - should be different)"
echo "----------------------------------------"

response=$(curl -s "$SERVER/v1/models")
echo "$response" | python3 -m json.tool | head -15
echo ""

if echo "$response" | grep -q '"object".*"list"' && echo "$response" | grep -q '"data":'; then
    echo "✓ OpenAI format for pure v1/models path"
else
    echo "✗ Expected OpenAI format with 'object' and 'data' fields"
fi
echo ""

echo "=== Summary ==="
echo ""
echo "Enchanted App Hybrid Paths:"
echo "  HEAD /v1/              → Health check (200 OK)"
echo "  GET  /v1/              → {\"status\": \"ok\", \"version\": \"...\"}"
echo "  GET  /v1/api/tags      → Ollama format: {\"models\": [...]}"
echo "  POST /v1/api/generate  → Ollama NDJSON streaming"
echo "  POST /v1/api/chat      → Ollama NDJSON streaming"
echo ""
echo "Standard OpenAI Paths (unchanged):"
echo "  GET  /v1/models        → OpenAI format: {\"object\": \"list\", \"data\": [...]}"
echo "  POST /v1/chat/completions → OpenAI SSE streaming"
echo ""
echo "Detection Logic:"
echo "  - User-Agent 'enchanted' → Ollama format"
echo "  - Path contains 'api/'   → Ollama format"
echo "  - Path is 'v1/models'    → OpenAI format (no 'api/')"
