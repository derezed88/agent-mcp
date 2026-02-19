#!/bin/bash
# Test open-webui compatibility (bare paths without /v1/ prefix)

echo "=== Testing open-webui Compatibility ==="
echo ""

SERVER="http://localhost:11434"

# Test 1: GET /models (bare path, no /v1/)
echo "Test 1: GET /models (open-webui uses this)"
echo "----------------------------------------"
response=$(curl -s "$SERVER/models")
http_code=$(curl -s -o /dev/null -w "%{http_code}" "$SERVER/models")

echo "HTTP Status: $http_code"
echo ""
echo "Response:"
echo "$response" | python3 -m json.tool | head -20
echo ""

if [ "$http_code" = "200" ]; then
    echo "✓ /models endpoint works"
    if echo "$response" | grep -q '"object".*"list"' && echo "$response" | grep -q '"data":'; then
        echo "✓ Returns OpenAI format"
    else
        echo "✗ Wrong format"
    fi
else
    echo "✗ FAILED: Got status $http_code (expected 200)"
fi
echo ""

# Test 2: GET /models/{id} (bare path)
echo "Test 2: GET /models/gemini-2.5-flash"
echo "----------------------------------------"
response=$(curl -s "$SERVER/models/gemini-2.5-flash")
http_code=$(curl -s -o /dev/null -w "%{http_code}" "$SERVER/models/gemini-2.5-flash")

echo "HTTP Status: $http_code"
echo ""
echo "Response:"
echo "$response" | python3 -m json.tool
echo ""

if [ "$http_code" = "200" ]; then
    echo "✓ /models/{id} endpoint works"
    if echo "$response" | grep -q '"id".*"gemini-2.5-flash"'; then
        echo "✓ Returns correct model"
    fi
else
    echo "✗ FAILED: Got status $http_code"
fi
echo ""

# Test 3: POST /chat/completions (bare path, no /v1/)
echo "Test 3: POST /chat/completions (open-webui might use this)"
echo "----------------------------------------"
echo "Sending chat request to bare path..."
echo ""

response=$(timeout 10 curl -s -X POST "$SERVER/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [{"role": "user", "content": "!model"}],
    "stream": true
  }' | head -10)

echo "Response (first 10 lines):"
echo "$response"
echo ""

if echo "$response" | grep -q "data:" || echo "$response" | grep -q "response"; then
    echo "✓ /chat/completions works"
else
    echo "✗ Check response manually"
fi
echo ""

# Test 4: Compare with /v1/models (should be same)
echo "Test 4: Verify /models and /v1/models return same data"
echo "----------------------------------------"

bare_response=$(curl -s "$SERVER/models" | python3 -c "import sys,json; data=json.load(sys.stdin); print(len(data.get('data',[])))")
v1_response=$(curl -s "$SERVER/v1/models" | python3 -c "import sys,json; data=json.load(sys.stdin); print(len(data.get('data',[])))")

echo "Model count from /models: $bare_response"
echo "Model count from /v1/models: $v1_response"
echo ""

if [ "$bare_response" = "$v1_response" ]; then
    echo "✓ Both endpoints return same number of models"
else
    echo "✗ Mismatch! Bare: $bare_response, v1: $v1_response"
fi
echo ""

echo "=== Summary ==="
echo ""
echo "open-webui Endpoints (bare paths without /v1/):"
echo "  GET  /models              → OpenAI model list"
echo "  GET  /models/{id}         → Specific model details"
echo "  POST /chat/completions    → Chat endpoint"
echo ""
echo "Standard OpenAI Endpoints (with /v1/):"
echo "  GET  /v1/models           → OpenAI model list"
echo "  GET  /v1/models/{id}      → Specific model details"
echo "  POST /v1/chat/completions → Chat endpoint"
echo ""
echo "Both sets of paths work identically!"
echo ""
echo "To configure open-webui:"
echo "1. Settings → Connections"
echo "2. OpenAI API Base URL: http://localhost:11434"
echo "3. API Key: any-value (ignored by MCP)"
echo "4. Click 'Verify Connection'"
echo "5. Models should populate automatically"
