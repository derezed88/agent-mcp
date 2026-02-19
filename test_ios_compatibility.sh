#!/bin/bash
# Test iOS LLM app compatibility with OpenAI format

echo "=== iOS LLM App Compatibility Test ==="
echo ""
echo "This simulates what an iOS chat app expects from OpenAI API"
echo ""

SERVER="http://localhost:11434"

# Test 1: List models (what "Fetch Models" button does)
echo "Test 1: GET /v1/models (Fetch Models button)"
echo "----------------------------------------"
response=$(curl -s -w "\nHTTP_CODE:%{http_code}" "$SERVER/v1/models")
http_code=$(echo "$response" | grep "HTTP_CODE:" | cut -d: -f2)
body=$(echo "$response" | grep -v "HTTP_CODE:")

echo "HTTP Status: $http_code"
echo ""
echo "Response body:"
echo "$body" | python3 -m json.tool 2>&1 || echo "$body"
echo ""

# Validate OpenAI format
if echo "$body" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    assert data.get('object') == 'list', 'Missing or incorrect object field'
    assert isinstance(data.get('data'), list), 'Missing or incorrect data field'
    assert len(data['data']) > 0, 'No models in data array'

    # Check first model has required fields
    model = data['data'][0]
    assert 'id' in model, 'Model missing id field'
    assert model.get('object') == 'model', 'Model object field incorrect'
    assert 'created' in model, 'Model missing created field'

    print('VALID')
    sys.exit(0)
except Exception as e:
    print(f'INVALID: {e}')
    sys.exit(1)
" 2>&1; then
    echo "✓ Response format is OpenAI-compatible"
else
    echo "✗ Response format is NOT OpenAI-compatible"
    echo "  iOS apps expect: {\"object\": \"list\", \"data\": [...]}"
fi

echo ""
echo "---"
echo ""

# Test 2: Get specific model
echo "Test 2: GET /v1/models/gemini-2.5-flash (Get Model Details)"
echo "----------------------------------------"
response=$(curl -s -w "\nHTTP_CODE:%{http_code}" "$SERVER/v1/models/gemini-2.5-flash")
http_code=$(echo "$response" | grep "HTTP_CODE:" | cut -d: -f2)
body=$(echo "$response" | grep -v "HTTP_CODE:")

echo "HTTP Status: $http_code"
echo ""
echo "Response body:"
echo "$body" | python3 -m json.tool 2>&1 || echo "$body"
echo ""

if echo "$body" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    assert data.get('object') == 'model', 'Incorrect object field'
    assert 'id' in data, 'Missing id field'
    print('VALID')
except:
    print('INVALID')
    sys.exit(1)
" 2>&1 | grep -q "VALID"; then
    echo "✓ Model details format is correct"
else
    echo "✗ Model details format is incorrect"
fi

echo ""
echo "---"
echo ""

# Test 3: Chat completion with streaming
echo "Test 3: POST /v1/chat/completions (Send Message)"
echo "----------------------------------------"
echo "Testing with simple prompt..."

response=$(timeout 10 curl -s -X POST "$SERVER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "any-model-name",
    "messages": [{"role": "user", "content": "!model"}],
    "stream": true
  }' 2>&1 | head -5)

echo "First 5 lines of streaming response:"
echo "$response"
echo ""

if echo "$response" | grep -q "data:" && echo "$response" | grep -q "choices"; then
    echo "✓ Streaming response format looks correct"
else
    echo "✗ Streaming response format may have issues"
fi

echo ""
echo "---"
echo ""
echo "Summary:"
echo "--------"
echo "If all tests show ✓, your iOS app should work correctly."
echo ""
echo "Common iOS app issues:"
echo "1. App expects exact OpenAI format - custom formats fail"
echo "2. App validates HTTP status codes - must be 200 for success"
echo "3. App may retry on any JSON parsing error"
echo "4. Some apps cache the model list - may need to restart app"
echo ""
echo "To test from iOS app:"
echo "1. Restart main.py server: python main.py --llama-proxy"
echo "2. In iOS app, configure server: http://$(hostname -I | awk '{print $1}'):11434"
echo "3. Tap 'Fetch Models' - should show: grok-4, openai, gemini-2.5-flash, Win11"
echo "4. Send a message - should work without immediate disconnect"
